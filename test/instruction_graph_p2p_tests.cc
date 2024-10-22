#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "instruction_graph_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;

namespace acc = celerity::access;


TEMPLATE_TEST_CASE_SIG("buffer subranges are sent and received to satisfy push and await-push commands",
    "[instruction_graph_generator][instruction-graph][p2p]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const node_id peer_nid = 1 - local_nid;
	CAPTURE(local_nid, peer_nid);

	test_utils::idag_test_context ictx(2 /* nodes */, local_nid, 1 /* devices */);

	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	const auto reader_tid = ictx.device_compute(test_range).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto writer = all_instrs.select_unique<device_kernel_instruction_record>("writer");
	const auto send = all_instrs.select_unique<send_instruction_record>();
	const auto recv = all_instrs.select_unique<receive_instruction_record>();
	const auto reader = all_instrs.select_unique<device_kernel_instruction_record>("reader");

	const transfer_id expected_trid(reader_tid, buf.get_id(), no_reduction_id);

	// we send exactly the part of the buffer that our node has written
	REQUIRE(writer->access_map.size() == 1);
	const auto& write_access = writer->access_map.front();
	CHECK(send->dest_node_id == peer_nid);
	CHECK(send->transfer_id == expected_trid);
	CHECK(subrange(send->offset_in_buffer, send->send_range) == write_access.accessed_region_in_buffer);
	CHECK(send->element_size == sizeof(int));

	// a pilot is attached to the send
	const auto pilot = ictx.query_outbound_pilots();
	CHECK(pilot.is_unique());
	CHECK(pilot->to == peer_nid);
	CHECK(pilot->message.transfer_id == send->transfer_id);
	CHECK(pilot->message.id == send->message_id);

	// we receive exactly the part of the buffer that our node has _not_ written
	REQUIRE(reader->access_map.size() == 1);
	const auto& read_access = reader->access_map.front();
	CHECK(recv->transfer_id == expected_trid);
	CHECK(recv->element_size == sizeof(int));
	CHECK(region_intersection(write_access.accessed_region_in_buffer, recv->requested_region).empty());
	CHECK(region_union(write_access.accessed_region_in_buffer, recv->requested_region) == read_access.accessed_region_in_buffer);

	// the logical dependencies are (writer -> send, writer -> reader, recv -> reader)
	CHECK(writer.transitive_successors_across<copy_instruction_record>().contains(send));
	CHECK(recv.transitive_successors_across<copy_instruction_record>().contains(reader));
	CHECK(send.is_concurrent_with(recv));
}

TEMPLATE_TEST_CASE_SIG("send and receive instructions are split on multi-device systems to allow computation-communication overlap",
    "[instruction_graph_generator][instruction-graph][p2p]", ((int Dims), Dims), 1, 2, 3) {
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const node_id peer_nid = 1 - local_nid;
	CAPTURE(local_nid, peer_nid);

	test_utils::idag_test_context ictx(2 /* nodes */, 0 /* my nid */, 2 /* devices */);
	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	const auto reader_tid = ictx.device_compute(test_range).name("reader").read(buf, test_utils::access::reverse_one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	const transfer_id expected_trid(reader_tid, buf.get_id(), no_reduction_id);

	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	CHECK(all_writers.count() == 2);
	CHECK(all_writers.all_concurrent());

	const auto all_sends = all_instrs.select_all<send_instruction_record>();
	CHECK(all_sends.count() == 2);
	CHECK(all_sends.all_concurrent());

	CHECK(all_pilots.count() == all_sends.count());

	// there is one send per writer instruction (with coherence copies in between)
	for(const auto& send : all_sends.iterate()) {
		CAPTURE(send);

		const auto associated_writer =
		    intersection_of(send.transitive_predecessors_across<copy_instruction_record>(), all_writers).assert_unique<device_kernel_instruction_record>();
		REQUIRE(associated_writer->access_map.size() == 1);
		const auto& write = associated_writer->access_map.front();

		// the send operates on a (host) allocation that is distinct from the (device) allocation that associated_writer writes to, but both instructions need
		// to access the same buffer subrange
		const auto send_box = box(subrange(send->offset_in_buffer, send->send_range));
		CHECK(send_box == write.accessed_region_in_buffer);
		CHECK(send->element_size == sizeof(int));
		CHECK(send->transfer_id == expected_trid);

		CHECK(all_pilots.count(send->dest_node_id, expected_trid, send_box) == 1);
	}

	const auto split_recv = all_instrs.select_unique<split_receive_instruction_record>();
	const auto all_await_recvs = all_instrs.select_all<await_receive_instruction_record>();
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");

	// There is one split-receive instruction which binds the allocation to a transfer id, because we don't know the shape / stride of incoming messages until
	// we receive pilots at runtime, and messages might either match our awaited subregions (and complete them independently), cover both (and need the
	// bounding-box allocation), or anything in between.
	CHECK(split_recv.successors().contains(all_await_recvs));
	CHECK(split_recv->requested_region == region_union(all_await_recvs[0]->received_region, all_await_recvs[1]->received_region));
	CHECK(split_recv->element_size == sizeof(int));
	CHECK(region_intersection(all_await_recvs[0]->received_region, all_await_recvs[1]->received_region).empty());

	CHECK(all_await_recvs.count() == 2); // one per device
	CHECK(all_await_recvs.all_concurrent());
	CHECK(all_readers.all_concurrent());

	// there is one reader per await-receive instruction (with coherence copies in between)
	for(const auto& await_recv : all_await_recvs.iterate()) {
		CAPTURE(await_recv);

		const auto associated_reader =
		    intersection_of(await_recv.transitive_successors_across<copy_instruction_record>(), all_readers).assert_unique<device_kernel_instruction_record>();
		REQUIRE(associated_reader->access_map.size() == 1);
		const auto& read = associated_reader->access_map.front();

		CHECK(await_recv->received_region == read.accessed_region_in_buffer);
		CHECK(await_recv->transfer_id == expected_trid);
	}
}

// see also: "overlapping requirements between chunks of an oversubscribed kernel generate one copy per reader-set" from memory tests.
TEMPLATE_TEST_CASE_SIG("overlapping requirements generate split-receives with one await per reader-set",
    "[instruction_graph_generator][instruction-graph][p2p]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 4 /* devices */);
	auto buf = ictx.create_buffer<int>(test_range);
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::all()).submit();
	ictx.device_compute(test_range).name("reader").read(buf, test_utils::access::make_neighborhood<Dims>(1)).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// We are N1, so we receive the entire buffer from N0.
	const auto split_recv = all_instrs.select_unique<split_receive_instruction_record>();

	// We do not know the send-split, so we create a receive-split that awaits subregions used by single chunks separately from subregions used by multiple
	// chunks (the neighborhood overlap) in order to un-block any instruction as soon as their requirements are fulfilled.
	const auto all_await_recvs = all_instrs.select_all<await_receive_instruction_record>();
	CHECK(split_recv.successors().contains(all_await_recvs));

	// await-receives for the same split-receive must never intersect, but their union must cover the entire received region
	region<3> awaited_region;
	for(const auto& await : all_await_recvs.iterate()) {
		CHECK(region_intersection(awaited_region, await->received_region).empty());
		awaited_region = region_union(awaited_region, await->received_region);
	}
	CHECK(awaited_region == split_recv->requested_region);

	// Each reader instruction sources its input data from multiple await-receive instructions, and by extension, the await-receives operating on the overlap
	// service multiple reader chunks.
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");
	for(const auto& reader : all_readers.iterate()) {
		const auto all_pred_awaits = reader.transitive_predecessors_across<copy_instruction_record>().select_all<await_receive_instruction_record>();
		CHECK(all_pred_awaits.count() > 1);

		// Sanity check: Each reader chunk depends on await-receives for the subranges it reads
		region<3> pred_awaited_region;
		for(const auto& pred_await : all_pred_awaits.iterate()) {
			pred_awaited_region = region_union(pred_awaited_region, pred_await->received_region);
		}

		REQUIRE(reader->access_map.size() == 1);
		const auto& read = reader->access_map.front();
		CHECK(read.buffer_id == buf.get_id());
		CHECK(read.accessed_region_in_buffer == pred_awaited_region);
	}
}

TEST_CASE("an await-push of disconnected subregions does not allocate their bounding-box", "[instruction_graph_generator][instruction-graph][p2p]") {
	test_utils::idag_test_context ictx(2 /* nodes */, 1 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer(range(1024));
	const auto acc_first = acc::fixed(subrange<1>(0, 1));
	const auto acc_last = acc::fixed(subrange<1>(1023, 1));
	ictx.host_task(range(1)).name("writer").discard_write(buf, acc::all()).submit(); // remote only
	ictx.host_task(range(2)).name("reader").read(buf, acc_first).read(buf, acc_last).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// since the individual elements (acc_first, acc_last) are bound to different accessors, we can (and should) allocate them separately to avoid allocating
	// the large bounding box. This means we have two allocations, with one receive each.
	const auto all_allocs = all_instrs.select_all<alloc_instruction_record>();
	CHECK(all_allocs.count() == 2);
	const auto all_recvs = all_instrs.select_all<receive_instruction_record>();
	CHECK(all_recvs.count() == 2);
	CHECK(all_recvs.all_concurrent());

	for(const auto& recv : all_recvs.iterate()) {
		CAPTURE(recv);
		CHECK(recv->requested_region.get_area() == 1);

		const auto alloc = recv.predecessors().select_unique<alloc_instruction_record>();
		CHECK(alloc->buffer_allocation->buffer_id == buf.get_id());
		CHECK(region(alloc->buffer_allocation->box) == recv->requested_region);
	}

	const auto reader = all_instrs.select_unique<host_task_instruction_record>("reader");
	CHECK(reader.predecessors() == all_recvs);
}

TEST_CASE("transfers on huge buffers are split into boxes with communicator-compatible strides", "[instruction_graph_generator][instruction-graph][p2p]") {
	constexpr size_t small_extent = 4096;
	constexpr size_t max_extent = INT_MAX;
	constexpr size_t overflow_extent = 2 * max_extent + 1;
	constexpr size_t huge_extent = 10 * max_extent;
	constexpr static node_id sender = 0;
	constexpr static node_id receiver = 1;

	const node_id local_nid = GENERATE(values<node_id>({sender, receiver}));
	CAPTURE(local_nid);

	const auto test_buffer_transfer = [&](const auto& buffer_range, const auto& transfer_box, auto expected_send_boxes) {
		constexpr int dimensions = std::remove_reference_t<decltype(buffer_range)>::dimensions;

		CAPTURE(buffer_range, transfer_box, expected_send_boxes);

		test_utils::idag_test_context ictx(2, local_nid, 1 /* devices */);
		auto buf = ictx.create_buffer(buffer_range);
		ictx.host_task(range(1)).name("writer").discard_write(buf, acc::fixed(transfer_box.get_subrange())).submit(); // write on sender only
		ictx.host_task(range(2)).name("reader").read(buf, acc::fixed(transfer_box.get_subrange())).submit();          // read on sender and receiver
		ictx.finish();

		const auto all_instrs = ictx.query_instructions();
		const auto all_sends = all_instrs.select_all<send_instruction_record>();
		const auto all_pilots = ictx.query_outbound_pilots();
		const auto all_recvs = all_instrs.select_all<receive_instruction_record>();

		if(local_nid == sender) {
			CHECK(all_sends.count() == expected_send_boxes.size());
			CHECK(all_pilots.count() == expected_send_boxes.size());
			CHECK(all_recvs.count() == 0);

			decltype(expected_send_boxes) actual_send_boxes;
			for(const auto& send : all_sends.iterate()) {
				const auto send_box = box(subrange(send->offset_in_buffer, send->send_range));
				actual_send_boxes.push_back(box_cast<dimensions>(send_box));
				CHECK(all_pilots.count([&](const outbound_pilot& pilot) { return pilot.message.box == send_box; }) == 1);
			}

			std::sort(expected_send_boxes.begin(), expected_send_boxes.end(), box_coordinate_order());
			std::sort(actual_send_boxes.begin(), actual_send_boxes.end(), box_coordinate_order());
			CHECK(actual_send_boxes == expected_send_boxes);
		} else {
			CHECK(all_sends.count() == 0);
			CHECK(all_pilots.count() == 0);
			CHECK(all_recvs.count() == 1); // the strides are only known at runtime and computed from inbound pilots
		}
	};

	SECTION("small transfer on a large 1D buffer remains contiguous") {
		const auto offset = 42;
		test_buffer_transfer(
		    /* buffer_range */ range(huge_extent),
		    /* transfer sr */ box(subrange<1>(offset, small_extent)),
		    /* expected send boxes */ box_vector<1>{{subrange<1>(offset, small_extent)}});
	}

	SECTION("maximum-sized transfer on a large 1D buffer remains contiguous") {
		// communicator can adjust the base pointer to transfer `max_extent` regardless of allocation size
		const auto offset = 999999;
		test_buffer_transfer(
		    /* buffer_range */ range(huge_extent),
		    /* transfer sr */ box(subrange<1>(offset, max_extent)),
		    /* expected send boxes */ box_vector<1>{{subrange<1>(offset, max_extent)}});
	}

	SECTION("overflowing transfer on a large 1D buffer is split into chunks") {
		const auto offset = 1234;
		test_buffer_transfer(
		    /* buffer_range */ range(huge_extent),
		    /* transfer sr */ box(subrange<1>(offset, overflow_extent)),
		    /* expected send boxes */
		    box_vector<1>{{
		        subrange<1>(offset, max_extent),
		        subrange<1>(offset + max_extent, max_extent),
		        subrange<1>(offset + 2 * max_extent, overflow_extent - 2 * max_extent),
		    }});
	}

	SECTION("transfer of > 2Gi elements on a 2D buffer without overflows is contiguous") {
		// communicator can adjust the base pointer to compensate for a large extent in dimension 0
		const size_t offset_0 = overflow_extent;
		const size_t offset_1 = 42;
		const size_t range_0 = max_extent;
		const size_t range_1 = max_extent - 1337;
		test_buffer_transfer(
		    /* buffer range */ range(huge_extent, max_extent),
		    /* transfer sr */ box(subrange(id(offset_0, offset_1), range(range_0, range_1))),
		    /* expected send boxes */ box_vector<2>{{subrange(id(offset_0, offset_1), range(range_0, range_1))}});
	}

	SECTION("transfer on 2D buffer with huge extent in dimension 0 is split in dimension 0") {
		const size_t offset_1 = 42;
		const size_t range_1 = max_extent - 1337;
		test_buffer_transfer(
		    /* buffer range */ range(huge_extent, max_extent),
		    /* transfer sr */ box(subrange(id(0, offset_1), range(overflow_extent, range_1))),
		    /* expected send boxes */
		    box_vector<2>{{
		        subrange(id(0, offset_1), range(max_extent, range_1)),
		        subrange(id(max_extent, offset_1), range(max_extent, range_1)),
		        subrange(id(2 * max_extent, offset_1), range(overflow_extent - 2 * max_extent, range_1)),
		    }});
	}

	SECTION("transfer on 2D buffer with huge extent in dimension 1 is split into rows") {
		const size_t offset_0 = 10;
		test_buffer_transfer(
		    /* buffer range */ range(small_extent, huge_extent),
		    /* transfer sr */ box(subrange(id(offset_0, 0), range(5, small_extent))),
		    /* expected send boxes */
		    box_vector<2>{{
		        subrange(id(offset_0, 0), range(1, small_extent)),
		        subrange(id(offset_0 + 1, 0), range(1, small_extent)),
		        subrange(id(offset_0 + 2, 0), range(1, small_extent)),
		        subrange(id(offset_0 + 3, 0), range(1, small_extent)),
		        subrange(id(offset_0 + 4, 0), range(1, small_extent)),
		    }});
	}

	SECTION("transfer on 2D buffer with huge extent in both dimensions is split into partial rows") {
		test_buffer_transfer(
		    /* buffer range */ range(huge_extent, huge_extent),
		    /* transfer sr */ box(subrange(id(0, 0), range(3, overflow_extent))),
		    /* expected send boxes */
		    box_vector<2>{{
		        // row 0
		        subrange(id(0, 0), range(1, max_extent)),
		        subrange(id(0, max_extent), range(1, max_extent)),
		        subrange(id(0, 2 * max_extent), range(1, overflow_extent - 2 * max_extent)),
		        // row 1
		        subrange(id(1, 0), range(1, max_extent)),
		        subrange(id(1, max_extent), range(1, max_extent)),
		        subrange(id(1, 2 * max_extent), range(1, overflow_extent - 2 * max_extent)),
		        // row 1
		        subrange(id(2, 0), range(1, max_extent)),
		        subrange(id(2, max_extent), range(1, max_extent)),
		        subrange(id(2, 2 * max_extent), range(1, overflow_extent - 2 * max_extent)),
		    }});
	}

	SECTION("transfer on 3D buffer with huge extent in dimension 1 is split into planes") {
		const auto offset = id(1, 4, 9);
		test_buffer_transfer(
		    /* buffer range */ range(2 * small_extent, huge_extent, 4 * small_extent),
		    /* transfer sr */ box(subrange(offset, range(4, small_extent, 2 * small_extent))),
		    /* expected send boxes */
		    box_vector<3>{{
		        subrange(offset + id(0, 0, 0), range(1, small_extent, 2 * small_extent)),
		        subrange(offset + id(1, 0, 0), range(1, small_extent, 2 * small_extent)),
		        subrange(offset + id(2, 0, 0), range(1, small_extent, 2 * small_extent)),
		        subrange(offset + id(3, 0, 0), range(1, small_extent, 2 * small_extent)),
		    }});
	}
}
