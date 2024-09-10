#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "instruction_graph_test_utils.h"
#include "test_utils.h"

#include <set>


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::experimental;

namespace acc = celerity::access;


TEST_CASE("reductions are equivalent to writes on a single-node single-device setup", "[instruction_graph_generator][instruction-graph][reduction]") {
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto writer = all_instrs.select_unique<device_kernel_instruction_record>("writer");
	const auto reader = all_instrs.select_unique<device_kernel_instruction_record>("reader");
	CHECK(writer.successors() == reader);
	CHECK(reader.predecessors() == writer);

	// there is no local (eager) reduce-instruction generated on the reduction-write, nor do we get a reduction_command to generate a global (lazy)
	// reduce-instruction between nodes.
	CHECK(all_instrs.count<send_instruction_record>() == 0);
	CHECK(all_instrs.count<gather_receive_instruction_record>() == 0);
	CHECK(all_instrs.count<copy_instruction_record>() == 0);
	CHECK(all_instrs.count<reduce_instruction_record>() == 0);
}

TEST_CASE("single-node single-device reductions locally include the initial buffer value"
          "[instruction_graph_generator][instruction-graph][reduction]") //
{
	// almost the same setup as above, but now we include the current buffer value, which generates a local reduce-instruction.
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<1>(1, true /* host initialized */);
	// initialize the buffer - the resulting host_task_instruction is concurrent with the `writer` kernel because they act on different memories
	ictx.host_task(buf.get_range()).name("init").discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute(range(256)).name("writer").reduce(buf, true /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto writer = all_instrs.select_unique<device_kernel_instruction_record>("writer");
	const auto init_buffer = all_instrs.select_unique<host_task_instruction_record>("init");
	CHECK(writer.is_concurrent_with(init_buffer));

	// initialization happens in a buffer allocation, from which we must copy into the gather allocation
	const auto gather_from_init = init_buffer.successors().assert_unique<copy_instruction_record>();
	CHECK(gather_from_init->origin == copy_instruction_record::copy_origin::gather);
	CHECK(gather_from_init->copy_region == region(box<3>(zeros, ones)));

	// we also directly perform a device-to-host copy into the gather allocation
	const auto gather_from_writer = writer.successors().assert_unique<copy_instruction_record>();
	CHECK(gather_from_writer->origin == copy_instruction_record::copy_origin::gather);
	CHECK(gather_from_writer->copy_region == region(box<3>(zeros, ones)));
	CHECK(gather_from_writer.is_concurrent_with(gather_from_init));

	const auto gather_alloc = intersection_of(gather_from_init.predecessors(), gather_from_writer.predecessors()).assert_unique<alloc_instruction_record>();
	CHECK(gather_alloc->origin == alloc_instruction_record::alloc_origin::gather);
	CHECK(gather_alloc->size_bytes == 2 * sizeof(int)); // 1 for the initial value, 1 for the "writer" contribution

	// the local reduction combines both values and writes into the (final) host buffer allocation
	const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(local_reduce->scope == reduce_instruction_record::reduction_scope::local);
	CHECK(local_reduce->num_source_values == 2);
	CHECK(local_reduce->source_allocation_id == gather_from_init->dest_allocation_id);
	CHECK(local_reduce->source_allocation_id == gather_from_writer->dest_allocation_id);

	const auto reader = all_instrs.select_unique<device_kernel_instruction_record>("reader");
	CHECK(reader.transitive_predecessors_across<copy_instruction_record>().contains(local_reduce));
}

TEST_CASE("reduction accesses on a single-node multi-device setup generate local reduce-instructions only",
    "[instruction_graph_generator][instruction-graph][reduction]") //
{
	const auto num_devices = GENERATE(values<size_t>({2, 4}));
	CAPTURE(num_devices);

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices);

	auto buf = ictx.create_buffer<int, 1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// partial results are written on the device
	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	CHECK(all_writers.count() == num_devices);
	CHECK(all_writers.all_concurrent());

	// partial results are written to the appropriate positions in a (host) gather buffer
	const auto all_gather_copies = all_writers.successors().select_all<copy_instruction_record>();
	CHECK(all_gather_copies.count() == num_devices);
	CHECK(all_gather_copies.all_concurrent());

	// a local reduction does not need to fill its gather buffer (a feature for gather-receive), because we know the number of partial results in advance
	CHECK(all_instrs.count<fill_identity_instruction_record>() == 0);

	// the local gather buffer is a single allocation
	const auto gather_alloc = all_gather_copies.predecessors().select_unique<alloc_instruction_record>();
	CHECK(gather_alloc->origin == alloc_instruction_record::alloc_origin::gather);
	CHECK(gather_alloc->size_bytes == num_devices * sizeof(int));

	for(const auto& gather_copy : all_gather_copies.iterate()) {
		CAPTURE(gather_copy);
		CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);
		CHECK(gather_copy->source_layout == region_layout(strided_layout(box<3>(zeros, ones))));
		CHECK(gather_copy->dest_allocation_id == gather_alloc->allocation_id);

		// the order of reduction inputs must be deterministic because the reduction operator is not necessarily associative
		const auto writer = intersection_of(all_writers, gather_copy.predecessors());
		CHECK(gather_copy->dest_layout == region_layout(linearized_layout(writer->device_id * sizeof(int))));
	}

	const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(local_reduce.predecessors().contains(all_gather_copies));

	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");
	CHECK(all_readers.all_concurrent());
	CHECK(local_reduce.transitive_successors_across<copy_instruction_record>().contains(all_readers));
}

TEST_CASE("reduction accesses on a multi-node single-device setup generate global reduce-instructions only",
    "[instruction_graph_generator][instruction-graph][reduction]") //
{
	const auto num_nodes = GENERATE(values<size_t>({2, 4}));
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	CAPTURE(num_nodes, local_nid);

	test_utils::idag_test_context ictx(num_nodes, local_nid, 1 /* num devices */);

	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	const auto reader_tid = ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	// there is exactly one (global) reduce-instruction per node.
	const auto reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(reduce->scope == reduce_instruction_record::reduction_scope::global);
	CHECK(reduce->num_source_values == num_nodes);
	CHECK(reduce->buffer_id == buf.get_id());
	CHECK(reduce->box == box<3>(zeros, ones));

	// we send partial results to peers - this operation anti-depends on the reduce operation, which will overwrite its buffer
	const auto all_sends_to_peers = all_instrs.select_all<send_instruction_record>();
	CHECK(all_sends_to_peers.all_concurrent());

	std::set<node_id> peers_sent_to;
	for(const auto& send_to_peer : all_sends_to_peers.iterate()) {
		CHECK(reduce.predecessors().contains(send_to_peer));
		CHECK(send_to_peer->offset_in_buffer == zeros);
		CHECK(send_to_peer->send_range == ones);
		CHECK(send_to_peer->dest_node_id != local_nid);
		CHECK(peers_sent_to.find(send_to_peer->dest_node_id) == peers_sent_to.end());
		peers_sent_to.insert(send_to_peer->dest_node_id);
		CHECK(send_to_peer->transfer_id.rid == reduce->reduction_id);
		CHECK(send_to_peer->transfer_id.bid == buf.get_id());
	}
	CHECK(peers_sent_to.size() == num_nodes - 1);

	// fill the gather-buffer before initiating the gather-receive because if the peer decides to not send a payload (but an empty pilot), the gather-recv can
	// simply skip writing to the appropriate position in the gather buffer.
	const auto fill_identity = all_instrs.select_unique<fill_identity_instruction_record>();
	CHECK(fill_identity->reduction_id == reduce->reduction_id);
	CHECK(fill_identity->num_values == num_nodes);

	// the global gather buffer is a single allocation
	const auto gather_alloc = fill_identity.predecessors().select_unique<alloc_instruction_record>();
	CHECK(gather_alloc->origin == alloc_instruction_record::alloc_origin::gather);
	CHECK(gather_alloc->size_bytes == num_nodes * sizeof(int));
	CHECK(gather_alloc->allocation_id == fill_identity->allocation_id);

	// we (gather-) copy the local partial result to the appropriate position in the gather buffer
	const auto gather_copy = reduce.predecessors().select_unique<copy_instruction_record>();
	CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);
	CHECK(gather_copy->source_layout == region_layout(strided_layout(box<3>(zeros, ones))));
	CHECK(gather_copy->dest_layout == region_layout(linearized_layout(local_nid * sizeof(int))));
	CHECK(gather_copy->copy_region == region(box<3>(zeros, ones)));

	// we gather-receive from all peers - this will _not_ write to the position `local_nid`
	const auto gather_recv = all_instrs.select_unique<gather_receive_instruction_record>();
	CHECK(reduce.predecessors().contains(gather_recv));
	CHECK(gather_recv->gather_box == box<3>(zeros, ones));
	CHECK(gather_recv->num_nodes == num_nodes);
	CHECK(gather_recv->allocation_id == gather_copy->dest_allocation_id);
	CHECK(gather_recv->transfer_id.bid == buf.get_id());
	CHECK(gather_recv->transfer_id.consumer_tid == reader_tid);
	CHECK(gather_recv->transfer_id.rid == reduce->reduction_id);
}

TEST_CASE("reduction accesses on a multi-node multi-device setup generate global and local reduce-instructions",
    "[instruction_graph_generator][instruction-graph][reduction]") //
{
	const size_t num_nodes = GENERATE(values<size_t>({2, 4}));
	const auto local_nid = GENERATE(values<node_id>({0, 1}));
	const size_t num_devices = GENERATE(values<size_t>({2, 4}));
	CAPTURE(num_nodes, local_nid, num_devices);

	test_utils::idag_test_context ictx(num_nodes, local_nid, num_devices);

	auto buf = ictx.create_buffer<1>(1);
	ictx.device_compute(range(256)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	const auto reader_tid = ictx.device_compute(range(256)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.count<reduce_instruction_record>() == 2);

	// At the time of writing this test, the local reduction is generated eagerly and writes to a buffer host allocation. Its results are then fed into the
	// global reduction once that command is compiled.
	const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>(
	    [](const reduce_instruction_record& rinstr) { return rinstr.scope == reduce_instruction_record::reduction_scope::local; });
	CHECK(local_reduce->buffer_id == buf.get_id());
	CHECK(local_reduce->box == box<3>(zeros, ones));
	CHECK(local_reduce->num_source_values == num_devices);

	// there is a distinct gather allocation into which all partial results from devices are copied.
	const auto gather_copies_to_local = local_reduce.predecessors().select_all<copy_instruction_record>();
	const auto local_gather_alloc = gather_copies_to_local.predecessors().select_unique<alloc_instruction_record>();
	CHECK(local_gather_alloc->origin == alloc_instruction_record::alloc_origin::gather);
	CHECK(local_gather_alloc->size_bytes == num_devices * sizeof(int));
	CHECK(local_gather_alloc->num_chunks == num_devices);

	for(const auto& gather_copy : gather_copies_to_local.iterate()) {
		CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);
		CHECK(gather_copy->copy_region == region(box<3>(zeros, ones)));
		CHECK(gather_copy->dest_allocation_id == local_gather_alloc->allocation_id);
		CHECK(gather_copy->dest_allocation_id == local_reduce->source_allocation_id);
	}

	// the global reduction has a single local contribution (the locally-reduced partial result from devices), and `num_nodes - 1` contributions from peers.
	const auto global_reduce = all_instrs.select_unique<reduce_instruction_record>(
	    [](const reduce_instruction_record& rinstr) { return rinstr.scope == reduce_instruction_record::reduction_scope::global; });
	CHECK(global_reduce->buffer_id == buf.get_id());
	CHECK(global_reduce->box == box<3>(zeros, ones));
	CHECK(global_reduce->num_source_values == num_nodes);

	// we transmit the output of the local reduction, i.e. the partial result from local_nid, to our peers
	const auto all_sends_to_peers = all_instrs.select_all<send_instruction_record>();
	CHECK(all_sends_to_peers.all_concurrent());

	std::set<node_id> peers_sent_to;
	for(const auto& send_to_peer : all_sends_to_peers.iterate()) {
		CHECK(global_reduce.predecessors().contains(send_to_peer));
		CHECK(send_to_peer.predecessors() == local_reduce);
		CHECK(send_to_peer->offset_in_buffer == zeros);
		CHECK(send_to_peer->send_range == ones);
		CHECK(send_to_peer->dest_node_id != local_nid);
		CHECK(peers_sent_to.find(send_to_peer->dest_node_id) == peers_sent_to.end());
		peers_sent_to.insert(send_to_peer->dest_node_id);
		CHECK(send_to_peer->transfer_id.rid == global_reduce->reduction_id);
		CHECK(send_to_peer->transfer_id.bid == buf.get_id());
	}
	CHECK(peers_sent_to.size() == num_nodes - 1);

	// since we don't know how many non-empty contributions we receive from peers, we initialize the gather buffer with the reduction identity
	const auto fill_identity = all_instrs.select_unique<fill_identity_instruction_record>();
	CHECK(fill_identity->reduction_id == global_reduce->reduction_id);
	CHECK(fill_identity->num_values == num_nodes);

	// we allocate one slot in the gather buffer for every node in the system
	const auto global_gather_alloc = fill_identity.predecessors().select_unique<alloc_instruction_record>();
	CHECK(global_gather_alloc->origin == alloc_instruction_record::alloc_origin::gather);
	CHECK(global_gather_alloc->size_bytes == num_nodes * sizeof(int));
	CHECK(global_gather_alloc->allocation_id == fill_identity->allocation_id);

	// the gather-receive writes (at most) `num_nodes - 1` values into the gather buffer at positions corresponding to the peer node id
	const auto gather_recv = all_instrs.select_unique<gather_receive_instruction_record>();
	CHECK(global_reduce.predecessors().contains(gather_recv));
	CHECK(gather_recv.predecessors() == fill_identity);
	CHECK(gather_recv->gather_box == box<3>(zeros, ones));
	CHECK(gather_recv->num_nodes == num_nodes);
	CHECK(gather_recv->transfer_id.bid == buf.get_id());
	CHECK(gather_recv->transfer_id.consumer_tid == reader_tid);
	CHECK(gather_recv->transfer_id.rid == global_reduce->reduction_id);

	// the local reduction could directly write to the global gather buffer, so we do not explicitly enumerate any copy-instructions between the two.
	CHECK(global_reduce.transitive_predecessors_across<copy_instruction_record>().contains(local_reduce));
	CHECK(gather_recv.is_concurrent_with(local_reduce));
}

TEST_CASE("local reductions can be initialized to a buffer value that is not present locally", "[instruction_graph_generator][instruction-graph][reduction]") {
	const size_t num_nodes = 2; // we need a remote writer
	const node_id my_nid = GENERATE(values<node_id>({0, 1}));
	const auto num_devices = 1; // we generate a local reduction even for a single device because there's a remote contribution
	CAPTURE(my_nid);

	constexpr auto item_1_accesses_0 = [](const chunk<1> ck) { return subrange(id(0), range(ck.offset[0] + ck.range[0] > 1 ? 1 : 0)); };

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);
	auto buf = ictx.create_buffer<int>(range(1));
	ictx.device_compute(range<1>(num_nodes)).name("init").discard_write(buf, item_1_accesses_0).submit();
	const auto reduce_tid = ictx.device_compute(range<1>(1)).name("writer").reduce(buf, true /* include_current_buffer_value */).submit();
	// local reductions are generated eagerly, even if there is no subsequent reader
	ictx.finish();

	// the generated push / await-push pair is not in preparation of a reduction command (since there is none in this example), instead, the kernel starting the
	// reduction defines an implicit read-requirement on the buffer on the reduction-initializer node (node 0), and push / await-push commands are generated
	// accordingly to establish coherence.
	const auto expected_trid = transfer_id(reduce_tid, buf.get_id(), no_reduction_id);

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	if(my_nid == 0) {
		// we are the receiver / reducer node
		const auto recv = all_instrs.select_unique<receive_instruction_record>();
		CHECK(recv->transfer_id == expected_trid);
		CHECK(recv->requested_region == region(box<3>(zeros, ones)));
		CHECK(recv->element_size == sizeof(int));

		const auto writer = all_instrs.select_unique<device_kernel_instruction_record>("writer");
		CHECK(writer->access_map.empty()); // we have reductions, not (regular) accesses
		REQUIRE(writer->reduction_map.size() == 1);
		const auto& red_acc = writer->reduction_map.front();
		CHECK(red_acc.buffer_id == buf.get_id());
		CHECK(red_acc.accessed_box_in_buffer == box<3>(zeros, ones));

		const auto gather_copies = all_instrs.select_all<copy_instruction_record>();
		for(const auto& copy : gather_copies.iterate()) {
			CHECK(copy->origin == copy_instruction_record::copy_origin::gather);
			CHECK(copy->buffer_id == buf.get_id());
			CHECK(copy->copy_region == region(box<3>(zeros, ones)));
		}

		const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();
		CHECK(local_reduce->box == box<3>(zeros, ones));
		CHECK(local_reduce->reduction_id != no_reduction_id);
		CHECK(local_reduce->reduction_id == red_acc.reduction_id);
		CHECK(local_reduce->num_source_values == 2); // the received remote init value + our contribution
		CHECK(local_reduce.predecessors() == gather_copies);
	} else {
		// we are the initializer / sender node
		const auto send = all_instrs.select_unique<send_instruction_record>();
		CHECK(send->transfer_id == expected_trid);
		CHECK(send->send_range == range_cast<3>(range(1)));
		CHECK(send->offset_in_buffer == zeros);
		CHECK(send->element_size == sizeof(int));

		const auto pilot = all_pilots.assert_unique();
		CHECK(pilot->to == 0);
		CHECK(pilot->message.transfer_id == expected_trid);
		CHECK(pilot->message.box == box<3>(zeros, ones));
	}
}

TEST_CASE("local reductions only include values from participating devices", "[instruction_graph_generator][instruction-graph][reduction]") {
	const size_t num_nodes = 1;
	const node_id my_nid = 0;
	const auto num_devices = 4; // we need multiple, but not all devices to produce partial reduction results

	test_utils::idag_test_context ictx(num_nodes, my_nid, num_devices);
	auto buf = ictx.create_buffer(range<1>(1));
	ictx.device_compute(range<1>(num_devices / 2)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	CHECK(all_writers.count() == num_devices / 2);

	// look up the reduce-instruction first because it defines which memory / allocation we need to gather-copy to
	const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();

	// there is one gather-copy for each writer kernel
	for(const auto& writer : all_writers.iterate()) {
		CAPTURE(writer);

		CHECK(writer->access_map.empty()); // we have reductions, not (regular) accesses
		REQUIRE(writer->reduction_map.size() == 1);
		const auto& red_acc = writer->reduction_map.front();

		const auto gather_copy = writer.successors().assert_unique<copy_instruction_record>();
		CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);

		CHECK(gather_copy->source_allocation_id == red_acc.allocation_id);
		CHECK(gather_copy->source_layout == region_layout(strided_layout(box<3>(zeros, ones))));

		// gather-order must be deterministic because the reduction operation is not necessarily associative
		CHECK(gather_copy->dest_allocation_id == local_reduce->source_allocation_id);
		CHECK(gather_copy->dest_layout == region_layout(linearized_layout(writer->device_id * sizeof(int))));

		CHECK(local_reduce.predecessors().contains(gather_copy));
	}
}

TEST_CASE("global reductions without a local contribution do not read stale local values", "[instruction_graph_generator][instruction-graph][reduction]") {
	const size_t num_nodes = 3;
	const node_id local_nid = GENERATE(values<node_id>({0, 1, 2}));
	const auto num_devices = 1;

	test_utils::idag_test_context ictx(num_nodes, local_nid, num_devices);
	auto buf = ictx.create_buffer(range<1>(1));
	ictx.device_compute(range<1>(2)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	const auto reader_tid = ictx.device_compute(range<1>(num_nodes)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_pilots = ictx.query_outbound_pilots();

	const auto global_reduce = all_instrs.select_unique<reduce_instruction_record>();
	CHECK(global_reduce->scope == reduce_instruction_record::reduction_scope::global);
	CHECK(global_reduce->buffer_id == buf.get_id());
	CHECK(global_reduce->box == box<3>(zeros, ones));
	CHECK(global_reduce->num_source_values == num_nodes);

	const auto gather_recv = all_instrs.select_unique<gather_receive_instruction_record>();
	CHECK(global_reduce.predecessors().contains(gather_recv));
	CHECK(gather_recv->transfer_id.rid == global_reduce->reduction_id);
	CHECK(gather_recv->transfer_id.bid == buf.get_id());

	// the gather-receive buffer must be filled with the reduction identity since some nodes might not contribute partial results (and notify us of that fact at
	// runtime via zero-range pilot messages).
	const auto fill_identity = all_instrs.select_unique<fill_identity_instruction_record>();
	CHECK(gather_recv.predecessors().contains(fill_identity));
	CHECK(fill_identity->allocation_id == gather_recv->allocation_id);
	CHECK(fill_identity->num_values == gather_recv->num_nodes);
	CHECK(fill_identity->reduction_id == global_reduce->reduction_id);

	if(local_nid < 2) {
		// there is a local contribution, which will be copied to the global gather buffer concurrent with the receive
		const auto gather_copy = global_reduce.predecessors().select_unique<copy_instruction_record>();
		CHECK(gather_copy->origin == copy_instruction_record::copy_origin::gather);
		CHECK(gather_copy->dest_allocation_id == gather_recv->allocation_id);
		CHECK(gather_copy->dest_layout == region_layout(linearized_layout(local_nid * sizeof(int))));
		CHECK(gather_copy->source_layout == region_layout(strided_layout(box<3>(zeros, ones))));
		CHECK(gather_copy->copy_region == region(box<3>(zeros, ones)));
		CHECK(gather_copy.is_concurrent_with(gather_recv));

		// fill_identity writes the entire buffer, so we need to overwrite one slot with our contribution
		CHECK(gather_copy.predecessors().contains(fill_identity));

		// since every node participates in the global reduction, we need to push our partial results to every peer
		const auto partial_result_sends = all_instrs.select_all<send_instruction_record>();
		CHECK(partial_result_sends.count() == num_nodes - 1);
		for(const auto& send : partial_result_sends.iterate()) {
			CAPTURE(send);
			CHECK(send->transfer_id == gather_recv->transfer_id);
			CHECK(send->send_range == ones);
			CHECK(send.is_concurrent_with(gather_copy));

			const auto pilot = all_pilots.select_unique(send->dest_node_id);
			CHECK(pilot->message.transfer_id == send->transfer_id);
			CHECK(pilot->message.box == box<3>(zeros, ones));
		}

		// global_reduce will overwrite the host buffer, so it must anti-depend on the partial-result send instructions
		CHECK(global_reduce.predecessors().contains(partial_result_sends));
	} else {
		// there is no local contribution, but we still participate in the global reduction
		CHECK(all_instrs.count<send_instruction_record>() == 0);

		// we signal all peers that we are not going to perform a `send` by transmitting zero-ranged pilots
		for(node_id peer = 0; peer < 2; ++peer) {
			const auto pilot = all_pilots.select_unique(peer);
			CHECK(pilot->message.transfer_id == transfer_id(reader_tid, buf.get_id(), global_reduce->reduction_id));
			CHECK(pilot->message.box == box<3>());
		}
	}

	const auto reader = all_instrs.select_unique<device_kernel_instruction_record>("reader");
	CHECK(reader.transitive_predecessors_across<copy_instruction_record>().contains(global_reduce));
}

TEST_CASE("horizons and epochs notify the executor of completed reductions", "[instruction_graph_generator][instruction-graph][reduction]") {
	const size_t num_nodes = GENERATE(values<size_t>({1, 2}));
	const auto num_devices = GENERATE(values<size_t>({1, 2}));
	const auto trigger = GENERATE(values<std::string>({"horizon", "epoch"}));
	CAPTURE(num_nodes, num_devices, trigger);

	test_utils::idag_test_context ictx(num_nodes, 0 /* local nid */, num_devices);
	ictx.set_horizon_step(trigger == "horizon" ? 2 : 999);
	auto buf = ictx.create_buffer(range<1>(1));
	ictx.device_compute(range<1>(num_nodes)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	SECTION("when the reduction result is subsequently read") { ictx.device_compute(range<1>(num_nodes)).name("reader").read(buf, acc::all()).submit(); }
	SECTION("when the reduction result is discarded") { ictx.device_compute(range<1>(num_nodes)).name("reader").read(buf, acc::all()).submit(); }
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	const auto rid = all_writers[0]->reduction_map.at(0).reduction_id;

	if(trigger == "horizon") {
		const auto horizon = all_instrs.select_unique<horizon_instruction_record>();
		CHECK(horizon->garbage.reductions == std::vector{rid});
	} else {
		const auto epoch = all_instrs.select_unique<epoch_instruction_record>(
		    [](const epoch_instruction_record& einstr) { return einstr.epoch_action == epoch_action::shutdown; });
		CHECK(epoch->garbage.reductions == std::vector{rid});
	}
}

TEST_CASE(
    "global reductions do not include stale local values even if the local node did not receive an execution_command for the kernel initiating the reduction",
    "[instruction_graph_generator][instruction-graph][reduction]") //
{
	// N0 and N1 contribute to the reduction. N2 consumes the reduction result and thus receives a `reduction_command`, but must not include its own (stale)
	// content of the buffer even though it locally appears up-to-date at this point.
	test_utils::idag_test_context ictx(3 /* num nodes */, 2 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<1>(1, true /* host initialized */);
	ictx.device_compute(range(2)).name("writer").reduce(buf, false /* include_current_buffer_value */).submit();
	ictx.device_compute(range(3)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto reduce = all_instrs.select_unique<reduce_instruction_record>();

	// the reduce-instruction operates on a temporary gather allocation, so any local contribution would have to be copied there first.
	CHECK(reduce.predecessors().count<gather_receive_instruction_record>() == 1);
	CHECK(reduce.predecessors().count<copy_instruction_record>() == 0);
}
