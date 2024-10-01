#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "instruction_graph_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;

namespace acc = celerity::access;


TEST_CASE("empty-range buffer accesses do not trigger allocations or cause dependencies", "[instruction_graph_generator][instruction-graph][memory]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer(range(256));
	ictx.device_compute(range(1)).discard_write(buf, acc::fixed<1>({0, 0})).submit();
	ictx.device_compute(range(1)).discard_write(buf, acc::fixed<1>({128, 0})).submit();
	ictx.master_node_host_task().read_write(buf, acc::fixed<1>({0, 0})).submit();
	ictx.master_node_host_task().read_write(buf, acc::fixed<1>({128, 0})).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.select_all<alloc_instruction_record>().count() == 0);
	CHECK(all_instrs.select_all<copy_instruction_record>().count() == 0);

	const auto all_device_kernels = all_instrs.select_all<device_kernel_instruction_record>();
	CHECK(all_device_kernels.count() == 2);
	for(const auto& kernel : all_device_kernels.iterate()) {
		REQUIRE(kernel->access_map.size() == 1); // we still need to encode the null allocation for the hydration mechanism
		CHECK(kernel->access_map.front().allocation_id == null_allocation_id);
	}

	const auto all_host_tasks = all_instrs.select_all<host_task_instruction_record>();
	CHECK(all_host_tasks.count() == 2);
	for(const auto& host_task : all_host_tasks.iterate()) {
		REQUIRE(host_task->access_map.size() == 1); // we still need to encode the null allocation for the hydration mechanism
		CHECK(host_task->access_map.front().allocation_id == null_allocation_id);
	}

	CHECK(union_of(all_device_kernels, all_host_tasks).all_concurrent());
}

TEMPLATE_TEST_CASE_SIG("multiple overlapping accessors trigger allocation of their bounding box", "[instruction_graph_generator][instruction-graph][memory]",
    ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto access_range = test_utils::truncate_range<Dims>({128, 128, 128});
	const auto access_offset_1 = test_utils::truncate_id<Dims>({32, 32, 32});
	const auto access_offset_2 = test_utils::truncate_id<Dims>({96, 96, 96});

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer(full_range);
	ictx.device_compute(access_range)
	    .discard_write(buf, [=](const chunk<Dims>& ck) { return subrange(ck.offset + access_offset_1, ck.range); })
	    .discard_write(buf, [=](const chunk<Dims>& ck) { return subrange(ck.offset + access_offset_2, ck.range); })
	    .submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto kernel = all_instrs.select_unique<device_kernel_instruction_record>();

	const auto alloc = all_instrs.select_unique<alloc_instruction_record>();
	CHECK(alloc->allocation_id.get_memory_id() == ictx.get_native_memory(kernel->device_id));
	CHECK(alloc->buffer_allocation->buffer_id == buf.get_id());

	// the IDAG must allocate the bounding box for both accessors to map to overlapping, contiguous memory
	const auto expected_box = bounding_box(box(subrange(access_offset_1, access_range)), box(subrange(access_offset_2, access_range)));
	CHECK(alloc->buffer_allocation->box == box_cast<3>(expected_box));
	CHECK(alloc->size_bytes == expected_box.get_area() * sizeof(int));
	CHECK(alloc.successors().contains(kernel));

	// alloc and free instructions are always symmetric
	const auto free = all_instrs.select_unique<free_instruction_record>();
	CHECK(free->allocation_id == alloc->allocation_id);
	CHECK(free->size == alloc->size_bytes);
	CHECK(free->buffer_allocation == alloc->buffer_allocation);
	CHECK(free.predecessors().contains(kernel));
}

TEMPLATE_TEST_CASE_SIG(
    "allocations and kernels are split between devices", "[instruction_graph_generator][instruction-graph][memory]", ((int Dims), Dims), 1, 2, 3) //
{
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256}); // dim0 split
	auto buf = ictx.create_buffer(full_range);
	ictx.device_compute(full_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	CHECK(all_instrs.select_all<alloc_instruction_record>().predecessors().all_match<epoch_instruction_record>());

	// we have two writer instructions, one per device, each operating on their separate allocations on separate memories.
	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>();
	CHECK(all_writers.count() == 2);
	CHECK(all_writers.all_match("writer"));
	CHECK(all_writers[0]->device_id != all_writers[1]->device_id);
	CHECK(
	    region_union(box(all_writers[0]->execution_range), box(all_writers[1]->execution_range)) == region(box(subrange<3>(zeros, range_cast<3>(full_range)))));

	for(const auto& writer : all_writers.iterate()) {
		CAPTURE(writer);
		REQUIRE(writer->access_map.size() == 1);

		// instruction_graph_generator guarantees the default dim0 split
		CHECK(writer->execution_range.get_range() == range_cast<3>(half_range));
		CHECK((writer->execution_range.get_offset()[0] == 0 || writer->execution_range.get_offset()[0] == half_range[0]));

		// the IDAG allocates appropriate boxes on the memories native to each executing device.
		const auto alloc = writer.predecessors().assert_unique<alloc_instruction_record>();
		CHECK(alloc->allocation_id.get_memory_id() == ictx.get_native_memory(writer->device_id));
		CHECK(writer->access_map.front().allocation_id == alloc->allocation_id);
		CHECK(alloc->buffer_allocation.value().box == writer->access_map.front().accessed_box_in_buffer);

		const auto free = writer.successors().assert_unique<free_instruction_record>();
		CHECK(free->allocation_id == alloc->allocation_id);
		CHECK(free->size == alloc->size_bytes);
	}

	CHECK(all_instrs.select_all<free_instruction_record>().successors().all_match<epoch_instruction_record>());
}

TEST_CASE("data-dependencies are generated between kernels on the same memory", "[instruction_graph_generator][instruction-graph][memory]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf1 = ictx.create_buffer<1>(256);
	auto buf2 = ictx.create_buffer<1>(256);
	ictx.device_compute(range(1)).name("write buf1").discard_write(buf1, acc::all()).submit();
	ictx.device_compute(range(1)).name("overwrite buf1 right").discard_write(buf1, acc::fixed<1>({128, 128})).submit();
	ictx.device_compute(range(1)).name("read buf 1, write buf2").read(buf1, acc::all()).discard_write(buf2, acc::all()).submit();
	ictx.device_compute(range(1)).name("read-write buf1 center").read_write(buf1, acc::fixed<1>({64, 128})).submit();
	ictx.device_compute(range(1)).name("read buf2").read(buf2, acc::all()).submit();
	ictx.device_compute(range(1)).name("read buf1+2").read(buf1, acc::all()).read(buf2, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto predecessor_kernels = [](const auto& q) { return q.predecessors().template select_all<device_kernel_instruction_record>(); };

	const auto write_buf1 = all_instrs.select_unique<device_kernel_instruction_record>("write buf1");
	CHECK(predecessor_kernels(write_buf1).count() == 0);

	const auto overwrite_buf1_right = all_instrs.select_unique<device_kernel_instruction_record>("overwrite buf1 right");
	CHECK(predecessor_kernels(overwrite_buf1_right) == write_buf1 /* output-dependency on buf1 [128] - [256]*/);

	const auto read_buf1_write_buf2 = all_instrs.select_unique<device_kernel_instruction_record>("read buf 1, write buf2");
	CHECK(predecessor_kernels(read_buf1_write_buf2).contains(overwrite_buf1_right /* true-dependency on buf1 [128] - [256]*/));
	// IDAG might also specify a true-dependency on "write buf1" for buf1 [0] - [128], but this is transitive

	const auto read_write_buf1_center = all_instrs.select_unique<device_kernel_instruction_record>("read-write buf1 center");
	CHECK(predecessor_kernels(read_write_buf1_center).contains(read_buf1_write_buf2 /* anti-dependency on buf1 [64] - [192]*/));
	// IDAG might also specify true-dependencies on "write buf1" and "overwrite buf1 right", but these are transitive

	const auto read_buf2 = all_instrs.select_unique<device_kernel_instruction_record>("read buf2");
	CHECK(predecessor_kernels(read_buf2) == read_buf1_write_buf2 /* true-dependency on buf2 [0] - [256] */);
	// This should not depend on any other kernel instructions, because none other are concerned with buf2.

	const auto read_buf1_buf2 = all_instrs.select_unique<device_kernel_instruction_record>("read buf1+2");
	CHECK(predecessor_kernels(read_buf1_buf2).contains(read_write_buf1_center) /* true-dependency on buf1 [64] - [192] */);
	CHECK(!predecessor_kernels(read_buf1_buf2).contains(read_buf2) /* readers are concurrent */);
	// IDAG might also specify true-dependencies on "write buf1", "overwrite buf1 right", "read buf1, write_buf2", but these are transitive
}

TEST_CASE("data dependencies across memories introduce coherence copies", "[instruction_graph_generator][instruction-graph][memory]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 2 /* devices */);
	const range<1> test_range = {256};
	auto buf = ictx.create_buffer(test_range);
	ictx.device_compute(test_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute(test_range).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");
	const auto coherence_copies = all_instrs.select_all<copy_instruction_record>(
	    [](const copy_instruction_record& copy) { return copy.origin == copy_instruction_record::copy_origin::coherence; });

	CHECK(all_readers.count() == 2);
	for(device_id did = 0; did < 2; ++did) {
		const device_id opposite_did = 1 - did;
		CAPTURE(did, opposite_did);

		const auto reader = all_readers.select_unique(did);
		REQUIRE(reader->access_map.size() == 1);
		const auto opposite_writer = all_writers.select_unique(opposite_did);
		REQUIRE(opposite_writer->access_map.size() == 1);

		// There is one coherence copy per reader kernel, which copies the portion written on the opposite device
		const auto coherence_copy = intersection_of(coherence_copies, reader.predecessors()).assert_unique();
		CHECK(coherence_copy->source_allocation_id.get_memory_id() == ictx.get_native_memory(opposite_did));
		CHECK(coherence_copy->dest_allocation_id.get_memory_id() == ictx.get_native_memory(did));
		CHECK(coherence_copy->copy_region == region(opposite_writer->access_map.front().accessed_box_in_buffer));
	}

	// Coherence copies are not sequenced with respect to each other
	CHECK(coherence_copies.all_concurrent());
}

TEMPLATE_TEST_CASE_SIG(
    "coherence copies of the same data are performed only once", "[instruction_graph_generator][instruction-graph][memory]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256});
	const auto first_half = subrange(id<Dims>(zeros), half_range);
	const auto second_half = subrange(test_utils::truncate_id<Dims>({128, 0, 0}), half_range);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(999);
	auto buf = ictx.create_buffer<int>(full_range);

	// write once to avoid resizes of host buffer later on
	ictx.host_task(full_range).name("init").discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute(range(1)).name("write 1st half").discard_write(buf, acc::fixed(first_half)).submit();
	// requires a coherence copy for the first half
	ictx.master_node_host_task().name("read 1st half").read(buf, acc::fixed(first_half)).submit();
	ictx.master_node_host_task().name("read 1st half").read(buf, acc::fixed(first_half)).submit();
	ictx.device_compute(range(1)).name("write 2nd half").discard_write(buf, acc::fixed(second_half)).submit();
	// requires a coherence copy for the second half
	ictx.master_node_host_task().name("read all").read(buf, acc::all()).submit();
	ictx.master_node_host_task().name("read all").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto write_first_half = all_instrs.select_all<device_kernel_instruction_record>("write 1st half");
	const auto read_first_half = all_instrs.select_all<host_task_instruction_record>("read 1st half");
	const auto write_second_half = all_instrs.select_all<device_kernel_instruction_record>("write 2nd half");
	const auto read_all = all_instrs.select_all<host_task_instruction_record>("read all");
	const auto all_copies = all_instrs.select_all<copy_instruction_record>();

	// there is one device -> host copy for each half
	CHECK(all_copies.count() == 2);
	const auto first_half_copy = intersection_of(all_copies, write_first_half.successors()).assert_unique();
	CHECK(first_half_copy->copy_region == region(box_cast<3>(box(first_half))));
	const auto second_half_copy = intersection_of(all_copies, write_second_half.successors()).assert_unique();
	CHECK(second_half_copy->copy_region == region(box_cast<3>(box(second_half))));

	// copies depend on their producers only
	CHECK(first_half_copy.predecessors().contains(write_first_half));
	CHECK(first_half_copy.successors().contains(read_first_half)); // both readers of 1st half
	CHECK(first_half_copy.successors().contains(read_all));        // both readers of full range

	CHECK(second_half_copy.predecessors().contains(write_second_half));
	CHECK(second_half_copy.successors().contains(read_all)); // both readers of full range
}

TEMPLATE_TEST_CASE_SIG("local copies are split on writers to facilitate compute-copy overlap", "[instruction_graph_generator][instruction-graph][memory]",
    ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256});
	const auto first_half = subrange(id<Dims>(zeros), half_range);
	const auto second_half = subrange(test_utils::truncate_id<Dims>({128, 0, 0}), half_range);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(999);

	auto buf = ictx.create_buffer<int>(full_range);
	// first write the entire range on D0 to guarantee a single allocation on M1
	ictx.device_compute(range(1)).name("init").discard_write(buf, acc::all()).submit();
	// both writers combined overwrite the entire buffer
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::fixed(first_half)).submit();
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::fixed(second_half)).submit();
	// read on host / M0 must create one copy per writer to allow compute-copy overlap
	ictx.master_node_host_task().name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	const auto all_copies = all_instrs.select_all<copy_instruction_record>();
	const auto reader = all_instrs.select_unique<host_task_instruction_record>("reader");

	REQUIRE(all_writers.count() == 2);
	REQUIRE(all_copies.count() == 2);

	// the reader depends on one copy per writer
	for(size_t copy_idx = 0; copy_idx < all_copies.count(); ++copy_idx) {
		const auto this_copy = all_copies[copy_idx];
		const auto writer = intersection_of(this_copy.predecessors(), all_writers).assert_unique<device_kernel_instruction_record>();
		REQUIRE(writer->access_map.size() == 1);

		const auto& write = writer->access_map.front();
		CHECK(write.buffer_id == this_copy->buffer_id);
		CHECK(region(write.accessed_box_in_buffer) == this_copy->copy_region);
		this_copy.successors().contains(reader);

		// each copy can be issued once its writer has completed in order to overlap with the other writer
		const auto other_copy = all_copies[1 - copy_idx];
		CHECK(writer.is_concurrent_with(other_copy));
	}
}

// see also: "overlapping requirements generate split-receives with one await per reader-set" from the p2p tests.
TEMPLATE_TEST_CASE_SIG("overlapping requirements between chunks of an oversubscribed kernel generate one copy per reader-set",
    "[instruction_graph_generator][instruction-graph][memory]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto test_range = test_utils::truncate_range<Dims>({256, 256, 256});
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer<int>(test_range);
	ictx.master_node_host_task() //
	    .name("writer")
	    .discard_write(buf, acc::all())
	    .submit();
	ictx.device_compute(test_range)
	    .name("reader")
	    .read(buf, test_utils::access::make_neighborhood<Dims>(1))
	    .hint(experimental::hints::oversubscribe(4))
	    .submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// We do not know the send-split, so we create a receive-split that awaits subregions used by single chunks separately from subregions used by multiple
	// chunks (the neighborhood overlap) in order to un-block any instruction as soon as their requirements are fulfilled.
	const auto all_copies = all_instrs.select_all<copy_instruction_record>();
	CHECK(all_copies.all_concurrent());
	CHECK(all_copies.count() == 7); // 1 for each chunk, 1 for each overlap between them

	// the copy ranges for the split must never intersect, but their union must cover the entire received region
	region<3> copied_region;
	for(const auto& copy : all_copies.iterate()) {
		CHECK(region_intersection(copied_region, copy->copy_region).empty());
		copied_region = region_union(copied_region, copy->copy_region);
	}
	CHECK(copied_region == box<3>::full_range(range_cast<3>(test_range)));

	// Each reader instruction sources its input data from multiple copy instructions, and by extension, the copies operating on the overlap service multiple
	// reader chunks.
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");
	for(const auto& reader : all_readers.iterate()) {
		const auto all_pred_copies = reader.predecessors().assert_all<copy_instruction_record>();
		CHECK(all_pred_copies.count() > 1);

		// Sanity check: Each reader chunk depends on copy-instructions for the subranges it reads
		region<3> pred_copied_region;
		for(const auto& pred_copy : all_pred_copies.iterate()) {
			pred_copied_region = region_union(pred_copied_region, pred_copy->copy_region);
		}

		REQUIRE(reader->access_map.size() == 1);
		const auto& read = reader->access_map.front();
		CHECK(read.buffer_id == buf.get_id());
		CHECK(read.accessed_box_in_buffer == pred_copied_region);
	}
}


// This test may fail in the future if we implement a more sophisticated allocator that decides to merge some allocations.
// When this happens, consider replacing the subranges with a pair for which it is never reasonable to allocate the bounding-box.
TEMPLATE_TEST_CASE_SIG("accessing non-overlapping buffer subranges in subsequent kernels triggers distinct allocations",
    "[instrution_graph_generator][instruction-graph][memory]", ((int Dims), Dims), 1, 2, 3) //
{
	const auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 128, 128});
	const auto first_half = subrange(id<Dims>(zeros), half_range);
	const auto second_half = subrange(id(half_range), half_range);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer(full_range);
	ictx.device_compute(first_half.range, first_half.offset).name("1st").discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute(second_half.range, second_half.offset).name("2nd").discard_write(buf, acc::one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	CHECK(all_instrs.select_all<alloc_instruction_record>().count() == 2);
	CHECK(all_instrs.select_all<copy_instruction_record>().count() == 0); // no coherence copies needed
	CHECK(all_instrs.select_all<device_kernel_instruction_record>().count() == 2);
	CHECK(all_instrs.select_all<free_instruction_record>().count() == 2);

	const auto first = all_instrs.select_unique<device_kernel_instruction_record>("1st");
	const auto second = all_instrs.select_unique<device_kernel_instruction_record>("2nd");
	REQUIRE(first->access_map.size() == 1);
	REQUIRE(second->access_map.size() == 1);

	// the kernels access distinct allocations
	CHECK(first->access_map.front().allocation_id != second->access_map.front().allocation_id);
	CHECK(first->access_map.front().accessed_box_in_buffer == first->access_map.front().allocated_box_in_buffer);

	// the allocations exactly match the accessed subrange
	CHECK(second->access_map.front().accessed_box_in_buffer == second->access_map.front().allocated_box_in_buffer);

	// kernels are fully concurrent
	CHECK(first.predecessors().is_unique<alloc_instruction_record>());
	CHECK(first.successors().is_unique<free_instruction_record>());
	CHECK(second.predecessors().is_unique<alloc_instruction_record>());
	CHECK(second.successors().is_unique<free_instruction_record>());
}

TEST_CASE("resizing a buffer allocation for a discard-write access preserves only the non-overwritten parts", //
    "[instruction_graph_generator][instruction-graph][memory]")                                               //
{
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer(range<1>(256));
	ictx.device_compute(range<1>(1)).name("1st writer").discard_write(buf, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute(range<1>(1)).name("2nd writer").discard_write(buf, acc::fixed<1>({64, 196})).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	// part of the buffer is allocated for the first writer
	const auto first_writer = all_instrs.select_unique<device_kernel_instruction_record>("1st writer");
	REQUIRE(first_writer->access_map.size() == 1);
	const auto first_alloc = first_writer.predecessors().select_unique<alloc_instruction_record>();
	const auto first_write_box = first_writer->access_map.front().accessed_box_in_buffer;
	CHECK(first_alloc->buffer_allocation.value().box == first_write_box);

	// first and second writer ranges overlap, so the bounding box has to be allocated (and the old allocation freed)
	const auto second_writer = all_instrs.select_unique<device_kernel_instruction_record>("2nd writer");
	REQUIRE(second_writer->access_map.size() == 1);
	const auto second_alloc = second_writer.predecessors().assert_unique<alloc_instruction_record>();
	const auto second_write_box = second_writer->access_map.front().accessed_box_in_buffer;
	const auto large_alloc_box = bounding_box(first_write_box, second_write_box);
	CHECK(second_alloc->buffer_allocation.value().box == large_alloc_box);

	// The copy must not attempt to preserve ranges that were not written in the old allocation ([128] - [256]) or that were written but are going to be
	// overwritten (without being read) in the command for which the resize was generated ([64] - [128]).
	const auto preserved_region = region_difference(first_write_box, second_write_box);
	REQUIRE(preserved_region.get_boxes().size() == 1);
	const auto preserved_box = preserved_region.get_boxes().front();

	const auto resize_copy = all_instrs.select_unique<copy_instruction_record>();
	CHECK(resize_copy->copy_region == region(preserved_box));

	const auto resize_copy_preds = resize_copy.predecessors();
	CHECK(resize_copy_preds.count() == 2);
	CHECK(resize_copy_preds.select_unique<device_kernel_instruction_record>() == first_writer);
	CHECK(resize_copy_preds.select_unique<alloc_instruction_record>() == second_alloc);

	// resize-copy and overwriting kernel are concurrent, because they access non-overlapping regions in the same allocation
	CHECK(second_writer.successors().all_match<free_instruction_record>());
	CHECK(resize_copy.successors().all_match<free_instruction_record>());
}

TEMPLATE_TEST_CASE_SIG("data from user-initialized buffers is copied lazily to managed allocations", "[instruction_graph_generator][instruction-graph][memory]",
    ((int Dims), Dims), 1, 2, 3) //
{
	const auto buffer_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto half_range = test_utils::truncate_range<Dims>({128, 256, 256});
	const auto second_half_offset = test_utils::truncate_id<Dims>({128, 0, 0});
	const auto buffer_box = box(subrange(id<Dims>(zeros), buffer_range));

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer<int, Dims>(buffer_range, true /* host_initialized */);

	SECTION("for host access") {
		ictx.master_node_host_task().read(buf, acc::all()).submit();
		ictx.finish();

		const auto all_instrs = ictx.query_instructions();
		const auto reader = all_instrs.select_unique<host_task_instruction_record>();
		const auto coherence_copy = all_instrs.select_unique<copy_instruction_record>();
		CHECK(coherence_copy->copy_region == region(box_cast<3>(buffer_box)));
		CHECK(coherence_copy->source_allocation_id.get_memory_id() == user_memory_id);
		CHECK(coherence_copy->dest_allocation_id.get_memory_id() == host_memory_id);

		// must not attempt to free user allocation
		CHECK(all_instrs.count<free_instruction_record>([](const free_instruction_record& finstr) {
			return finstr.allocation_id.get_memory_id() == user_memory_id;
		}) == 0);
	}

	SECTION("for device access, by staging through (pinned) host memory") {
		ictx.device_compute(range(1)).read(buf, acc::fixed(subrange({}, half_range))).submit();
		ictx.device_compute(range(1)).read(buf, acc::fixed(subrange(second_half_offset, half_range))).submit();
		ictx.finish();

		const auto all_instrs = ictx.query_instructions();
		CHECK(all_instrs.count<copy_instruction_record>() == 4);

		const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>();
		const auto all_coherence_copies = all_readers.predecessors().select_all<copy_instruction_record>();
		CHECK(all_coherence_copies.count() == 2);

		for(const auto& coherence_copy : all_coherence_copies.iterate()) {
			REQUIRE(coherence_copy->copy_region.get_boxes().size() == 1);
			const auto& copy_box = coherence_copy->copy_region.get_boxes().front();
			CHECK(copy_box.get_range() == range_cast<3>(half_range));
			CHECK(coherence_copy->source_allocation_id.get_memory_id() == host_memory_id);
			CHECK(coherence_copy->dest_allocation_id.get_memory_id() == first_device_memory_id);
		}

		const auto all_staging_copies = all_coherence_copies.predecessors().select_all<copy_instruction_record>();
		CHECK(all_staging_copies.count() == 2);

		for(const auto& staging_copy : all_staging_copies.iterate()) {
			REQUIRE(staging_copy->copy_region.get_boxes().size() == 1);
			const auto& copy_box = staging_copy->copy_region.get_boxes().front();
			CHECK(copy_box.get_range() == range_cast<3>(half_range));
			CHECK(staging_copy->source_allocation_id.get_memory_id() == user_memory_id);
			CHECK(staging_copy->dest_allocation_id.get_memory_id() == host_memory_id);
		}

		// must not attempt to free user allocation
		CHECK(all_instrs.count<free_instruction_record>([](const free_instruction_record& finstr) {
			return finstr.allocation_id.get_memory_id() == user_memory_id;
		}) == 0);
	}
}

TEST_CASE("copies are staged through host memory for devices that are not peer-copy capable", "[instruction_graph_generator][instruction-graph][memory]") {
	const auto buffer_range = range(256);
	const size_t num_devices = 2;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices, false /* supports d2d copies */);
	auto buf = ictx.create_buffer<int>(buffer_range);
	ictx.device_compute(buffer_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	ictx.device_compute(buffer_range).name("reader").read(buf, test_utils::access::reverse_one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto stage_alloc = all_instrs.select_unique<alloc_instruction_record>(
	    [](const alloc_instruction_record& alloc) { return alloc.allocation_id.get_memory_id() == host_memory_id; });
	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	const auto all_copies_from_source = all_writers.successors().select_all<copy_instruction_record>();
	const auto all_copies_to_dest = all_copies_from_source.successors().select_all<copy_instruction_record>();
	const auto all_readers = all_copies_to_dest.successors().select_all<device_kernel_instruction_record>("reader");
	const auto stage_free = all_instrs.select_unique<free_instruction_record>(
	    [](const free_instruction_record& free) { return free.allocation_id.get_memory_id() == host_memory_id; });

	CHECK(all_writers.count() == num_devices);
	CHECK(all_copies_from_source.count() == num_devices);
	CHECK(all_copies_to_dest.count() == num_devices);
	CHECK(all_readers.count() == num_devices);

	for(const auto& writer : all_writers.iterate()) {
		const auto copy_from_source = intersection_of(all_copies_from_source, writer.successors()).assert_unique();
		const auto copy_to_dest = intersection_of(all_copies_to_dest, copy_from_source.successors()).assert_unique();
		const auto reader = intersection_of(all_readers, copy_to_dest.successors()).assert_unique();

		CHECK(copy_from_source->origin == copy_instruction_record::copy_origin::staging);
		CHECK(copy_from_source->source_allocation_id.get_memory_id() == ictx.get_native_memory(writer->device_id));
		CHECK(copy_from_source->dest_allocation_id.get_memory_id() == host_memory_id);
		CHECK(std::holds_alternative<linearized_layout>(copy_from_source->dest_layout));
		CHECK(copy_to_dest->origin == copy_instruction_record::copy_origin::coherence);
		CHECK(copy_to_dest->source_allocation_id.get_memory_id() == host_memory_id);
		CHECK(copy_to_dest->dest_allocation_id.get_memory_id() == ictx.get_native_memory(reader->device_id));
		CHECK(std::holds_alternative<linearized_layout>(copy_to_dest->source_layout));
	}
}

TEST_CASE("host-staged copies preserve concurrency between multiple readers and writers ", "[instruction_graph_generator][instruction-graph][memory]") {
	const auto buffer_range = range(256);
	const size_t num_devices = 2;
	const size_t oversub_factor = 2;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices, false /* supports d2d copies */);
	auto buf = ictx.create_buffer<int>(buffer_range);
	ictx.device_compute(buffer_range) //
	    .name("writer")
	    .discard_write(buf, acc::one_to_one())
	    .hint(experimental::hints::oversubscribe(oversub_factor))
	    .submit();
	ictx.device_compute(buffer_range)
	    .name("reader")
	    .read(buf, test_utils::access::reverse_one_to_one())
	    .hint(experimental::hints::oversubscribe(oversub_factor))
	    .submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	const auto all_copies_from_source = all_writers.successors().select_all<copy_instruction_record>();
	const auto all_copies_to_dest = all_copies_from_source.successors().select_all<copy_instruction_record>();
	const auto all_readers = all_copies_to_dest.successors().select_all<device_kernel_instruction_record>("reader");

	CHECK(all_writers.count() == num_devices * oversub_factor);
	CHECK(all_copies_from_source.count() == num_devices * oversub_factor);
	CHECK(all_copies_from_source.all_concurrent());
	CHECK(all_copies_to_dest.count() == num_devices * oversub_factor);
	CHECK(all_copies_to_dest.all_concurrent());
	CHECK(all_readers.count() == num_devices * oversub_factor);

	std::vector<size_t> byte_offsets_in_staging_buffer;
	for(const auto& copy_from_source : all_copies_from_source.iterate()) {
		const auto copy_to_dest = intersection_of(all_copies_to_dest, copy_from_source.successors()).assert_unique();

		CHECK(copy_from_source->origin == copy_instruction_record::copy_origin::staging);
		CHECK(copy_from_source->dest_allocation_id.get_memory_id() == host_memory_id);
		CHECK(std::holds_alternative<linearized_layout>(copy_from_source->dest_layout));
		CHECK(copy_to_dest->origin == copy_instruction_record::copy_origin::coherence);
		CHECK(copy_to_dest->source_allocation_id == copy_from_source->dest_allocation_id);
		CHECK(std::holds_alternative<linearized_layout>(copy_to_dest->source_layout));

		byte_offsets_in_staging_buffer.push_back(std::get<linearized_layout>(copy_from_source->dest_layout).offset_bytes);
	}

	std::sort(byte_offsets_in_staging_buffer.begin(), byte_offsets_in_staging_buffer.end());
	const size_t step = (buffer_range.size() / (num_devices * oversub_factor)) * sizeof(int);
	CHECK(byte_offsets_in_staging_buffer == std::vector<size_t>({0 * step, 1 * step, 2 * step, 3 * step}));
}

TEST_CASE("staging copies to host are deduplicated in case of a broadcast pattern", "[instruction_graph_generator][instruction-graph][memory]") {
	const auto buffer_range = range(256);
	const size_t num_devices = 4;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices, false /* supports d2d copies */);
	auto buf = ictx.create_buffer<int>(buffer_range);
	ictx.device_compute(range(1)).name("writer").discard_write(buf, acc::all()).submit();
	ictx.device_compute(buffer_range).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto stage_alloc = all_instrs.select_unique<alloc_instruction_record>(
	    [](const alloc_instruction_record& alloc) { return alloc.allocation_id.get_memory_id() == host_memory_id; });
	const auto writer = all_instrs.select_unique<device_kernel_instruction_record>("writer");
	const auto copy_from_source = writer.successors().select_unique<copy_instruction_record>();
	const auto all_copies_to_dest = copy_from_source.successors().select_all<copy_instruction_record>();
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");
	const auto stage_free = all_instrs.select_unique<free_instruction_record>(
	    [](const free_instruction_record& free) { return free.allocation_id.get_memory_id() == host_memory_id; });

	CHECK(copy_from_source->dest_allocation_id == stage_alloc->allocation_id);
	CHECK(copy_from_source->origin == copy_instruction_record::copy_origin::staging);
	CHECK(copy_from_source->dest_allocation_id.get_memory_id() == host_memory_id);
	CHECK(copy_from_source->dest_layout == region_layout(linearized_layout(0 /* offset */)));

	CHECK(all_copies_to_dest.predecessors().select_all<copy_instruction_record>() == copy_from_source);
	CHECK(all_copies_to_dest.count() == num_devices - 1);
	CHECK(all_copies_to_dest.all_concurrent());

	CHECK(all_readers.count() == num_devices);
	CHECK(stage_free->allocation_id == stage_alloc->allocation_id);
	CHECK(stage_free.predecessors() == all_copies_to_dest);

	std::vector<size_t> byte_offsets_in_staging_buffer;
	for(const auto& copy_to_dest : all_copies_to_dest.iterate()) {
		CHECK(copy_to_dest->origin == copy_instruction_record::copy_origin::coherence);
		CHECK(copy_to_dest->source_allocation_id == copy_from_source->dest_allocation_id);
		CHECK(copy_to_dest->source_layout == region_layout(linearized_layout(0 /* offset */)));
	}
}

TEST_CASE("host staging only writes to device allocations that are not yet up to date", "[instruction_graph_generator][instruction-graph][memory]") {
	const auto buffer_range = range(256);
	const size_t num_devices = 4;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices, false /* supports d2d copies */);
	auto buf = ictx.create_buffer<int>(buffer_range);
	ictx.device_compute(range(1)).name("write").discard_write(buf, acc::all()).submit();
	ictx.device_compute(range(2)).name("read 1").read(buf, acc::all()).submit();
	ictx.device_compute(range(num_devices)).name("read 2").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_reads_1 = all_instrs.select_all<device_kernel_instruction_record>("read 1");
	const auto all_unstage_copies_1 = all_reads_1.predecessors().select_all<copy_instruction_record>();
	const auto all_reads_2 = all_instrs.select_all<device_kernel_instruction_record>("read 2");
	const auto all_reads_on_non_writer =
	    union_of(all_reads_1, all_reads_2) //
	        .select_all<device_kernel_instruction_record>([](const device_kernel_instruction_record& k) { return k.device_id != device_id(0); });
	const auto all_unstage_copies = all_instrs.select_all<copy_instruction_record>(
	    [](const copy_instruction_record& copy) { return copy.origin == copy_instruction_record::copy_origin::coherence; });
	const auto all_unstage_copies_2 = difference_of(all_unstage_copies, all_unstage_copies_1);

	CHECK(all_unstage_copies_1.count() == 1);
	CHECK(all_unstage_copies_1->dest_allocation_id.get_memory_id() == first_device_memory_id + 1); // to D2

	CHECK(all_unstage_copies_2.count() == 2);
	std::array<memory_id, 2> dest_mids = {                           //
	    all_unstage_copies_2[0]->dest_allocation_id.get_memory_id(), //
	    all_unstage_copies_2[1]->dest_allocation_id.get_memory_id()};
	std::sort(dest_mids.begin(), dest_mids.end());
	CHECK(dest_mids == std::array<memory_id, 2>{first_device_memory_id + 2, first_device_memory_id + 3});

	CHECK(all_unstage_copies.count() == 3);
	CHECK(intersection_of(all_unstage_copies_1.successors(), all_reads_1).count() == 1);
	CHECK(intersection_of(all_unstage_copies_1.successors(), all_reads_2).count() == 1); // "read 2" on D1 depends on the unstage-copy inserted for "read 1"
	CHECK(intersection_of(all_unstage_copies_2.successors(), all_reads_2).count() == 2);
}

TEST_CASE("staging allocations are recycled after two horizons", "[instruction_graph_generator][instruction-graph][memory]") {
	const auto buffer_range = range(256);
	const size_t num_devices = 2;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices, false /* supports d2d copies */);
	auto buf1 = ictx.create_buffer<int>(buffer_range);
	auto buf2 = ictx.create_buffer<int>(buffer_range);

	ictx.set_horizon_step(1);
	ictx.device_compute(buffer_range).name("init").discard_write(buf1, acc::one_to_one()).submit();
	for(int i = 0; i < 4; ++i) {
		ictx.device_compute(buffer_range)
		    .name(fmt::format("r/w {}", i))
		    .read(buf1, test_utils::access::reverse_one_to_one())
		    .discard_write(buf2, acc::one_to_one())
		    .submit();
		std::swap(buf1, buf2);
	}
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto stage_allocs = all_instrs.select_all<alloc_instruction_record>(
	    [](const alloc_instruction_record& alloc) { return alloc.allocation_id.get_memory_id() == host_memory_id; });
	const auto init_kernels = all_instrs.select_all<device_kernel_instruction_record>("init");
	const auto copies_from_source_0 = init_kernels.successors().select_all<copy_instruction_record>();
	const auto rw_kernels_0 = all_instrs.select_all<device_kernel_instruction_record>("r/w 0");
	const auto copies_from_source_1 = rw_kernels_0.successors().select_all<copy_instruction_record>();
	const auto rw_kernels_1 = all_instrs.select_all<device_kernel_instruction_record>("r/w 1");
	const auto copies_from_source_2 = rw_kernels_1.successors().select_all<copy_instruction_record>();
	const auto rw_kernels_2 = all_instrs.select_all<device_kernel_instruction_record>("r/w 2");
	const auto copies_from_source_3 = rw_kernels_2.successors().select_all<copy_instruction_record>();
	const auto stage_frees = all_instrs.select_all<free_instruction_record>(
	    [](const free_instruction_record& alloc) { return alloc.allocation_id.get_memory_id() == host_memory_id; });

	// each kernel task triggers a horizon, and after two horizons we re-use the first staging allocation. Within each (task) command, at most one staging
	// allocation is made and shared between all copies that will traverse host memory.
	CHECK(stage_allocs.count() == 2 /* horizons */);
	CHECK(init_kernels.count() == num_devices);
	CHECK(copies_from_source_0.count() == num_devices);
	CHECK(copies_from_source_0[0]->dest_allocation_id == stage_allocs[0]->allocation_id);
	CHECK(copies_from_source_0[1]->dest_allocation_id == stage_allocs[0]->allocation_id);
	CHECK(rw_kernels_0.count() == num_devices);
	CHECK(copies_from_source_1.count() == num_devices);
	CHECK(copies_from_source_1[0]->dest_allocation_id == stage_allocs[1]->allocation_id);
	CHECK(copies_from_source_1[1]->dest_allocation_id == stage_allocs[1]->allocation_id);
	CHECK(rw_kernels_1.count() == num_devices);
	CHECK(copies_from_source_2.count() == num_devices);
	CHECK(copies_from_source_2[0]->dest_allocation_id == stage_allocs[0]->allocation_id);
	CHECK(copies_from_source_2[1]->dest_allocation_id == stage_allocs[0]->allocation_id);
	CHECK(rw_kernels_2.count() == num_devices);
	CHECK(copies_from_source_3.count() == num_devices);
	CHECK(copies_from_source_3[0]->dest_allocation_id == stage_allocs[1]->allocation_id);
	CHECK(copies_from_source_3[1]->dest_allocation_id == stage_allocs[1]->allocation_id);
	CHECK(stage_frees.count() == 2 /* horizons */);
}

TEST_CASE("host copy staging does not introduce dependencies between concurrent kernels", "[instruction_graph_generator][instruction-graph][memory]") {
	const auto buffer_range = range(256);
	const size_t num_devices = 2;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices, false /* supports d2d copies */);

	auto buf1 = ictx.create_buffer<int>(buffer_range);
	ictx.device_compute(buffer_range).name("write 1").discard_write(buf1, acc::one_to_one()).submit();
	ictx.device_compute(buffer_range).name("read 1").read(buf1, test_utils::access::reverse_one_to_one()).submit();

	auto buf2 = ictx.create_buffer<int>(buffer_range);
	ictx.device_compute(buffer_range).name("write 2").discard_write(buf2, acc::one_to_one()).submit();
	ictx.device_compute(buffer_range).name("read 2").read(buf2, test_utils::access::reverse_one_to_one()).submit();

	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto write_1_kernels = all_instrs.select_all<device_kernel_instruction_record>("write 1");
	const auto read_1_kernels = all_instrs.select_all<device_kernel_instruction_record>("read 1");
	const auto write_2_kernels = all_instrs.select_all<device_kernel_instruction_record>("write 2");
	const auto read_2_kernels = all_instrs.select_all<device_kernel_instruction_record>("read 2");

	CHECK(write_1_kernels.count() == num_devices);
	CHECK(read_1_kernels.count() == num_devices);
	CHECK(write_2_kernels.count() == num_devices);
	CHECK(read_2_kernels.count() == num_devices);

	CHECK(read_1_kernels.is_concurrent_with(write_2_kernels));
	CHECK(read_1_kernels.is_concurrent_with(read_2_kernels));
	CHECK(read_2_kernels.is_concurrent_with(write_1_kernels));
	CHECK(read_2_kernels.is_concurrent_with(read_1_kernels));
}

TEST_CASE("narrow strided data columns are device-linearized before and after host-staging", "[instruction_graph_generator][instruction-graph][memory]") {
	const auto buffer_range = range(65536, 65536);
	const size_t num_devices = 4;

	const bool is_column_access = GENERATE(values<int>({false, true})); // rows are already contiguous
	const size_t ext_width = GENERATE(values<size_t>({1, 4096}));       // Celerity shouldn't attempt to linearize 64k x 4k = 1 GiB "columns"
	CAPTURE(is_column_access);

	const size_t ext_0 = is_column_access ? 0 : ext_width;
	const size_t ext_1 = is_column_access ? ext_width : 0;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices, false /* supports d2d copies */);

	auto buf = ictx.create_buffer<int>(buffer_range);
	ictx.device_compute(buffer_range).name("writer").discard_write(buf, acc::one_to_one()).hint(experimental::hints::split_2d()).submit();
	ictx.device_compute(buffer_range).name("reader").read(buf, acc::neighborhood(ext_0, ext_1)).hint(experimental::hints::split_2d()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	const auto all_resizes_after_write = all_writers.successors().select_all<copy_instruction_record>().assert_all(
	    [](const copy_instruction_record& copy) { return copy.origin == copy_instruction_record::copy_origin::resize; });
	const auto all_copies_after_write = all_resizes_after_write.successors().select_all<copy_instruction_record>();
	const auto all_readers = all_instrs.select_all<device_kernel_instruction_record>("reader");

	for(auto& resize : all_resizes_after_write.iterate()) {
		CHECK(resize->dest_allocation_id.get_memory_id() >= first_device_memory_id);
		CHECK(resize->dest_allocation_id.get_memory_id() == resize->source_allocation_id.get_memory_id());
		if(is_column_access && ext_width == 1) {
			for(const auto& linearize_in_source : intersection_of(all_copies_after_write, resize.successors()).iterate()) {
				CHECK(linearize_in_source->source_allocation_id == resize->dest_allocation_id);
				CHECK(std::holds_alternative<strided_layout>(linearize_in_source->source_layout));
				CHECK(linearize_in_source->dest_allocation_id.get_memory_id() == linearize_in_source->source_allocation_id.get_memory_id());
				CHECK(std::holds_alternative<linearized_layout>(linearize_in_source->dest_layout));

				const auto copy_to_host = linearize_in_source.successors().select_unique<copy_instruction_record>();
				CHECK(copy_to_host->source_allocation_id == linearize_in_source->dest_allocation_id);
				CHECK(copy_to_host->source_layout == linearize_in_source->dest_layout);
				CHECK(copy_to_host->dest_allocation_id.get_memory_id() == host_memory_id);
				CHECK(std::holds_alternative<linearized_layout>(copy_to_host->dest_layout));
				CHECK(copy_to_host->copy_region == linearize_in_source->copy_region);

				const auto copy_from_host = copy_to_host.successors().select_unique<copy_instruction_record>();
				CHECK(copy_from_host->source_allocation_id == copy_to_host->dest_allocation_id);
				CHECK(copy_from_host->source_layout == copy_to_host->dest_layout);
				CHECK(copy_from_host->dest_allocation_id.get_memory_id() >= first_device_memory_id);
				CHECK(std::holds_alternative<linearized_layout>(copy_from_host->dest_layout));
				CHECK(copy_from_host->copy_region == copy_to_host->copy_region);

				const auto delinearize_in_dest = copy_from_host.successors().select_unique<copy_instruction_record>();
				CHECK(delinearize_in_dest->source_allocation_id == copy_from_host->dest_allocation_id);
				CHECK(delinearize_in_dest->source_layout == copy_from_host->dest_layout);
				CHECK(delinearize_in_dest->dest_allocation_id.get_memory_id() >= first_device_memory_id);
				CHECK(delinearize_in_dest->dest_allocation_id.get_memory_id() != linearize_in_source->source_allocation_id.get_memory_id());
				CHECK(std::holds_alternative<strided_layout>(delinearize_in_dest->dest_layout));
				CHECK(delinearize_in_dest->copy_region == copy_from_host->copy_region);
			}
		} else {
			for(const auto& copy_to_host : intersection_of(all_copies_after_write, resize.successors()).iterate()) {
				CHECK(copy_to_host->source_allocation_id == resize->dest_allocation_id);
				CHECK(std::holds_alternative<strided_layout>(copy_to_host->source_layout));
				CHECK(copy_to_host->dest_allocation_id.get_memory_id() == host_memory_id);
				CHECK(std::holds_alternative<linearized_layout>(copy_to_host->dest_layout));

				const auto copy_from_host = copy_to_host.successors().select_unique<copy_instruction_record>();
				CHECK(copy_from_host->source_allocation_id == copy_to_host->dest_allocation_id);
				CHECK(copy_from_host->source_layout == copy_to_host->dest_layout);
				CHECK(copy_from_host->dest_allocation_id.get_memory_id() >= first_device_memory_id);
				CHECK(copy_from_host->dest_allocation_id.get_memory_id() != copy_to_host->source_allocation_id.get_memory_id());
				CHECK(std::holds_alternative<strided_layout>(copy_from_host->dest_layout));
			}
		}
	}
}

TEST_CASE("oddly-shaped coherence copies generate a single region-copy instruction", "[instruction_graph_generator][instruction-graph][memory]") {
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 2 /* num devices */);
	auto buf = ictx.create_buffer(range<2>(256, 256));
	// init full buffer on D0 to avoid future resizes
	ictx.device_compute(range(1)).discard_write(buf, acc::all()).submit();
	// overwrite entire buffer on D1
	ictx.device_compute(range(2))
	    .discard_write(buf, [](const chunk<1> ck) { return subrange<2>(zeros, ck.offset[0] + ck.range[0] > 1 ? range(256, 256) : range(0, 0)); })
	    .submit();
	// overwrite box in the center on D0
	ictx.device_compute(range(1)).discard_write(buf, acc::fixed(subrange(id(64, 64), range(128, 128)))).submit();
	// read full buffer on D0 (requires a coherence copy on the border box)
	ictx.device_compute(range(1)).read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.count<alloc_instruction_record>() == 2);

	// we expect one coherence copy
	const auto copy = all_instrs.select_unique<copy_instruction_record>();
	CHECK(copy->copy_region == region_cast<3>(region_difference(box(id(0, 0), id(256, 256)), box(id(64, 64), id(192, 192)))));
}

TEST_CASE("oddly-shaped resize copies generate a single region-copy instruction", "[instruction_graph_generator][instruction-graph][memory]") {
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer(range<2>(256, 256));
	// single-device: allocate and write the upper half of the buffer
	ictx.device_compute(range(1)).discard_write(buf, acc::fixed(subrange(id(0, 0), range(128, 256)))).submit();
	// now write a box that overlaps with the previous write but extends into the previously unallocated part
	ictx.device_compute(range(1)).discard_write(buf, acc::fixed(subrange(id(64, 64), range(128, 128)))).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.count<alloc_instruction_record>() == 2);

	// we expect one resize copy
	const auto copy = all_instrs.select_unique<copy_instruction_record>();
	CHECK(copy->copy_region == region_cast<3>(region_difference(box(id(0, 0), id(128, 256)), box(id(64, 64), id(192, 192)))));
}

TEST_CASE("fully overwriting an allocated region which requires a resize does not generate copy instructions",
    "[instruction_graph_generator][instruction-graph][memory]") //
{
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<float, 1>(32);
	ictx.device_compute(range(1)).name("write 1st half").discard_write(buf, acc::fixed<1>({0, 16})).submit();
	ictx.device_compute(range(1)).name("write 2nd half").discard_write(buf, acc::fixed<1>({0, 32})).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.count<copy_instruction_record>() == 0);

	CHECK(all_instrs.select_unique("write 1st half").is_concurrent_with(all_instrs.select_unique("write 2nd half")));
}

TEST_CASE("resize-copy instructions are only generated from allocations that are not fully overwritten in the same task",
    "[instruction_graph_generator][instruction-graph][memory]") //
{
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<float, 1>(32);
	// trigger two separate allocations for the same buffer and memory
	ictx.device_compute(range(1)).name("alloc 1st").discard_write(buf, acc::fixed<1>({0, 16})).submit();
	ictx.device_compute(range(1)).name("alloc 2nd").discard_write(buf, acc::fixed<1>({16, 16})).submit();
	// resize to the full buffer range, but only read from the first allocation and discard the second
	ictx.device_compute(range(1)).name("resize").read(buf, acc::fixed<1>({0, 8})).discard_write(buf, acc::fixed<1>({8, 24})).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto alloc_1st = all_instrs.select_unique<alloc_instruction_record>(
	    [](const alloc_instruction_record& alloc) { return alloc.buffer_allocation->box == box_cast<3>(box<1>(0, 16)); });
	const auto alloc_2nd = all_instrs.select_unique<alloc_instruction_record>(
	    [](const alloc_instruction_record& alloc) { return alloc.buffer_allocation->box == box_cast<3>(box<1>(16, 32)); });
	const auto resize_copy_from_1st = all_instrs.select_unique<copy_instruction_record>();
	CHECK(resize_copy_from_1st->source_allocation_id == alloc_1st->allocation_id);

	CHECK(all_instrs.select_unique("alloc 2nd").is_concurrent_with(all_instrs.select_unique("resize")));
}

TEST_CASE("overlapping accessors with read + discard_write modes are equivalent to a read_write access",
    "[instruction_graph_generator][instruction-graph][memory]") //
{
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer<float, 1>(32);
	ictx.device_compute(range(1)).name("alloc").discard_write(buf, acc::fixed<1>({0, 16})).submit();
	ictx.device_compute(range(1)).name("alloc").discard_write(buf, acc::fixed<1>({16, 16})).submit();
	ictx.device_compute(range(1)).name("read + discard_write").read(buf, acc::all()).discard_write(buf, acc::all()).submit();
	ictx.device_compute(range(1)).name("consume").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto both_alloc_kernels = all_instrs.select_all<device_kernel_instruction_record>("alloc");
	CHECK(both_alloc_kernels.count() == 2);
	const auto read_discard_write_kernel = all_instrs.select_unique<device_kernel_instruction_record>("read + discard_write");
	const auto consume_kernel = all_instrs.select_unique<device_kernel_instruction_record>("consume");

	CHECK(read_discard_write_kernel.transitive_predecessors_across<copy_instruction_record>().contains(both_alloc_kernels));
	CHECK(consume_kernel.predecessors() == read_discard_write_kernel);
}

TEST_CASE("device kernels report global memory traffic estimate based on range mappers", "[instruction_graph_generator][instruction-graph][memory]") {
	constexpr size_t num_devices = 2;
	constexpr size_t buffer_size = 1024;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* my nid */, num_devices);
	auto buf_1 = ictx.create_buffer<float, 1>(buffer_size, true /* host initialized */);
	auto buf_2 = ictx.create_buffer<float, 1>(buffer_size, true /* host initialized */);
	ictx.device_compute(range(buffer_size)).name("all read").read(buf_1, acc::all()).submit();
	ictx.device_compute(range(buffer_size)).name("1:1 read").read(buf_1, acc::one_to_one()).submit();
	ictx.device_compute(range(buffer_size)).name("1:1 read_write").read_write(buf_1, acc::one_to_one()).submit();
	ictx.device_compute(range(buffer_size)).name("1:1 discard_write + all_read").discard_write(buf_1, acc::one_to_one()).read(buf_1, acc::all()).submit();
	ictx.device_compute(range(buffer_size)).name("reduce").reduce(buf_2, false /* include_current_buffer_value */).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	for(const auto& all_read : all_instrs.select_all<device_kernel_instruction_record>("all_read").iterate()) {
		CHECK(all_read->estimated_global_memory_traffic_bytes == buffer_size * sizeof(float));
	}
	for(const auto& o2o_read : all_instrs.select_all<device_kernel_instruction_record>("1:1 read").iterate()) {
		CHECK(o2o_read->estimated_global_memory_traffic_bytes == buffer_size / num_devices * sizeof(float));
	}
	for(const auto& o2o_read_write : all_instrs.select_all<device_kernel_instruction_record>("1:1 read_write").iterate()) {
		CHECK(o2o_read_write->estimated_global_memory_traffic_bytes == buffer_size / num_devices * 2 * sizeof(float));
	}
	for(const auto& write_plus_read : all_instrs.select_all<device_kernel_instruction_record>("1:1 discard_write + all_read").iterate()) {
		CHECK(write_plus_read->estimated_global_memory_traffic_bytes == (buffer_size / num_devices + buffer_size) * sizeof(float));
	}
	for(const auto& reduce : all_instrs.select_all<device_kernel_instruction_record>("reduce").iterate()) {
		CHECK(reduce->estimated_global_memory_traffic_bytes == buffer_size / num_devices * sizeof(float));
	}
}
