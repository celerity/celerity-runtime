#include "command_graph_generator.h"
#include "grid.h"
#include "ranges.h"
#include "region_map.h"
#include "types.h"

static const auto epoch_task = celerity::detail::task::make_epoch(celerity::detail::task_id{0}, celerity::detail::epoch_action::none, nullptr);
static const auto epoch_cmd = std::make_unique<celerity::detail::epoch_command>(
    celerity::detail::command_id{123}, epoch_task.get(), celerity::detail::epoch_action::none, std::vector<celerity::detail::reduction_id>());

int main() {
	[[maybe_unused]] const celerity::detail::task_id tid = 10;
	[[maybe_unused]] const celerity::detail::buffer_id bid = 11;
	[[maybe_unused]] const celerity::detail::node_id nid = 12;
	[[maybe_unused]] const celerity::detail::command_id cid = 13;
	[[maybe_unused]] const celerity::detail::collective_group_id cgid = 14;
	[[maybe_unused]] const celerity::detail::reduction_id rid = 15;
	[[maybe_unused]] const celerity::detail::host_object_id hoid = 16;
	[[maybe_unused]] const celerity::detail::hydration_id hyid = 17;
	[[maybe_unused]] const celerity::detail::transfer_id trid{18, 19};
	[[maybe_unused]] const celerity::detail::transfer_id reduction_trid{20, 21, 22};
	[[maybe_unused]] const celerity::detail::memory_id mid = 23;
	[[maybe_unused]] const celerity::detail::raw_allocation_id raid = 24;
	[[maybe_unused]] const celerity::detail::device_id did = 25;
	[[maybe_unused]] const celerity::detail::instruction_id iid = 26;
	[[maybe_unused]] const celerity::detail::allocation_id aid(27, 28);
	[[maybe_unused]] const celerity::detail::message_id msgid(34);

	[[maybe_unused]] const celerity::id<3> id(1, 2, 3);
	[[maybe_unused]] const celerity::range<3> range(1, 2, 3);
	[[maybe_unused]] const celerity::subrange<3> subrange(celerity::id(1, 2, 3), celerity::range(4, 5, 6));
	[[maybe_unused]] const celerity::chunk<3> chunk(celerity::id(1, 2, 3), celerity::range(4, 5, 6), celerity::range(7, 8, 9));
	[[maybe_unused]] const celerity::nd_range<3> nd_range(celerity::range(2, 4, 6), celerity::range(1, 2, 3), celerity::id(7, 8, 9));
	[[maybe_unused]] const celerity::detail::box<3> box(celerity::id(1, 2, 3), celerity::id(4, 5, 6));
	[[maybe_unused]] const celerity::detail::region<3> empty_region;
	[[maybe_unused]] const celerity::detail::region<3> region({
	    celerity::detail::box(celerity::id(1, 2, 3), celerity::id(4, 5, 6)),
	    celerity::detail::box(celerity::id(11, 2, 3), celerity::id(14, 5, 6)),
	    celerity::detail::box(celerity::id(21, 2, 3), celerity::id(24, 5, 6)),
	});

	[[maybe_unused]] celerity::detail::region_map<int> region_map(celerity::range<3>(10, 10, 10));
	region_map.update_box(celerity::detail::box<3>({1, 1, 1}, {5, 5, 5}), 42);
	region_map.update_box(celerity::detail::box<3>({1, 1, 1}, {3, 3, 3}), 69);
	region_map.update_box(celerity::detail::box<3>({1, 1, 1}, {2, 2, 2}), 1337);

	[[maybe_unused]] const celerity::detail::region_map<int> region_map_0d(celerity::range<3>(1, 1, 1), 42);

	[[maybe_unused]] const celerity::detail::write_command_state wcs_fresh(epoch_cmd.get());
	[[maybe_unused]] celerity::detail::write_command_state wcs_stale(epoch_cmd.get());
	wcs_stale.mark_as_stale();
	[[maybe_unused]] const celerity::detail::write_command_state wcs_replicated(epoch_cmd.get(), true /* replicated */);

	// tell GDB to break here so we can examine locals
	__builtin_trap();
}
