#pragma once

#include "handler.h"
#include "runtime.h"

namespace celerity {

template <typename MAF>
void with_master_access(MAF maf) {
	auto& queue = runtime::get_instance().get_queue();
	const task_id tid = queue.create_master_access_task(maf);
	master_access_prepass_handler mah(queue, tid);
	maf(mah);
}

} // namespace celerity
