#pragma once

#include "runtime.h"

namespace celerity {

template <typename MAF>
void with_master_access(MAF maf) {
	detail::runtime::get_instance().get_task_manager().create_master_access_task(maf);
}

} // namespace celerity
