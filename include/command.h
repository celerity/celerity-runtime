#pragma once

#include "ranges.h"
#include "types.h"

namespace celerity {
namespace detail {

	enum class command { NOP, COMPUTE, MASTER_ACCESS, PUSH, AWAIT_PUSH, SHUTDOWN };
	constexpr const char* command_string[] = {"NOP", "COMPUTE", "MASTER_ACCESS", "PUSH", "AWAIT_PUSH", "SHUTDOWN"};

	struct command_subrange {
		size_t offset[3] = {0, 0, 0};
		size_t range[3] = {1, 1, 1};

		command_subrange() = default;

		command_subrange(const subrange<1>& sr) {
			offset[0] = sr.offset[0];
			range[0] = sr.range[0];
		}

		command_subrange(const subrange<2>& sr) {
			offset[0] = sr.offset[0];
			offset[1] = sr.offset[1];
			range[0] = sr.range[0];
			range[1] = sr.range[1];
		}

		command_subrange(const subrange<3>& sr) {
			offset[0] = sr.offset[0];
			offset[1] = sr.offset[1];
			offset[2] = sr.offset[2];
			range[0] = sr.range[0];
			range[1] = sr.range[1];
			range[2] = sr.range[2];
		}

		bool operator==(const command_subrange& rhs) const {
			return offset[0] == rhs.offset[0] && offset[1] == rhs.offset[1] && offset[2] == rhs.offset[2] && range[0] == rhs.range[0]
			       && range[1] == rhs.range[1] && range[2] == rhs.range[2];
		}

		operator subrange<3>() const { return {cl::sycl::id<3>(offset[0], offset[1], offset[2]), cl::sycl::range<3>(range[0], range[1], range[2])}; }
	};

	struct nop_data {};

	struct compute_data {
		command_subrange subrange;
	};

	struct master_access_data {};

	struct push_data {
		buffer_id bid;
		node_id target;
		command_subrange subrange;
	};

	struct await_push_data {
		buffer_id bid;
		node_id source;
		command_id source_cid;
		command_subrange subrange;
	};

	struct shutdown_data {};

	union command_data {
		nop_data nop;
		compute_data compute;
		master_access_data master_access;
		push_data push;
		await_push_data await_push;
		shutdown_data shutdown;
	};

	/**
	 * A command package is what is actually transferred between nodes.
	 */
	struct command_pkg {
		task_id tid;
		command_id cid;
		command cmd;
		command_data data;

		command_pkg() : data({}) {}
		command_pkg(task_id tid, command_id cid, command cmd, command_data data) : tid(tid), cid(cid), cmd(cmd), data(data) {}
	};

} // namespace detail
} // namespace celerity
