#pragma once

#include "subrange.h"
#include "types.h"

namespace celerity {

enum class command { NOP, COMPUTE, PULL, AWAIT_PULL, SHUTDOWN };

struct command_subrange {
	size_t offset0 = 0;
	size_t offset1 = 0;
	size_t offset2 = 0;
	size_t range0 = 0;
	size_t range1 = 0;
	size_t range2 = 0;

	command_subrange() = default;

	command_subrange(const subrange<1>& sr) {
		offset0 = sr.start[0];
		range0 = sr.range[0];
	}

	command_subrange(const subrange<2>& sr) {
		offset0 = sr.start[0];
		offset1 = sr.start[1];
		range0 = sr.range[0];
		range1 = sr.range[1];
	}

	command_subrange(const subrange<3>& sr) {
		offset0 = sr.start[0];
		offset1 = sr.start[1];
		offset2 = sr.start[2];
		range0 = sr.range[0];
		range1 = sr.range[1];
		range2 = sr.range[2];
	}

	bool operator==(const command_subrange& rhs) const {
		return offset0 == rhs.offset0 && offset1 == rhs.offset1 && offset2 == rhs.offset2 && range0 == rhs.range0 && range1 == rhs.range1
		       && range2 == rhs.range2;
	}
};

struct nop_data {};

struct compute_data {
	command_subrange chunk;
};

struct pull_data {
	buffer_id bid;
	node_id source;
	command_subrange subrange;
};

struct await_pull_data {
	buffer_id bid;
	node_id target;
	task_id target_tid;
	command_subrange subrange;
};

struct shutdown_data {};

union command_data {
	nop_data nop;
	compute_data compute;
	pull_data pull;
	await_pull_data await_pull;
	shutdown_data shutdown;
};

/**
 * A command package is what is actually transferred between nodes.
 */
struct command_pkg {
	task_id tid;
	command cmd;
	command_data data;

	command_pkg() : data({}) {}
	command_pkg(task_id tid, command cmd, command_data data) : tid(tid), cmd(cmd), data(data) {}
};

} // namespace celerity
