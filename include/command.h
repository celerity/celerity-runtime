#pragma once

#include "ranges.h"
#include "types.h"

namespace celerity {

enum class command { NOP, COMPUTE, MASTER_ACCESS, PUSH, AWAIT_PUSH, SHUTDOWN };

struct command_subrange {
	size_t offset0 = 0;
	size_t offset1 = 0;
	size_t offset2 = 0;
	size_t range0 = 1;
	size_t range1 = 1;
	size_t range2 = 1;

	command_subrange() = default;

	command_subrange(const subrange<1>& sr) {
		offset0 = sr.offset[0];
		range0 = sr.range[0];
	}

	command_subrange(const subrange<2>& sr) {
		offset0 = sr.offset[0];
		offset1 = sr.offset[1];
		range0 = sr.range[0];
		range1 = sr.range[1];
	}

	command_subrange(const subrange<3>& sr) {
		offset0 = sr.offset[0];
		offset1 = sr.offset[1];
		offset2 = sr.offset[2];
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

} // namespace celerity
