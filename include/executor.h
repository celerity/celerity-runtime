#pragma once

#include "types.h"

#include <memory>

namespace celerity::detail {

struct host_object_instance;
class instruction;
struct outbound_pilot;
class runtime_reduction;

class executor {
  public:
	class delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		virtual void horizon_reached(task_id tid) = 0;
		virtual void epoch_reached(task_id tid) = 0;
	};

	executor() = default;
	executor(const executor&) = delete;
	executor(executor&&) = delete;
	executor& operator=(const executor&) = delete;
	executor& operator=(executor&&) = delete;
	virtual ~executor() = default;

	virtual void wait() = 0;

	virtual void submit_instruction(const instruction* instr) = 0;
	virtual void submit_pilot(const outbound_pilot& pilot) = 0;

	virtual void announce_user_allocation(allocation_id aid, void* ptr) = 0;
	virtual void announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) = 0;
	virtual void announce_reduction(reduction_id rid, std::unique_ptr<runtime_reduction> reduction) = 0;
};

} // namespace celerity::detail
