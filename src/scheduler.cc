#include "scheduler.h"

#include "graph_generator.h"

namespace celerity {
namespace detail {

	scheduler::scheduler(std::shared_ptr<graph_generator> ggen, size_t num_nodes) : ggen(ggen), num_nodes(num_nodes) {}

	void scheduler::startup() { schd_thrd = std::thread(&scheduler::schedule, this); }

	void scheduler::shutdown() {
		should_shutdown = true;
		notify_task_created(); // Hack: We hijack this functionality to wake the worker thread up

		if(schd_thrd.joinable()) { schd_thrd.join(); }
	}

	void scheduler::notify_task_created() {
		{
			std::lock_guard<std::mutex> lk(tasks_available_mutex);
			unscheduled_tasks++;
		}
		tasks_available_cv.notify_one();
	}

	void scheduler::schedule() {
		std::unique_lock<std::mutex> lk(tasks_available_mutex);

		while(true) {
			// TODO: We currently operate in lockstep with the main thread. This is less than ideal.
			tasks_available_cv.wait(lk, [this] { return unscheduled_tasks > 0; });
			if(should_shutdown && unscheduled_tasks == 1) return;

			const auto tid = ggen->get_unbuilt_task();
			assert(tid);
			ggen->build_task(*tid);
			ggen->flush(*tid);
			unscheduled_tasks--;
		}
	}

} // namespace detail
} // namespace celerity
