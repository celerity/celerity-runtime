#pragma once

#include <memory>

#include "distr_queue.h"

namespace celerity {

class runtime {
  public:
	static void init(int* argc, char** argv[]);
	static runtime& get_instance();

	~runtime();

	void TEST_do_work();
	void register_queue(distr_queue* queue);

  private:
	static std::unique_ptr<runtime> instance;

	distr_queue* queue = nullptr;
	bool is_master;

	runtime(int* argc, char** argv[]);
	runtime(const runtime&) = delete;
	runtime(runtime&&) = delete;
};

} // namespace celerity
