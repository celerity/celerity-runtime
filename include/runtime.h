#pragma once

#include <memory>

#include "buffer_state.h"
#include "distr_queue.h"
#include "graph.h"
#include "types.h"

namespace celerity {

class runtime {
  public:
	static void init(int* argc, char** argv[]);
	static runtime& get_instance();

	~runtime();

	void TEST_do_work();
	void register_queue(distr_queue* queue);

	template <int Dims>
	buffer_id register_buffer(cl::sycl::range<Dims> size) {
		const buffer_id bid = buffer_count++;
		valid_buffer_regions[bid] = std::make_unique<detail::buffer_state<Dims>>(size, num_nodes);
		return bid;
	}

  private:
	static std::unique_ptr<runtime> instance;

	distr_queue* queue = nullptr;
	size_t num_nodes;
	bool is_master;

	size_t buffer_count = 0;

	// This is a data structe view which encodes where (= on which node) valid
	// regions of a buffer can be found. A valid region is any region that has not
	// been written to on another node.
	// NOTE: This represents the buffer regions after all commands in the current
	// command graph have been completed.
	std::unordered_map<buffer_id, std::unique_ptr<detail::buffer_state_base>> valid_buffer_regions;

	command_dag command_graph;

	runtime(int* argc, char** argv[]);
	runtime(const runtime&) = delete;
	runtime(runtime&&) = delete;

	void build_command_graph();
};

} // namespace celerity
