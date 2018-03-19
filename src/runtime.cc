#include "runtime.h"

#include <mpi.h>

namespace celerity {

std::unique_ptr<runtime> runtime::instance = nullptr;

void runtime::init(int* argc, char** argv[]) {
	instance = std::unique_ptr<runtime>(new runtime(argc, argv));
}

runtime& runtime::get_instance() {
	if(instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
	return *instance;
}

runtime::runtime(int* argc, char** argv[]) {
	MPI_Init(argc, argv);

	// int world_size;
	// MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	is_master = world_rank == 0;
}

runtime::~runtime() {
	MPI_Finalize();
}

void runtime::TEST_do_work() {
	assert(queue != nullptr);
	if(is_master) {
		queue->debug_print_task_graph();
		queue->TEST_execute_deferred();
		queue->build_command_graph();
	} else {
		std::cout << "Worker is idle" << std::endl;
	}
}

void runtime::register_queue(distr_queue* queue) {
	if(this->queue != nullptr) { throw std::runtime_error("Only one celerity::distr_queue can be created per process"); }
	this->queue = queue;
}

} // namespace celerity
