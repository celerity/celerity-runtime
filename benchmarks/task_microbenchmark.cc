// Benchmarks a chain of connected tasks

#include <args.hxx>
#include <celerity.h>

#include "benchmark_utils.h"

enum class Topology { Soup, Chain, Map, Reduce };
constexpr const char* topologyNames[] = {"Soup", "Chain", "Map", "Reduce"};
inline const char* getTopologyName(Topology t) {
	return topologyNames[(int)t];
}

int getMinBufferSizeForTopology(Topology topology, int numTasks) {
	switch(topology) {
	case Topology::Soup: return 0;
	case Topology::Chain: return 1;
	case Topology::Map:
	case Topology::Reduce: return (numTasks + 1) / 2;
	}
}

struct Args {
	int numTasks = 1000;
	int bufferSize = 2048;
	Topology topology = Topology::Soup;
};

Args parseArgs(int argc, char** argv) {
	Args ret;

	args::ArgumentParser parser("Celerity Task and Command Microbenchmarks");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::Group commands(parser, "commands");
	args::Command chain(commands, "chain", "benchmark a chain of connected tasks");
	args::Command soup(commands, "soup", "benchmark a soup of unconnected tasks");
	args::Command map(commands, "map", "benchmark tasks branching out from a single root");
	args::Command reduce(commands, "reduce", "benchmark tasks converging towards a single result");
	args::Group arguments(parser, "task arguments", args::Group::Validators::DontCare, args::Options::Global);
	args::ValueFlag<int> numTasks(arguments, "num-tasks", "The number of tasks to generate", {'n'}, 1000);
	args::ValueFlag<int> bufferSize(arguments, "buffer-size", "Size of buffers used to establish task connections", {'b'}, 2048);

	Benchmark::get().addArguments(parser);

	try {
		parser.ParseCLI(argc, argv);
	} catch(args::Help) {
		std::cout << parser;
		exit(0);
	} catch(args::Error e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		exit(1);
	}

	ret.numTasks = numTasks.Get();
	ret.bufferSize = bufferSize.Get();
	if(chain) ret.topology = Topology::Chain;
	if(map) ret.topology = Topology::Map;
	if(reduce) ret.topology = Topology::Reduce;

	auto requiredSize = getMinBufferSizeForTopology(ret.topology, ret.numTasks);
	if(ret.bufferSize < requiredSize) {
		std::cerr << "Topology '" << getTopologyName(ret.topology) << "' requires a buffer size of at least " << requiredSize << " for " << ret.numTasks
		          << " tasks, but buffer size is set to " << ret.bufferSize << "." << std::endl;
		exit(2);
	}

	return ret;
}

int main(int argc, char** argv) {
	Args args = parseArgs(argc, argv);

	celerity::distr_queue queue;

	Benchmark::get().run([&]() {
		std::vector<float> host_data(args.bufferSize);
		celerity::buffer<float, 1> buffer(host_data.data(), args.bufferSize);

		if(args.topology == Topology::Soup || args.topology == Topology::Chain) {
			for(int t = 0; t < args.numTasks; ++t) {
				queue.submit([=](celerity::handler& cgh) {
					if(args.topology == Topology::Chain) {
						celerity::accessor acc{buffer, cgh, celerity::access::one_to_one(), celerity::read_write};
						cgh.parallel_for<class ChainKernel>(celerity::range<1>(args.bufferSize), [=](celerity::item<1> item) { acc[item]++; });
					} else {
						cgh.parallel_for<class SoupKernel>(celerity::range<1>(args.bufferSize), [=](celerity::item<1> item) {});
					}
				});
			}
		} else if(args.topology == Topology::Map || args.topology == Topology::Reduce) {
			celerity::buffer<float, 1> buffer2(host_data.data(), args.bufferSize);

			int numEpochs = std::log2(args.numTasks);
			int curEpochTasks = args.topology == Topology::Map ? 1 : 1 << numEpochs;
			int sentinelEpoch = args.topology == Topology::Map ? numEpochs - 1 : 0;
			int sentinelEpochMax = args.numTasks - (curEpochTasks - 1); // how many tasks to generate at the last/first epoch to reach exactly args.numTasks

			for(int e = 0; e < numEpochs; ++e) {
				int taskCount = curEpochTasks;
				if(e == sentinelEpoch) taskCount = sentinelEpochMax;

				// build tasks for this epoch
				for(int t = 0; t < taskCount; ++t) {
					queue.submit([=](celerity::handler& cgh) {
						// mappers constructed to build a binary (potentially inverted) tree
						auto read_mapper = [=](const celerity::chunk<1>& chunk) {
							return args.topology == Topology::Map ? celerity::subrange<1>(t / 2, 1) : celerity::subrange<1>(t * 2, 2);
						};
						auto write_mapper = [=](const celerity::chunk<1>& chunk) { return celerity::subrange<1>(t, 1); };
						celerity::accessor write_acc{buffer, cgh, write_mapper, celerity::write_only};
						celerity::accessor read_acc{buffer2, cgh, read_mapper, celerity::read_only};
						cgh.parallel_for<class TreeKernel>(celerity::range<1>(1), [=](celerity::item<1> item) { write_acc[item] = read_acc[item]; });
					});
				}

				// get ready for the next epoch
				if(args.topology == Topology::Map) {
					curEpochTasks *= 2;
				} else {
					curEpochTasks /= 2;
				}
				std::swap(buffer, buffer2);
			}
		}

		{ // basic verification
			// check that there are more than the requested number of tasks generated in this iteration, but less than 2x the requested number
			// (it will be more due to the initialization task and horizon tasks)
			static int prevTaskCount = 0;
			int totalTaskCount = celerity::detail::runtime::get_instance().get_task_manager().get_total_task_count() - prevTaskCount;
			prevTaskCount = celerity::detail::runtime::get_instance().get_task_manager().get_total_task_count();
			if(totalTaskCount < args.numTasks || totalTaskCount > args.numTasks * 2) {
				std::cerr << "Error: asked to generate " << args.numTasks << " tasks, but generated " << totalTaskCount << "." << std::endl;
			}
		}

		queue.slow_full_sync();
	});
}
