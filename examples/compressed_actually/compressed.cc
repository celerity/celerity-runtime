#include <algorithm>
#include <numeric>

#include <celerity.h>

int main() {
	celerity::distr_queue queue;

	std::vector<float> data(1000);
	std::iota(data.begin(), data.end(), 0.0f);

	celerity::compressed<celerity::compression::quantization<float, int8_t>> compression_type;
	celerity::buffer<float, 1> buff(data.data(), data.size(), compressed_type);

	//
	// not shure if this kernel is working with compressed buffers each write might mean a complete recompression for upper and lower bound
	// since it could be that we have a new max or min value. This also means that we have to uncompress and recompress the whole buffer.
	//
	// One way to mitigate this is when we have a write buffer (read_write, write_only) we could just uncompress the whole buffer and then
	// recompress it after the kernel is done.
	//
	// Other way would be if we know the upper and lower bound of the buffer before hand we don't run into this problem.
	//
	// Third way (not to sure about this one) we can make fixed upper and lower bounds and clamp each value outside of the bounds to the
	// bounds. This would enshure that we don't have to uncompress and recompress the whole buffer but result in a loss of precision, outside
	// of the bounds.
	//
	// Improvement to all would be to have multiple smaller blocks of quantized compression and then we can just uncompress and recompress the block
	// To small and we would have more data than we started with. Block size at least as big as the compression size 8 bit -> 255 values ...
	//

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{buff, cgh, celerity::access::one_to_one{}, celerity::read_write};
		const auto range = buff.get_range();
		cgh.parallel_for<class quantized_kernel>(range, [=](celerity::item<1> item) { dw[item] = dw[item] / 1000.0f; });
	});

	// host_task_print numbers
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor acc{buff, cgh, celerity::access::all{}, celerity::read_only_host_task};
		const auto range = buff.get_range();
		cgh.host_task(celerity::on_master_node, [=] {
			for(size_t i = 0; i < range.get(0); ++i) {
				fmt::print("{} ", acc[i]);
			}
			fmt::print("\n");
		});
	});


	queue.slow_full_sync();

	return EXIT_SUCCESS;
}
