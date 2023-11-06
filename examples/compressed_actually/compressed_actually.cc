#include <algorithm>
#include <numeric>

#include <celerity.h>

template <typename Compressed, typename Uncompressed>
Compressed quantized_number(const Uncompressed number, const Uncompressed upper_bound, const Uncompressed lower_bound) {
	return static_cast<Compressed>((number - lower_bound) / (upper_bound)*std::numeric_limits<Compressed>::max());
}

template <typename Compressed, typename Uncompressed>
Uncompressed dequntized_number(const Compressed number, const Uncompressed upper_bound, const Uncompressed lower_bound) {
	return static_cast<Uncompressed>(number) / static_cast<Uncompressed>(std::numeric_limits<Compressed>::max()) * (upper_bound - lower_bound) + lower_bound;
}

template <typename Compressed, typename Uncompressed>
std::vector<Compressed> compress_vector(const std::vector<Uncompressed>& data, const Uncompressed upper_bound, const Uncompressed lower_bound) {
	std::vector<uint8_t> data_two(data.size());

	for(size_t i = 0; i < data.size(); ++i) {
		data_two[i] = quantized_number<uint8_t>(data[i], upper_bound, lower_bound);
	}

	return std::move(data_two);
}

int main() {
	celerity::distr_queue queue;

	std::vector<float> data(1000);
	std::iota(data.begin(), data.end(), 0.0f);

	std::vector<uint8_t> data_two = compress_vector<uint8_t>(data, 1000.0f, 0.0f);
	celerity::buffer<uint8_t, 1> buff(data_two.data(), data_two.size());

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{buff, cgh, celerity::access::one_to_one{}, celerity::read_write};
		const auto range = buff.get_range();
		cgh.parallel_for<class quantized_kernel>(range, [=](celerity::item<1> item) {
			// bounds change here because of writeing to the buffer.
			// This means that we have to uncompress and recompress the whole buffer, additionally we have to check what the new bounds are.
			dw[item] = quantized_number<uint8_t>(dequntized_number(dw[item], 1000.0f, 0.0f) / 1000.0f, 1.0f, 0.0f);
		});
	});

	// host_task_print numbers
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{buff, cgh, celerity::access::all{}, celerity::read_only_host_task};
		const auto range = buff.get_range();
		cgh.host_task(celerity::on_master_node, [=] {
			for(size_t i = 0; i < range.get(0); ++i) {
				fmt::print("{} ", dequntized_number(dw[i], 1.0f, 0.0f));
			}
			fmt::print("\n");
		});
	});


	queue.slow_full_sync();

	return EXIT_SUCCESS;
}
