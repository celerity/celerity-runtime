#include <iostream>
#include <string>

#include <celerity.h>

int main() {
	const std::string input_str = "Ifmmp!Xpsme\"\x01";

	celerity::distr_queue queue;
	celerity::buffer<char> str_buffer(input_str.data(), input_str.size());
	celerity::debug::set_buffer_name(str_buffer, "string");

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor str_acc{str_buffer, cgh, celerity::access::one_to_one{}, celerity::read_write};
		cgh.parallel_for(str_buffer.get_range(), [=](celerity::item<1> item) { str_acc[item] -= 1; });
	});

	auto output = queue.fence(str_buffer);
	std::cout << output.get().get_data() << std::endl;
}
