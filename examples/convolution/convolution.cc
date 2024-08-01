#include <cassert>
#include <cstdio>
#include <vector>

#include <celerity.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"


bool is_on_boundary(celerity::range<2> range, size_t filter_size, celerity::id<2> id) {
	return (id[0] < (filter_size / 2) || id[1] < (filter_size / 2) || id[0] > range[0] - (filter_size / 2) - 1 || id[1] > range[1] - (filter_size / 2) - 1);
}

int main(int argc, char* argv[]) {
	if(argc != 2) {
		fprintf(stderr, "Usage: %s <image file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	std::vector<sycl::float3> image_input;
	int image_width = 0, image_height = 0, image_channels = 0;
	{
		uint8_t* image_data = stbi_load(argv[1], &image_width, &image_height, &image_channels, 3);
		assert(image_data != nullptr);
		image_input.resize(image_height * image_width);
		for(auto y = 0; y < image_height; ++y) {
			for(auto x = 0; x < image_width; ++x) {
				const auto idx = y * image_width * 3 + x * 3;
				image_input[y * image_width + x] = {image_data[idx + 0] / 255.f, image_data[idx + 1] / 255.f, image_data[idx + 2] / 255.f};
			}
		}
		stbi_image_free(image_data);
	}

	constexpr int filter_size = 16;
	constexpr float sigma = 3.f;
	constexpr float pi = 3.141592f;

	std::vector<float> gaussian_matrix(filter_size * filter_size);
	for(size_t j = 0; j < filter_size; ++j) {
		for(size_t i = 0; i < filter_size; ++i) {
			const auto x = i - (filter_size / 2);
			const auto y = j - (filter_size / 2);
			const auto value = std::exp(-1.f * (x * x + y * y) / (2 * sigma * sigma)) / (2 * pi * sigma * sigma);
			gaussian_matrix[j * filter_size + i] = value;
		}
	}

	celerity::distr_queue queue;

	celerity::buffer<sycl::float3, 2> image_input_buf(image_input.data(), celerity::range<2>(image_height, image_width));
	celerity::buffer<sycl::float3, 2> image_tmp_buf(celerity::range<2>(image_height, image_width));

	celerity::buffer<float, 2> gaussian_mat_buf(gaussian_matrix.data(), celerity::range<2>(filter_size, filter_size));

	// Do a gaussian blur
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor in{image_input_buf, cgh, celerity::access::neighborhood{filter_size / 2, filter_size / 2}, celerity::read_only};
		celerity::accessor gauss{gaussian_mat_buf, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor out{image_tmp_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};

		cgh.parallel_for<class gaussian_blur>(celerity::range<2>(image_height, image_width), [=, fs = filter_size](celerity::item<2> item) {
			using sycl::float3;
			if(is_on_boundary(celerity::range<2>(image_height, image_width), fs, item)) {
				out[item] = float3(0.f, 0.f, 0.f);
				return;
			}

			float3 sum = float3(0.f, 0.f, 0.f);
			for(auto y = -(fs / 2); y < fs / 2; ++y) {
				for(auto x = -(fs / 2); x < fs / 2; ++x) {
					sum += gauss[celerity::id<2>(fs / 2 + y, fs / 2 + x)] * in[{item[0] + y, item[1] + x}];
				}
			}
			out[item] = sum;
		});
	});

	celerity::buffer<sycl::float3, 2> image_output_buf(celerity::range<2>(image_height, image_width));

	// Now apply a sharpening kernel
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor in{image_tmp_buf, cgh, celerity::access::neighborhood{1, 1}, celerity::read_only};
		celerity::accessor out{image_output_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};

		cgh.parallel_for<class sharpen>(celerity::range<2>(image_height, image_width), [=, fs = filter_size](celerity::item<2> item) {
			using sycl::float3;
			if(is_on_boundary(celerity::range<2>(image_height, image_width), fs, item)) {
				out[item] = float3(0.f, 0.f, 0.f);
				return;
			}

			float3 sum = 5.f * in[item];
			sum -= in[{item[0] - 1, item[1]}];
			sum -= in[{item[0] + 1, item[1]}];
			sum -= in[{item[0], item[1] - 1}];
			sum -= in[{item[0], item[1] + 1}];
			out[item].x() = sycl::fmin(1.f, sum.x());
			out[item].y() = sycl::fmin(1.f, sum.y());
			out[item].z() = sycl::fmin(1.f, sum.z());
		});
	});

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor out{image_output_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};

		cgh.host_task(celerity::on_master_node, [=] {
			std::vector<uint8_t> image_output(image_width * image_height * 3);
			for(size_t y = 0; y < static_cast<size_t>(image_height); ++y) {
				for(size_t x = 0; x < static_cast<size_t>(image_width); ++x) {
					const auto idx = y * image_width * 3 + x * 3;
					image_output[idx + 0] = static_cast<uint8_t>(static_cast<float>(out[{y, x}].x()) * 255.f);
					image_output[idx + 1] = static_cast<uint8_t>(static_cast<float>(out[{y, x}].y()) * 255.f);
					image_output[idx + 2] = static_cast<uint8_t>(static_cast<float>(out[{y, x}].z()) * 255.f);
				}
			}
			stbi_write_png("./output.png", image_width, image_height, 3, image_output.data(), 0);
		});
	});

	return EXIT_SUCCESS;
}
