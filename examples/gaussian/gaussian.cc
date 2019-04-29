#include <cstdlib>

#include <CL/sycl.hpp>
#include <celerity.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {
	int image_width = 0, image_height = 0, image_channels = 0;
	uint8_t* image_data = stbi_load("./Lenna.png", &image_width, &image_height, &image_channels, 3);
	assert(image_data != nullptr);

	std::vector<cl::sycl::float3> image_input(image_width * image_height);
	for(auto y = 0; y < image_height; ++y) {
		for(auto x = 0; x < image_width; ++x) {
			const auto idx = y * image_width * 3 + x * 3;
			image_input[y * image_width + x] = {image_data[idx + 0] / 255.f, image_data[idx + 1] / 255.f, image_data[idx + 2] / 255.f};
		}
	}

	constexpr int KERNEL_SIZE = 16;
	constexpr float sigma = 3.f;
	constexpr float PI = 3.141592f;

	std::vector<float> gaussian_matrix(KERNEL_SIZE * KERNEL_SIZE);
	for(auto j = 0u; j < KERNEL_SIZE; ++j) {
		for(auto i = 0u; i < KERNEL_SIZE; ++i) {
			const auto x = i - (KERNEL_SIZE / 2);
			const auto y = j - (KERNEL_SIZE / 2);
			const auto value = std::exp(-1.f * (x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
			gaussian_matrix[j * KERNEL_SIZE + i] = value;
		}
	}

	try {
		celerity::distr_queue queue;

		celerity::buffer<cl::sycl::float3, 2> image_input_buf(image_input.data(), cl::sycl::range<2>(image_height, image_width));
		celerity::buffer<cl::sycl::float3, 2> image_tmp_buf(cl::sycl::range<2>(image_height, image_width));
		celerity::buffer<cl::sycl::float3, 2> image_output_buf(cl::sycl::range<2>(image_height, image_width));

		celerity::buffer<float, 2> gaussian_mat_buf(gaussian_matrix.data(), cl::sycl::range<2>(KERNEL_SIZE, KERNEL_SIZE));

		// Do a gaussian blur
		// TODO: Due to some weird issue with Clang on Windows, we have to capture some of these values explicitly
		queue.submit([&, image_height, image_width, KERNEL_SIZE](celerity::handler& cgh) {
			auto in = image_input_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(KERNEL_SIZE / 2, KERNEL_SIZE / 2));
			auto gauss = gaussian_mat_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2, 2>());
			auto out = image_tmp_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

			cgh.parallel_for<class gaussian_blur>(
			    cl::sycl::range<2>(image_height, image_width), [=, ih = image_height, iw = image_width, ks = KERNEL_SIZE](cl::sycl::item<2> item) {
				    using cl::sycl::float3;

				    // Unfortunately we don't support images (yet?), so we can't just us a clamping sampler
				    if(item[0] < (ks / 2) || item[1] < (ks / 2) || item[0] > ih - (ks / 2) - 1 || item[1] > iw - (ks / 2) - 1) {
					    out[item] = float3(0.f, 0.f, 0.f);
					    return;
				    }

				    float3 sum = float3(0.f, 0.f, 0.f);
				    for(auto y = -(ks / 2); y < ks / 2; ++y) {
					    for(auto x = -(ks / 2); x < ks / 2; ++x) {
						    sum += gauss[{(size_t)ks / 2 + y, (size_t)ks / 2 + x}] * in[{item[0] + y, item[1] + x}];
					    }
				    }
				    out[item] = sum;
			    });
		});

		// Now apply a sharpening kernel
		queue.submit([&, image_height, image_width](celerity::handler& cgh) {
			auto in = image_tmp_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));
			auto out = image_output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
			cgh.parallel_for<class sharpen>(cl::sycl::range<2>(image_height, image_width), [=, iw = image_width, ih = image_height](cl::sycl::item<2> item) {
				using cl::sycl::float3;
				if(item[0] == 0u || item[1] == 0u || item[0] == ih - 1u || item[1] == iw - 1u) {
					out[item] = float3(0.f, 0.f, 0.f);
					return;
				}

				float3 sum = 5.f * in[item];
				sum -= in[{item[0] - 1, item[1]}];
				sum -= in[{item[0] + 1, item[1]}];
				sum -= in[{item[0], item[1] - 1}];
				sum -= in[{item[0], item[1] + 1}];
				out[item] = float3(std::min(1.f, (float)sum.x()), std::min(1.f, (float)sum.y()), std::min(1.f, (float)sum.z()));
			});
		});

		celerity::with_master_access([&](celerity::handler& cgh) {
			auto out = image_output_buf.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(image_height, image_width));

			cgh.run([=]() {
				for(size_t y = 0; y < (size_t)image_height; ++y) {
					for(size_t x = 0; x < (size_t)image_width; ++x) {
						const auto idx = y * image_width * 3 + x * 3;
						image_data[idx + 0] = static_cast<uint8_t>(static_cast<float>(out[{y, x}].x() * 255.f));
						image_data[idx + 1] = static_cast<uint8_t>(static_cast<float>(out[{y, x}].y() * 255.f));
						image_data[idx + 2] = static_cast<uint8_t>(static_cast<float>(out[{y, x}].z() * 255.f));
					}
				}
				stbi_write_png("./output.png", image_width, image_height, 3, image_data, 0);
			});
		});
	} catch(std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	} catch(cl::sycl::exception& e) {
		std::cerr << "SYCL Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	stbi_image_free(image_data);
	return EXIT_SUCCESS;
}
