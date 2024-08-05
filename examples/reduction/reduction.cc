#include <cassert>
#include <cstdio>

#include <celerity.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"


sycl::float4 srgb_to_rgb(sycl::float4 srgb) {
	const auto linearize = [](float u) {
		if(u <= 0.04045f) {
			return u / 12.92f;
		} else {
			return sycl::pow((u + 0.055f) / 1.055f, 2.4f);
		}
	};
	return sycl::float4{
	    linearize(srgb.r()),
	    linearize(srgb.g()),
	    linearize(srgb.b()),
	    0,
	};
}

sycl::float4 rgb_to_srgb(sycl::float4 linear) {
	const auto compress = [](float u) {
		if(u <= 0.0031308f) {
			return 12.92f * u;
		} else {
			return 1.055f * sycl::pow(u, 1.f / 2.4f) - 0.055f;
		}
	};
	return sycl::float4{
	    compress(linear.r()),
	    compress(linear.g()),
	    compress(linear.b()),
	    0,
	};
}


// Reads an image, finds minimum/maximum pixel values, stretches the histogram to increase contrast, and saves the resulting image to output.jpg.
int main(int argc, char* argv[]) {
	if(argc != 2) {
		fprintf(stderr, "Usage: %s <image file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	int image_width = 0;
	int image_height = 0;
	int image_channels = 0;
	std::unique_ptr<uint8_t, decltype((stbi_image_free))> srgb_255_data{stbi_load(argv[1], &image_width, &image_height, &image_channels, 4), stbi_image_free};
	assert(srgb_255_data != nullptr);

	celerity::distr_queue q;

	celerity::range<2> image_size{static_cast<size_t>(image_height), static_cast<size_t>(image_width)};
	celerity::buffer<sycl::uchar4, 2> srgb_255_buf{reinterpret_cast<const sycl::uchar4*>(srgb_255_data.get()), image_size};
	celerity::buffer<sycl::float4, 2> rgb_buf{image_size};
	celerity::buffer<float, 0> min_buf;
	celerity::buffer<float, 0> max_buf;

	q.submit([&](celerity::handler& cgh) {
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor rgb_acc{rgb_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		const auto min_r = celerity::reduction(min_buf, cgh, sycl::minimum<float>(), celerity::property::reduction::initialize_to_identity{});
		const auto max_r = celerity::reduction(max_buf, cgh, sycl::maximum<float>(), celerity::property::reduction::initialize_to_identity{});
		cgh.parallel_for<class linearize_and_accumulate>(image_size, min_r, max_r, [=](celerity::item<2> item, auto& min, auto& max) {
			const auto rgb = srgb_to_rgb(srgb_255_acc[item].convert<float>() / 255.0f);
			rgb_acc[item] = rgb;
			for(int i = 0; i < 3; ++i) {
				min.combine(rgb[i]);
				max.combine(rgb[i]);
			}
		});
	});

	q.submit([&](celerity::handler& cgh) {
		celerity::accessor min{min_buf, cgh, celerity::read_only_host_task};
		celerity::accessor max{max_buf, cgh, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=] { printf("Before contrast stretch: min = %f, max = %f\n", *min, *max); });
	});

	q.submit([&](celerity::handler& cgh) {
		celerity::accessor rgb_acc{rgb_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor min{min_buf, cgh, celerity::read_only};
		celerity::accessor max{max_buf, cgh, celerity::read_only};
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class correct_and_compress>(image_size, [=](celerity::item<2> item) {
			auto rgb = rgb_acc[item];
			for(int i = 0; i < 3; ++i) {
				rgb[i] = (rgb[i] - *min) / (*max - *min);
			}
			// we want to sycl::round() here, but this causes a segfault in CUDA 12.1 with AdaptiveCpp
			srgb_255_acc[item] = (rgb_to_srgb(rgb) * 255.0f).convert<unsigned char>();
		});
	});

	q.submit([&](celerity::handler& cgh) {
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=] { stbi_write_jpg("output.jpg", image_width, image_height, 4, srgb_255_acc.get_pointer(), 90); });
	});

	return EXIT_SUCCESS;
}
