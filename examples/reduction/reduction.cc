#include <cassert>
#include <cstdio>

#include <celerity.h>

// Conditional shouldn't really be required, but ComputeCpp 2.8.0 Experimental fails trying to compile SSE intrinsics in device code
#ifndef __SYCL_DEVICE_ONLY__
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

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


// We could use two reduction variables to calculate minimum and maximum, but some SYCL implementations currently only support a single reductio per kernel.
// Instead we build a combined minimum-maximum operation, with the side effect that we have to call `combine(x, x)` instead of `combine(x)` below.
const auto minmax = [](sycl::float2 a, sycl::float2 b) { //
	return sycl::float2{sycl::min(a[0], b[0]), sycl::max(a[1], b[1])};
};

const sycl::float2 minmax_identity{INFINITY, -INFINITY};


// Reads an image, finds minimum/maximum pixel values, stretches the histogram to increase contrast, and saves the resulting image to output.jpg.
int main(int argc, char* argv[]) {
	if(argc != 2) {
		fprintf(stderr, "Usage: %s <image file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	int image_width = 0, image_height = 0, image_channels = 0;
	std::unique_ptr<uint8_t, decltype((stbi_image_free))> srgb_255_data{stbi_load(argv[1], &image_width, &image_height, &image_channels, 4), stbi_image_free};
	assert(srgb_255_data != nullptr);

	celerity::distr_queue q;

	celerity::range<2> image_size{static_cast<size_t>(image_height), static_cast<size_t>(image_width)};
	celerity::buffer<sycl::uchar4, 2> srgb_255_buf{reinterpret_cast<const sycl::uchar4*>(srgb_255_data.get()), image_size};
	celerity::buffer<sycl::float4, 2> lab_buf{image_size};
	celerity::buffer<sycl::float2, 1> minmax_buf{celerity::range{1}};

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor rgb_acc{lab_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		auto minmax_r = celerity::reduction(minmax_buf, cgh, minmax_identity, minmax, celerity::property::reduction::initialize_to_identity{});
		cgh.parallel_for<class linearize_and_accumulate>(image_size, minmax_r, [=](celerity::item<2> item, auto& minmax) {
			const auto rgb = srgb_to_rgb(srgb_255_acc[item].convert<float>() / 255.0f);
			rgb_acc[item] = rgb;
			for(int i = 0; i < 3; ++i) {
				minmax.combine({rgb[i], rgb[i]});
			}
		});
	});

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor minmax_acc{minmax_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=] { printf("Before contrast stretch: min = %f, max = %f\n", minmax_acc[0][0], minmax_acc[0][1]); });
	});

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor rgb_acc{lab_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor minmax_acc{minmax_buf, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class correct_and_compress>(image_size, [=](celerity::item<2> item) {
			const auto min = minmax_acc[0][0], max = minmax_acc[0][1];
			auto rgb = rgb_acc[item];
			for(int i = 0; i < 3; ++i) {
				rgb[i] = (rgb[i] - min) / (max - min);
			}
			srgb_255_acc[item] = sycl::round((rgb_to_srgb(rgb) * 255.0f)).convert<unsigned char>();
		});
	});

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=] { stbi_write_jpg("output.jpg", image_width, image_height, 4, srgb_255_acc.get_pointer(), 90); });
	});

	return EXIT_SUCCESS;
}
