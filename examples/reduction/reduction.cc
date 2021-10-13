#include <cassert>
#include <cstdio>

#include <celerity.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>


cl::sycl::float4 srgb_to_rgb(cl::sycl::float4 srgb) {
	const auto linearize = [](float u) {
		if(u <= 0.04045f) {
			return u / 12.92f;
		} else {
			return cl::sycl::pow((u + 0.055f) / 1.055f, 2.4f);
		}
	};
	return cl::sycl::float4{
	    linearize(srgb.r()),
	    linearize(srgb.g()),
	    linearize(srgb.b()),
	    0,
	};
}

cl::sycl::float4 rgb_to_srgb(cl::sycl::float4 linear) {
	const auto compress = [](float u) {
		if(u <= 0.0031308f) {
			return 12.92f * u;
		} else {
			return 1.055f * cl::sycl::pow(u, 1.f / 2.4f) - 0.055f;
		}
	};
	return cl::sycl::float4{
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

	int image_width = 0, image_height = 0, image_channels = 0;
	std::unique_ptr<uint8_t, decltype((stbi_image_free))> srgb_255_data{stbi_load(argv[1], &image_width, &image_height, &image_channels, 4), stbi_image_free};
	assert(srgb_255_data != nullptr);

	celerity::distr_queue q;

	cl::sycl::range<2> image_size{static_cast<size_t>(image_height), static_cast<size_t>(image_width)};
	celerity::buffer<cl::sycl::uchar4, 2> srgb_255_buf{reinterpret_cast<const cl::sycl::uchar4*>(srgb_255_data.get()), image_size};
	celerity::buffer<cl::sycl::float4, 2> lab_buf{image_size};
	celerity::buffer<float, 1> min_value_buf{cl::sycl::range{1}};
	celerity::buffer<float, 1> max_value_buf{cl::sycl::range{1}};

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor rgb_acc{lab_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		auto min_value_r = celerity::reduction(min_value_buf, cgh, cl::sycl::minimum<float>{}, cl::sycl::property::reduction::initialize_to_identity{});
		auto max_value_r = celerity::reduction(max_value_buf, cgh, cl::sycl::maximum<float>{}, cl::sycl::property::reduction::initialize_to_identity{});
		cgh.parallel_for<class linearize_and_accumulate>(image_size, min_value_r, max_value_r, [=](celerity::item<2> item, auto& min_value, auto& max_value) {
			const auto rgb = srgb_to_rgb(srgb_255_acc[item].convert<float>() / 255.0f);
			rgb_acc[item] = rgb;
			for(int i = 0; i < 3; ++i) {
				min_value.combine(rgb[i]);
				max_value.combine(rgb[i]);
			}
		});
	});

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor min_value_acc{min_value_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor max_value_acc{max_value_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=] { printf("Before contrast stretch: min = %f, max = %f\n", min_value_acc[0], max_value_acc[0]); });
	});

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor rgb_acc{lab_buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
		celerity::accessor min_value_acc{min_value_buf, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor max_value_acc{max_value_buf, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class correct_and_compress>(image_size, [=](celerity::item<2> item) {
			const auto min = min_value_acc[0], max = max_value_acc[0];
			auto rgb = rgb_acc[item];
			for(int i = 0; i < 3; ++i) {
				rgb[i] = (rgb[i] - min) / (max - min);
			}
			srgb_255_acc[item] = cl::sycl::round((rgb_to_srgb(rgb) * 255.0f)).convert<unsigned char>();
		});
	});

	q.submit([=](celerity::handler& cgh) {
		celerity::accessor srgb_255_acc{srgb_255_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=] { stbi_write_jpg("output.jpg", image_width, image_height, 4, srgb_255_acc.get_pointer(), 90); });
	});

	return EXIT_SUCCESS;
}
