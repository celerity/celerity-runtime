#include <vector>

#include <celerity.h>
#include <tuple>

struct accessor_wrapper {
	template <celerity::access_mode Mode, celerity::access_mode TagMode, typename Functor>
	accessor_wrapper(celerity::buffer<size_t, 1> a, celerity::buffer<size_t, 1> b, celerity::handler& cgh, const Functor& rmfn,
	    const celerity::detail::access_tag<TagMode, Mode, celerity::target::device> tag)
	    : m_acc_a(a, cgh, rmfn, tag), m_acc_b(b, cgh, rmfn, tag) {}

	std::tuple<celerity::range<1>, celerity::range<1>> get_allocation_range() const { return {m_acc_a.get_allocation_range(), m_acc_b.get_allocation_range()}; }

  private:
	celerity::accessor<size_t, 1, celerity::access_mode::discard_read_write, celerity::target::device> m_acc_a;
	celerity::accessor<size_t, 1, celerity::access_mode::discard_read_write, celerity::target::device> m_acc_b;
};

namespace celerity {
struct test_comp {};

template <typename DataT, int Dims, typename F, compression_category Category>
class buffer<DataT, Dims, celerity::compressed<F, Category>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;

	buffer(const DataT* data, range<Dims> range) : buffer(std::move(data), range), m_data(data, data + range.size()) {
		std::cout << "Buffer created" << std::endl;
	}

	buffer(range<Dims> range) : base(range), m_data(range) { std::cout << "Buffer created" << std::endl; }

	buffer<DataT, Dims, compression::uncompressed>& get_underlying_buffer() { return m_data; }

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range) : base(data.data(), range), m_data(std::move(data)) {}

	buffer<DataT, Dims, compression::uncompressed> m_data;
};

template <typename DataT, int Dims, celerity::access_mode Mode, typename F>
class accessor<DataT, Dims, Mode, target::device, F> : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit, compression_category Category>
	accessor(buffer<T, D, celerity::compressed<F, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : base(buff, cgh, rmfn, tag), m_acc_b(buff.get_underlying_buffer(), cgh, rmfn, celerity::read_write, celerity::no_init) {}

	template <typename T, int D, typename Functor, access_mode TagMode, compression_category Category>
	accessor(buffer<T, D, celerity::compressed<F, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_acc_b(buff.get_underlying_buffer(), cgh, rmfn, celerity::read_write, celerity::no_init) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit, compression_category Category>
	accessor(buffer<DataT, Dims, celerity::compressed<F, Category>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::device> tag,
	    const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_acc_b(buff.get_underlying_buffer(), cgh, celerity::read_write, celerity::no_init) {}

	// template <typename DataAccess>
	// inline auto decompress_data(celerity::nd_item<3> item, DataAccess& data_point_available, const Intype min, const int tile_size) const {
	// 	return decompress_data(item, 0, 0, data_point_available, min, tile_size);
	// }

	std::tuple<celerity::range<1>, celerity::range<1>> get_allocation_range_t() const { return {this->get_allocation_range(), m_acc_b.get_allocation_range()}; }

  private:
	celerity::accessor<size_t, 1, celerity::access_mode::discard_read_write, celerity::target::device> m_acc_b;
};
} // namespace celerity


int main(int argc, char* argv[]) {
	constexpr size_t buf_size = 1024;

	celerity::queue queue;
	celerity::buffer<size_t, 1> buf(buf_size);

	celerity::buffer<size_t, 1> buf_a(buf_size);
	celerity::buffer<size_t, 1> buf_b(buf_size);

	celerity::buffer<size_t, 1, celerity::compressed<celerity::test_comp, celerity::compression_category::element_wise>> buf_c(buf_size);

	// celerity::buffer<size_t, 1, celerity::test_comp, celerity::compression_category::element_wise> buf_c(buf_size);

	// Initialize buffer in a distributed device kernel
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		accessor_wrapper wrapper{buf_a, buf_b, cgh, celerity::access::one_to_one{},
		    celerity::detail::access_tag<celerity::access_mode::discard_read_write, celerity::access_mode::discard_read_write, celerity::target::device>{}};
		celerity::accessor<size_t, 1, celerity::access_mode::write, celerity::target::device, celerity::test_comp> a{
		    buf_c, cgh, celerity::access::one_to_one{}, celerity::write_only};
		cgh.parallel_for<class write_linear_id>(buf.get_range(), [=](celerity::item<1> item) {
			b[item] = item.get_linear_id();
			// print get_allocation_range from wrapper
			auto [range_a, range_b] = wrapper.get_allocation_range();
			auto [a_range, b_range] = a.get_allocation_range_t();
			if(item.get_linear_id() == 0 || item.get_linear_id() == buf_size - 1) {
				printf("fAllocation range a: (%ld), Allocation range b: (%ld)\n", range_a[0], range_b[0]);
				printf("fAccessor a allocation range: (%ld), Accessor b allocation range: (%ld)\n", a_range[0], b_range[0]);
			}
		});
	});

	queue.wait(celerity::experimental::barrier);

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		accessor_wrapper wrapper{buf_a, buf_b, cgh, celerity::access::one_to_one{},
		    celerity::detail::access_tag<celerity::access_mode::discard_read_write, celerity::access_mode::discard_read_write, celerity::target::device>{}};
		celerity::accessor<size_t, 1, celerity::access_mode::write, celerity::target::device, celerity::test_comp> a{
		    buf_c, cgh, celerity::access::neighborhood<1>{2}, celerity::write_only};
		cgh.parallel_for<class write_great_id>(buf.get_range(), [=](celerity::item<1> item) {
			b[item] = item.get_linear_id();
			// print get_allocation_range from wrapper
			auto [range_a, range_b] = wrapper.get_allocation_range();
			auto [a_range, b_range] = a.get_allocation_range_t();
			if(item.get_linear_id() == 0 || item.get_linear_id() == buf_size - 1) {
				printf("Allocation range a: (%ld), Allocation range b: (%ld)\n", range_a[0], range_b[0]);
				printf("Accessor a allocation range: (%ld), Accessor b allocation range: (%ld)\n", a_range[0], b_range[0]);
			}
		});
	});

	// Process values on the host
	std::vector<size_t> host_buf(buf_size);
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::experimental::collective, [=, &host_buf](celerity::experimental::collective_partition) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100)); // give the synchronization more time to fail
			for(size_t i = 0; i < buf_size; i++) {
				host_buf[i] = 2 * b[i];
			}
		});
	});

	// Wait until both tasks have completed
	queue.wait();

	// At this point we can safely interact with host_buf from within the application thread
	bool valid = true;
	for(size_t i = 0; i < buf_size; i++) {
		if(host_buf[i] != 2 * i) {
			valid = false;
			break;
		}
	}

	return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
