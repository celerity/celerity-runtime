#include <cstdio>
#include <random>
#include <vector>

#include <celerity.h>
#include <hdf5.h>


static std::pair<hid_t, hid_t> host_memory_layout_to_dataspace(const celerity::host_memory_layout& layout) {
	auto& layout_dims = layout.get_dimensions();

	hsize_t file_size[2], file_start[2], file_count[2];
	hsize_t buffer_size[2], buffer_start[2], buffer_count[2];
	for(int d = 0; d < 2; ++d) {
		file_size[d] = layout_dims[d].get_global_size();
		file_start[d] = layout_dims[d].get_global_offset();
		buffer_size[d] = layout_dims[d].get_local_size();
		buffer_start[d] = layout_dims[d].get_local_offset();
		file_count[d] = buffer_count[d] = layout_dims[d].get_extent();
	}

	auto file_space = H5Screate_simple(2, file_size, nullptr);
	H5Sselect_hyperslab(file_space, H5S_SELECT_SET, file_start, nullptr, file_count, nullptr);
	auto buffer_space = H5Screate_simple(2, buffer_size, nullptr);
	H5Sselect_hyperslab(buffer_space, H5S_SELECT_SET, buffer_start, nullptr, buffer_count, nullptr);

	return {file_space, buffer_space};
}


static void read_hdf5_file(celerity::distr_queue& q, const celerity::buffer<float, 2>& buffer, const char* file_name) {
	q.submit([=](celerity::handler& cgh) {
		auto a = buffer.get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>(
		    cgh, celerity::experimental::access::even_split<2>());
		cgh.host_task(celerity::experimental::collective, [=](celerity::experimental::collective_partition part) {
			auto plist = H5Pcreate(H5P_FILE_ACCESS);
			H5Pset_fapl_mpio(plist, part.get_collective_mpi_comm(), MPI_INFO_NULL);
			auto file = H5Fopen(file_name, H5F_ACC_RDONLY, plist);
			H5Pclose(plist);

			plist = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);

			auto [data, layout] = a.get_host_memory(part);
			auto [file_space, buffer_space] = host_memory_layout_to_dataspace(layout);
			auto dataset = H5Dopen(file, "data", H5P_DEFAULT);
			H5Dread(dataset, H5T_NATIVE_FLOAT, buffer_space, file_space, plist, data);

			H5Dclose(dataset);
			H5Sclose(buffer_space);
			H5Sclose(file_space);
			H5Fclose(file);
			H5Pclose(plist);
		});
	});
}


static void write_hdf5_file(celerity::distr_queue& q, const celerity::buffer<float, 2>& buffer, const char* file_name) {
	q.submit([=](celerity::handler& cgh) {
		auto a = buffer.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::experimental::access::even_split<2>());
		cgh.host_task(celerity::experimental::collective, [=](celerity::experimental::collective_partition part) {
			auto plist = H5Pcreate(H5P_FILE_ACCESS);
			H5Pset_fapl_mpio(plist, part.get_collective_mpi_comm(), MPI_INFO_NULL);
			auto file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist);
			H5Pclose(plist);

			plist = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);

			auto [data, layout] = a.get_host_memory(part);
			auto [file_space, buffer_space] = host_memory_layout_to_dataspace(layout);
			auto dataset = H5Dcreate(file, "data", H5T_NATIVE_FLOAT, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dwrite(dataset, H5T_NATIVE_FLOAT, buffer_space, file_space, plist, data);

			H5Dclose(dataset);
			H5Sclose(buffer_space);
			H5Sclose(file_space);
			H5Fclose(file);
			H5Pclose(plist);
		});
	});
}


celerity::subrange<2> transposed(celerity::chunk<2> chnk) {
	using std::swap;
	assert(chnk.global_size[0] == chnk.global_size[1]);
	return {
	    {chnk.offset[1], chnk.offset[0]},
	    {chnk.range[1], chnk.range[0]},
	};
}


int main(int argc, char* argv[]) {
	const size_t N = 1000;

	celerity::detail::runtime::init(&argc, &argv);

	int rank = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if((argc == 3 || argc == 4) && strcmp(argv[1], "--generate") == 0) {
		std::vector<float> initial(N * N);
		unsigned long seed = 1234567890;
		if(argc == 4) { seed = std::stoul(argv[3]); }
		auto gen = std::minstd_rand{seed};
		auto dist = std::uniform_real_distribution{-1.0f, 1.0f};
		std::generate(initial.begin(), initial.end(), [&] { return dist(gen); });
		celerity::buffer<float, 2> out(initial.data(), cl::sycl::range<2>{N, N});

		celerity::distr_queue q;
		write_hdf5_file(q, out, argv[2]);
	} else if(argc == 4 && strcmp(argv[1], "--transpose") == 0) {
		celerity::buffer<float, 2> in(cl::sycl::range<2>{N, N});
		celerity::buffer<float, 2> out(cl::sycl::range<2>{N, N});

		celerity::distr_queue q;

		read_hdf5_file(q, in, argv[2]);

		q.submit([=](celerity::handler& cgh) {
			auto a = in.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
			auto b = out.get_access<cl::sycl::access::mode::discard_write>(cgh, transposed);
			cgh.parallel_for<class transpose>(cl::sycl::range<2>{N, N}, [=](cl::sycl::item<2> item) {
				auto id = item.get_id();
				b[{id[1], id[0]}] = a[id];
			});
		});

		write_hdf5_file(q, out, argv[3]);
	} else if(argc == 4 && strcmp(argv[1], "--compare") == 0) {
		bool equal = true;
		{
			celerity::distr_queue q;

			celerity::buffer<float, 2> left(cl::sycl::range<2>{N, N});
			celerity::buffer<float, 2> right(cl::sycl::range<2>{N, N});

			read_hdf5_file(q, left, argv[2]);
			read_hdf5_file(q, right, argv[3]);

			q.submit(celerity::allow_by_ref, [=, &equal](celerity::handler& cgh) {
				auto a = left.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<2>());
				auto b = right.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<2>());
				cgh.host_task(celerity::on_master_node, [=, &equal] {
					for(size_t i = 0; i < N; ++i) {
						for(size_t j = 0; j < N; ++j) {
							equal &= a[{i, j}] == b[{i, j}];
						}
					}
				});
			});
		}

		if(rank == 0) { fprintf(stderr, "=> Files are %sequal\n", equal ? "" : "NOT "); }
	} else {
		fprintf(stderr,
		    "Usage: %s --generate <out-file>               to generate random data\n"
		    "       %s --transpose <in-file> <out-file>    to transpose\n"
		    "       %s --compare <in-file> <out-file>      to compare for equality\n",
		    argv[0], argv[0], argv[0]);
		return EXIT_FAILURE;
	}
}
