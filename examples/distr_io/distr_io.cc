#include <cstdio>
#include <random>
#include <vector>

#include <celerity.h>
#include <hdf5.h>


static std::pair<hid_t, hid_t> allocation_window_to_dataspace(const celerity::buffer_allocation_window<float, 2>& layout) {
	hsize_t file_size[2], file_start[2];
	hsize_t allocation_size[2], allocation_start[2];
	hsize_t count[2];
	for(int d = 0; d < 2; ++d) {
		file_size[d] = layout.get_buffer_range()[d];
		file_start[d] = layout.get_window_offset_in_buffer()[d];
		allocation_size[d] = layout.get_allocation_range()[d];
		allocation_start[d] = layout.get_window_offset_in_allocation()[d];
		count[d] = layout.get_window_range()[d];
	}

	auto file_space = H5Screate_simple(2, file_size, nullptr);
	H5Sselect_hyperslab(file_space, H5S_SELECT_SET, file_start, nullptr, count, nullptr);
	auto allocation_space = H5Screate_simple(2, allocation_size, nullptr);
	H5Sselect_hyperslab(allocation_space, H5S_SELECT_SET, allocation_start, nullptr, count, nullptr);

	return {file_space, allocation_space};
}


static void read_hdf5_file(celerity::distr_queue& q, const celerity::buffer<float, 2>& buffer, const char* file_name) {
	q.submit([=](celerity::handler& cgh) {
		celerity::accessor a{buffer, cgh, celerity::experimental::access::even_split<2>{}, celerity::write_only_host_task, celerity::no_init};
		cgh.host_task(celerity::experimental::collective, [=](celerity::experimental::collective_partition part) {
			auto plist = H5Pcreate(H5P_FILE_ACCESS);
			H5Pset_fapl_mpio(plist, part.get_collective_mpi_comm(), MPI_INFO_NULL);
			auto file = H5Fopen(file_name, H5F_ACC_RDONLY, plist);
			H5Pclose(plist);

			plist = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);

			auto allocation_window = a.get_allocation_window(part);
			auto [file_space, allocation_space] = allocation_window_to_dataspace(allocation_window);
			auto dataset = H5Dopen(file, "data", H5P_DEFAULT);
			H5Dread(dataset, H5T_NATIVE_FLOAT, allocation_space, file_space, plist, allocation_window.get_allocation());

			H5Dclose(dataset);
			H5Sclose(allocation_space);
			H5Sclose(file_space);
			H5Fclose(file);
			H5Pclose(plist);
		});
	});
}


static void write_hdf5_file(celerity::distr_queue& q, const celerity::buffer<float, 2>& buffer, const char* file_name) {
	q.submit([=](celerity::handler& cgh) {
		celerity::accessor a{buffer, cgh, celerity::experimental::access::even_split<2>{}, celerity::read_only_host_task};
		cgh.host_task(celerity::experimental::collective, [=](celerity::experimental::collective_partition part) {
			auto plist = H5Pcreate(H5P_FILE_ACCESS);
			H5Pset_fapl_mpio(plist, part.get_collective_mpi_comm(), MPI_INFO_NULL);
			auto file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist);
			H5Pclose(plist);

			plist = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);

			auto allocation_window = a.get_allocation_window(part);
			auto [file_space, allocation_space] = allocation_window_to_dataspace(allocation_window);
			auto dataset = H5Dcreate(file, "data", H5T_NATIVE_FLOAT, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dwrite(dataset, H5T_NATIVE_FLOAT, allocation_space, file_space, plist, allocation_window.get_allocation());

			H5Dclose(dataset);
			H5Sclose(allocation_space);
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

	if((argc == 3 || argc == 4) && strcmp(argv[1], "--generate") == 0) {
		std::vector<float> initial(N * N);
		unsigned long seed = 1234567890;
		if(argc == 4) { seed = std::stoul(argv[3]); }
		auto gen = std::minstd_rand{seed};
		auto dist = std::uniform_real_distribution{-1.0f, 1.0f};
		std::generate(initial.begin(), initial.end(), [&] { return dist(gen); });
		celerity::buffer<float, 2> out(initial.data(), celerity::range<2>{N, N});

		celerity::distr_queue q;
		write_hdf5_file(q, out, argv[2]);
		return EXIT_SUCCESS;
	}

	if(argc == 4 && strcmp(argv[1], "--transpose") == 0) {
		celerity::buffer<float, 2> in(celerity::range<2>{N, N});
		celerity::buffer<float, 2> out(celerity::range<2>{N, N});

		celerity::distr_queue q;

		read_hdf5_file(q, in, argv[2]);

		q.submit([=](celerity::handler& cgh) {
			celerity::accessor a{in, cgh, celerity::access::one_to_one{}, celerity::read_only};
			celerity::accessor b{out, cgh, transposed, celerity::write_only, celerity::no_init};
			cgh.parallel_for<class transpose>(celerity::range<2>{N, N}, [=](celerity::item<2> item) {
				auto id = item.get_id();
				b[{id[1], id[0]}] = a[id];
			});
		});

		write_hdf5_file(q, out, argv[3]);
		return EXIT_SUCCESS;
	}

	if(argc == 4 && strcmp(argv[1], "--compare") == 0) {
		celerity::distr_queue q;

		celerity::buffer<float, 2> left(celerity::range<2>{N, N});
		celerity::buffer<float, 2> right(celerity::range<2>{N, N});
		celerity::buffer<bool> equal(1);

		read_hdf5_file(q, left, argv[2]);
		read_hdf5_file(q, right, argv[3]);

		q.submit([=](celerity::handler& cgh) {
			celerity::accessor a{left, cgh, celerity::access::all{}, celerity::read_only_host_task};
			celerity::accessor b{right, cgh, celerity::access::all{}, celerity::read_only_host_task};
			celerity::accessor e{equal, cgh, celerity::access::all{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(celerity::on_master_node, [=] {
				e[0] = true;
				for(size_t i = 0; i < N; ++i) {
					for(size_t j = 0; j < N; ++j) {
						e[0] &= a[{i, j}] == b[{i, j}];
					}
				}
			});
		});

		const auto files_equal = celerity::experimental::fence(q, equal).get().get_data()[0];
		fprintf(stderr, "=> Files are %sequal\n", files_equal ? "" : "NOT ");
		return files_equal ? EXIT_SUCCESS : EXIT_FAILURE;
	}

	fprintf(stderr,
	    "Usage: %s --generate <out-file>               to generate random data\n"
	    "       %s --transpose <in-file> <out-file>    to transpose\n"
	    "       %s --compare <in-file> <out-file>      to compare for equality\n",
	    argv[0], argv[0], argv[0]);
	return EXIT_FAILURE;
}
