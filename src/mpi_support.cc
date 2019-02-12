#include "mpi_support.h"

#include <vector>

namespace celerity {
namespace detail {
	namespace mpi_support {

		single_use_data_type build_single_use_composite_type(const std::vector<std::pair<size_t, void*>>& blocks) {
			std::vector<int> block_lengths;
			block_lengths.reserve(blocks.size());
			std::vector<MPI_Aint> disps;
			disps.reserve(blocks.size());
			for(auto& b : blocks) {
				block_lengths.push_back(static_cast<int>(b.first));
				disps.push_back(reinterpret_cast<MPI_Aint>(b.second));
			}
			std::vector<MPI_Datatype> block_types(blocks.size(), MPI_BYTE);
			MPI_Datatype data_type;
			MPI_Type_create_struct(static_cast<int>(blocks.size()), block_lengths.data(), disps.data(), block_types.data(), &data_type);
			MPI_Type_commit(&data_type);
			return data_type;
		}

	} // namespace mpi_support
} // namespace detail
} // namespace celerity
