#pragma once

#include <utility>
#include <vector>

#include <mpi.h>

namespace celerity {
namespace detail {
	namespace mpi_support {

		constexpr int TAG_CMD = 0;
		constexpr int TAG_DATA_TRANSFER = 1;
		constexpr int TAG_TELEMETRY = 2;

		class single_use_data_type {
		  public:
			single_use_data_type() = default;
			single_use_data_type(MPI_Datatype dt) : dt(dt){};

			single_use_data_type(single_use_data_type&& other) noexcept { *this = std::move(other); }
			single_use_data_type& operator=(single_use_data_type&& other) noexcept {
				if(this != &other) {
					dt = other.dt;
					other.dt = MPI_DATATYPE_NULL;
				}
				return *this;
			}

			single_use_data_type(const single_use_data_type& other) = delete;
			single_use_data_type& operator=(const single_use_data_type& other) = delete;

			MPI_Datatype operator*() const { return dt; }

			~single_use_data_type() {
				if(dt != MPI_DATATYPE_NULL) { MPI_Type_free(&dt); }
			}

		  private:
			MPI_Datatype dt = MPI_DATATYPE_NULL;
		};

		/**
		 * @brief Constructs a new MPI data type for a particular list of blocks.
		 *
		 * The returned data type uses MPI_BYTE internally, with block displacements set to the given pointers, i.e. using the type
		 * operates directly on the objects pointed to. This is useful e.g. when transferring multiple objects that don't exist in a contiguous memory region.
		 *
		 * @param blocks A list pairs of an object size (in bytes) and a pointer to the object
		 * @returns A RAII-wrapped MPI data type
		 */
		single_use_data_type build_single_use_composite_type(const std::vector<std::pair<size_t, void*>>& blocks);

	} // namespace mpi_support
} // namespace detail
} // namespace celerity
