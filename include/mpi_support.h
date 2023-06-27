#pragma once

#include <mpi.h>

namespace celerity::detail::mpi_support {

constexpr int TAG_CMD = 0;
constexpr int TAG_DATA_TRANSFER = 1;
constexpr int TAG_TELEMETRY = 2;
constexpr int TAG_PRINT_GRAPH = 3;

class data_type {
  public:
	explicit data_type(MPI_Datatype dt) : m_dt(dt) {}
	data_type(const data_type&) = delete;
	data_type& operator=(const data_type&) = delete;
	~data_type() { MPI_Type_free(&m_dt); }

	operator MPI_Datatype() const { return m_dt; }

  private:
	MPI_Datatype m_dt;
};

} // namespace celerity::detail::mpi_support
