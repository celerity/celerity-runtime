#pragma once

#include "async_event.h"
#include "pilot.h"
#include "utils.h"

namespace celerity::detail {

class communicator {
  public:
	class collective_group {
	  public:
		collective_group(const collective_group&) = delete;
		collective_group(collective_group&&) = delete;
		collective_group& operator=(const collective_group&) = delete;
		collective_group& operator=(collective_group&&) = delete;

		virtual collective_group* clone() = 0;
		virtual void barrier() = 0;

	  protected:
		collective_group() = default;
		~collective_group() = default;
	};

	struct stride {
		range<3> allocation;
		celerity::subrange<3> subrange;
		size_t element_size = 1;

		friend bool operator==(const stride& lhs, const stride& rhs) {
			return lhs.allocation == rhs.allocation && lhs.subrange == rhs.subrange && lhs.element_size == rhs.element_size;
		}
		friend bool operator!=(const stride& lhs, const stride& rhs) { return !(lhs == rhs); }
	};

	virtual ~communicator() = default;

	virtual size_t get_num_nodes() const = 0;
	virtual node_id get_local_node_id() const = 0;

	virtual void send_outbound_pilot(const outbound_pilot& pilot) = 0;
	[[nodiscard]] virtual std::vector<inbound_pilot> poll_inbound_pilots() = 0;

	[[nodiscard]] virtual async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) = 0;
	[[nodiscard]] virtual async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) = 0;

	virtual collective_group* get_collective_root() = 0;

  protected:
	communicator() = default;
	communicator(const communicator&) = default;
	communicator(communicator&&) = default;
	communicator& operator=(const communicator&) = default;
	communicator& operator=(communicator&&) = default;
};

} // namespace celerity::detail

template <>
struct std::hash<celerity::detail::communicator::stride> {
	size_t operator()(const celerity::detail::communicator::stride& stride) const {
		size_t h = 0;
		for(int d = 0; d < 3; ++d) {
			celerity::detail::utils::hash_combine(h, stride.allocation[d]);
			celerity::detail::utils::hash_combine(h, stride.subrange.offset[d]);
			celerity::detail::utils::hash_combine(h, stride.subrange.range[d]);
		}
		celerity::detail::utils::hash_combine(h, stride.element_size);
		return h;
	}
};
