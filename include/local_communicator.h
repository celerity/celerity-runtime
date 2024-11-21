#pragma once

#include "async_event.h"
#include "communicator.h"
#include "pilot.h"
#include "types.h"
#include "utils.h"

#include <cstddef>
#include <memory>
#include <vector>


namespace celerity::detail {

/// Single-node stub implementation for the `communicator` interface.
class local_communicator final : public communicator {
  public:
	size_t get_num_nodes() const override { return 1; }
	node_id get_local_node_id() const override { return 0; }

	void send_outbound_pilot(const outbound_pilot& pilot) override { unavailable(); }
	[[nodiscard]] std::vector<inbound_pilot> poll_inbound_pilots() override { return {}; }

	[[nodiscard]] async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) override { unavailable(); }
	[[nodiscard]] async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) override { unavailable(); }

	[[nodiscard]] std::unique_ptr<communicator> collective_clone() override { return std::make_unique<local_communicator>(); }
	void collective_barrier() override {}

  private:
	[[noreturn]] static void unavailable() { utils::panic("Invoking an unsupported operation on local_communicator"); }
};

} // namespace celerity::detail
