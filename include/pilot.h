#pragma once

#include "grid.h"
#include "types.h"


namespace celerity::detail {

/// Metadata exchanged in preparation for a peer-to-peer data transfer with send_instruction / receive_instruction (and cousins). Pilots allow the receiving
/// side to issue MPI_*recv instructions directly to the appropriate target memory and (optionally) stride without additional staging or buffering.
struct pilot_message {
	detail::message_id id = -1;
	detail::transfer_id transfer_id;
	detail::box<3> box;
};

/// A pilot message as packaged on the sender side.
struct outbound_pilot {
	node_id to = -1;
	pilot_message message;
};

/// A pilot message as packaged on the receiver side.
struct inbound_pilot {
	node_id from = -1;
	pilot_message message;
};

} // namespace celerity::detail
