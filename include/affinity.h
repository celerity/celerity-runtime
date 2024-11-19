#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// The goal of this thread pinning mechanism, when enabled, is to ensure that threads which benefit from fast communication
// are pinned to cores that are close to each other in terms of cache hierarchy.
// It currently accomplishes this by pinning threads to cores in a round-robin fashion according to their order in the `thread_type` enum.
//
// In terms of interface design, the goal is to provide a very simple entry point (`pin_this_thread`), that is safe to use from any thread at any time,
// and does not require polluting any other modules with state related to thread pinning. The `thread_pinner` RAII class offers the only way to manage the
// lifetime of the pinning mechanism, and prevents misuse. The implementation safely removes pinning from any thread it previously pinned on teardown.
//
// TODO: A future extension would be to respect NUMA for threads performing memory operations, but this requires in-depth knowledge of the system's topology.
namespace celerity::detail::thread_pinning {

constexpr uint32_t thread_type_step = 10000;

// The threads Celerity interacts with ("user") and creates (everything else), identified for the purpose of pinning.
// Note: this is not an enum class to make interactions such as specifying `first_backend_worker+i` easier
enum thread_type : uint32_t {
	application = 0 * thread_type_step,
	scheduler = 1 * thread_type_step,
	executor = 2 * thread_type_step,
	first_backend_worker = 3 * thread_type_step,
	first_host_queue = 4 * thread_type_step,
	max = 5 * thread_type_step,
};
std::string thread_type_to_string(const thread_type t_type);

// User-level configuration of the thread pinning mechanism (set by the user via environment variables)
struct environment_configuration {
	bool enabled = true; // we want thread pinning to be enabled by default
	uint32_t starting_from_core = 1;
	std::vector<uint32_t> hardcoded_core_ids;
};

// Parses and validates the environment variable string, returning the corresponding configuration
environment_configuration parse_validate_env(const std::string_view str);

// Configures the pinning mechanism
// For now, only "standard" threads are pinned
//   these are threads that benefit from rapid communication between each other,
//   i.e. scheduler -> executor -> backend workers
// Extensible for future use where some threads might benefit from NUMA-aware per-GPU pinning
struct runtime_configuration {
	// Whether or not to perform pinning
	bool enabled = false;

	// Number of devices that will need corresponding threads
	uint32_t num_devices = 1;
	// Whether backend device submission threads are used and need to have cores allocated to them
	bool use_backend_device_submission_threads = true;

	// Number of processes running in legacy mode
	uint32_t num_legacy_processes = 1;
	// Process index of current process running in legacy mode
	uint32_t legacy_process_index = 0;

	// The core to start pinning "standard" threads to
	uint32_t standard_core_start_id = 1;

	// If set, this list of core ids will be used for pinning instead of the default round-robin scheme
	// The list must contain exactly as many elements as there are standard threads
	std::vector<uint32_t> hardcoded_core_ids = {}; // NOLINT(readability-redundant-member-init) -- to allow partial designated init elsewhere
};

// An RAII class for managing thread pinning
// Only one instance of this class may be active at a time (this is enforced by the implementation)
// Threads pinned by this class will be unpinned when the instance is destroyed
class thread_pinner {
  public:
	thread_pinner(const runtime_configuration& cfg);
	~thread_pinner();
	thread_pinner(const thread_pinner&) = delete;
	thread_pinner& operator=(const thread_pinner&) = delete;
	thread_pinner(thread_pinner&&) = default;
	thread_pinner& operator=(thread_pinner&&) = default;
};

// Pins the invoking thread of type `t_type` according to the current configuration
// This is a no-op if the thread pinning machinery is not currently initialized (by a `thread_pinner` instance)
void pin_this_thread(const thread_type t_type);

} // namespace celerity::detail::thread_pinning
