#pragma once

#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <CL/sycl.hpp>

#include "access_modes.h"
#include "buffer_storage.h"
#include "device_queue.h"
#include "ranges.h"
#include "region_map.h"
#include "types.h"
#include <unordered_set>

namespace celerity {
namespace detail {

	class raw_buffer_data;


	/**
	 * The buffer_manager keeps track of all Celerity buffers currently existing within the runtime.
	 *
	 * This includes both host and device buffers. Note that instead of relying on SYCL's host-side buffers,
	 * we keep separate copies that allow for more explicit control. All data accesses within device buffers
	 * are on the device or through explicit memory operations, meaning that a sufficiently optimized SYCL
	 * implementation would never have to allocate any host memory whatsoever. Users need to ensure that
	 * device buffers returned from the buffer_manager are also only being used on the device.
	 *
	 * Most operations of the buffer_manager are performed lazily. For example, upon registering a buffer,
	 * no memory is being allocated on either the host or device. Only when requesting an explicit range of
	 * a buffer on either side, an allocation takes place.
	 *
	 * The buffer_manager keeps track of buffer versioning for host and device buffers, performing coherence
	 * updates whenever necessary (again lazily, upon requesting a buffer). Any buffer returned can thus be
	 * assumed to be in its most up-to-date version.
	 *
	 * Importantly, the buffer_manager takes care of buffer "virtualization". This means that while a buffer
	 * can be registered with an arbitrary size, the actual allocation entirely depends on which subranges
	 * of a buffer end up being used. The registered buffer is called the "virtual buffer", while the allocated
	 * memory is called the "backing buffer".
	 *
	 * The backing buffer is resized whenever an access exceeds the current allocation.
	 * NOTE: Currently, only a single backing buffer exists per virtual buffer and side (host/device).
	 *		 This means that accessing two very distant subranges of the virtual buffer will cause the backing
	 *		 buffer to be resized to fit their entire bounding box.
	 * NOTE: Currently, for the duration of their lifetime, (backing) buffers ONLY ever GROW.
	 *
	 * Besides managing buffers for host or device access, the buffer manager also acts as an interface for
	 * incoming and outgoing data transfers, through the buffer_manager::set_buffer_data and
	 * buffer_manager::get_buffer_data functions. Incoming transfers are processed lazily.
	 *
	 * Importantly, when requesting access to a buffer on the host or device, the buffer_manager does not
	 * keep track on when this access has completed. Instead, it assumes that the effects of the
	 * access (e.g., using a writing access mode to update the buffer contents) take place immediately.
	 *
	 * Essentially, this means that any requests made to the buffer_manager are assumed to be operations
	 * that are currently allowed by the command graph.
	 *
	 * There are two important caveats that we need to deal with:
	 *
	 * - Reading from a buffer is no longer a const operation, as the buffer may need to be resized.
	 *   This means that two tasks that could be considered independent on a TDAG basis actually have an
	 *   implicit anti-dependency relationship.
	 *   Note: In this case "reading" refers not only to accessing a buffer with a read mode, but also
	 *	       calling get_buffer_data for an outgoing data transfer.
	 *
	 * - Since buffer accesses are considered to have immediate effect, requesting access to the same buffer
	 *   more than once form within a single CGF can have unintended consequences. For example, accessing a
	 *   buffer first with "discard_write" and followed by a "read" should result in a combined "write" mode.
	 *   However the effect of the discard_write is recorded immediately, and the buffer_manager will thus
	 *   wrongly assume that no coherence update for the "read" is required.
	 *
	 * Currently, these issues are handled through the buffer locking mechanism.
	 * See buffer_manager::try_lock, buffer_manager::unlock and buffer_manager::is_locked.
	 *
	 * FIXME: The current buffer locking mechanism limits task parallelism. Come up with a better solution.
	 */
	class buffer_manager {
	  public:
		enum class buffer_lifecycle_event { REGISTERED, UNREGISTERED };

		using buffer_lifecycle_callback = std::function<void(buffer_lifecycle_event, buffer_id)>;

		struct buffer_info {
			cl::sycl::range<3> range;
			bool is_host_initialized;
		};

		/**
		 * When requesting a host or device buffer through the buffer_manager, this is what is returned.
		 */
		template <typename DataT, int Dims, template <typename, int> class BufferT>
		struct access_info {
			/**
			 * This is the *currently used* backing buffer for the requested virtual buffer.
			 * This reference can become stale if the backing buffer needs to be resized by a subsequent access.
			 */
			BufferT<DataT, Dims>& buffer;

			/**
			 * This is the offset of the backing buffer relative to the requested virtual buffer.
			 */
			cl::sycl::id<Dims> offset;
		};

		using buffer_lock_id = size_t;

	  public:
		buffer_manager(device_queue& queue, buffer_lifecycle_callback lifecycle_cb);

		template <typename DataT, int Dims>
		buffer_id register_buffer(cl::sycl::range<3> range, const DataT* host_init_ptr = nullptr) {
			buffer_id bid;
			const bool is_host_initialized = host_init_ptr != nullptr;
			{
				assert(range.size() > 0);
				std::unique_lock lock(mutex);
				bid = buffer_count++;
				buffer_infos[bid] = buffer_info{range, is_host_initialized};
				newest_data_location.emplace(bid, region_map<data_location>(range, data_location::NOWHERE));

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
				buffer_types.emplace(bid, new buffer_type_guard<DataT, Dims>());
#endif
			}
			if(is_host_initialized) {
				// We need to access the full range for host-initialized buffers.
				auto info = get_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::discard_write, range, cl::sycl::id<3>(0, 0, 0));
				std::memcpy(info.buffer.get_pointer(), host_init_ptr, range.size() * sizeof(DataT));
			}
			lifecycle_cb(buffer_lifecycle_event::REGISTERED, bid);
			return bid;
		}

		/**
		 * @brief Unregisters a buffer, releasing the internally stored reference.
		 *
		 * This function must not be called while the runtime is still active, as Celerity currently does not know whether
		 * it is safe to release a buffer at any given point in time.
		 */
		void unregister_buffer(buffer_id bid) noexcept;

		/**
		 * @brief Checks whether the buffer with id \p bid has already been registered.
		 *
		 * This is useful in rare situations where worker nodes might receive data for buffers they haven't registered yet.
		 */
		bool has_buffer(buffer_id bid) const {
			std::shared_lock lock(mutex);
			return buffer_infos.count(bid) == 1;
		}

		bool has_active_buffers() const {
			std::shared_lock lock(mutex);
			return !buffer_infos.empty();
		}

		const buffer_info& get_buffer_info(buffer_id bid) const {
			std::shared_lock lock(mutex);
			assert(buffer_infos.find(bid) != buffer_infos.end());
			return buffer_infos.at(bid);
		}

		/**
		 * Returns a dense copy of the newest version data of the requested buffer range.
		 *
		 * This function is mainly intended for outgoing data transfers.
		 *
		 * NOTE: Currently this function might incur a host-side buffer allocation/resize.
		 *
		 * TODO:
		 * - Ideally we would transfer data directly out of the original buffer (at least on the host, need RDMA otherwise).
		 * - We'd have to consider the data striding in the MPI data type we build.
		 *
		 * @param bid
		 * @param offset
		 * @param range
		 */
		raw_buffer_data get_buffer_data(buffer_id bid, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range);

		/**
		 * Updates a buffer's content with the provided @p data.
		 *
		 * This update is performed lazily, the next time the updated subrange is requested on either the host or device.
		 *
		 * TODO: Consider doing eager updates directly into host memory. However:
		 * - Host buffer might not be large enough.
		 * - H->D transfers currently work better for contiguous copies.
		 */
		void set_buffer_data(buffer_id bid, cl::sycl::id<3> offset, raw_buffer_data&& data);

		template <typename DataT, int Dims>
		access_info<DataT, Dims, device_buffer> get_device_buffer(
		    buffer_id bid, cl::sycl::access::mode mode, const cl::sycl::range<3>& range, const cl::sycl::id<3>& offset) {
			std::unique_lock lock(mutex);
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			assert((buffer_types.at(bid)->has_type<DataT, Dims>()));
#endif
			assert((range_cast<3>(offset + range) <= buffer_infos.at(bid).range) == cl::sycl::range<3>(true, true, true));

			auto& old_buffer = buffers[bid].device_buf;
			backing_buffer new_buffer;

			if(!old_buffer.is_allocated()) {
				new_buffer = backing_buffer{std::make_unique<device_buffer_storage<DataT, Dims>>(range_cast<Dims>(range), queue.get_sycl_queue()), offset};
			} else {
				// FIXME: For large buffers we might not be able to store two copies in device memory at once.
				// Instead, we'd first have to transfer everything to the host and free the old buffer before allocating the new one.
				// TODO: What we CAN do however already is to free the old buffer early iff we're requesting a discard_* access!
				// (AND that access request covers the entirety of the old buffer!)
				const auto info = is_resize_required(old_buffer, range, offset);
				if(info.resize_required) {
					new_buffer = backing_buffer{
					    std::make_unique<device_buffer_storage<DataT, Dims>>(range_cast<Dims>(info.new_range), queue.get_sycl_queue()), info.new_offset};
				}
			}

			audit_buffer_access(bid, new_buffer.is_allocated(), mode);

			backing_buffer& target_buffer = new_buffer.is_allocated() ? new_buffer : old_buffer;
			const backing_buffer empty{};
			const backing_buffer& previous_buffer = new_buffer.is_allocated() ? old_buffer : empty;
			make_buffer_subrange_coherent(bid, mode, target_buffer, {offset, range}, previous_buffer);

			if(new_buffer.is_allocated()) { buffers[bid].device_buf = std::move(new_buffer); }

			return {dynamic_cast<device_buffer_storage<DataT, Dims>*>(buffers[bid].device_buf.storage.get())->get_device_buffer(),
			    id_cast<Dims>(buffers[bid].device_buf.offset)};
		}

		template <typename DataT, int Dims>
		access_info<DataT, Dims, host_buffer> get_host_buffer(
		    buffer_id bid, cl::sycl::access::mode mode, const cl::sycl::range<3>& range, const cl::sycl::id<3>& offset) {
			std::unique_lock lock(mutex);
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			assert((buffer_types.at(bid)->has_type<DataT, Dims>()));
#endif
			assert((range_cast<3>(offset + range) <= buffer_infos.at(bid).range) == cl::sycl::range<3>(true, true, true));

			auto& old_buffer = buffers[bid].host_buf;
			backing_buffer new_buffer;

			if(!old_buffer.is_allocated()) {
				new_buffer = backing_buffer{std::make_unique<host_buffer_storage<DataT, Dims>>(range_cast<Dims>(range)), offset};
			} else {
				const auto info = is_resize_required(old_buffer, range, offset);
				if(info.resize_required) {
					new_buffer = backing_buffer{std::make_unique<host_buffer_storage<DataT, Dims>>(range_cast<Dims>(info.new_range)), info.new_offset};
				}
			}

			audit_buffer_access(bid, new_buffer.is_allocated(), mode);

			backing_buffer& target_buffer = new_buffer.is_allocated() ? new_buffer : old_buffer;
			const backing_buffer empty{};
			const backing_buffer& previous_buffer = new_buffer.is_allocated() ? old_buffer : empty;
			make_buffer_subrange_coherent(bid, mode, target_buffer, {offset, range}, previous_buffer);

			if(new_buffer.is_allocated()) { buffers[bid].host_buf = std::move(new_buffer); }

			return {static_cast<host_buffer_storage<DataT, Dims>*>(buffers[bid].host_buf.storage.get())->get_host_buffer(),
			    id_cast<Dims>(buffers[bid].host_buf.offset)};
		}

		/**
		 * @brief Tries to lock the given list of @p buffers using the given lock @p id.
		 *
		 * If any of the buffers is currently locked, the locking attempt fails.
		 *
		 * Locking is currently an optional (opt-in) mechanism, i.e., buffers can also be
		 * accessed without being locked. This is because locking is a bit of a band-aid fix
		 * that doesn't properly cover all use-cases (for example, host-pointer initialized buffers).
		 *
		 * However, when accessing a locked buffer, the buffer_manager enforces additional
		 * rules to ensure they are used in a safe manner for the duration of the lock:
		 *	- A locked buffer may only be resized at most once, and only for the first access.
		 *	- A locked buffer may not be accessed using consumer access modes, if it was previously
		 *	  accessed using a pure producer mode.
		 *
		 * @returns Returns true if the list of buffers was successfully locked.
		 */
		bool try_lock(buffer_lock_id, const std::unordered_set<buffer_id>& buffers);

		/**
		 * Unlocks all buffers that were previously locked with a call to try_lock with the given @p id.
		 */
		void unlock(buffer_lock_id id);

		bool is_locked(buffer_id bid) const;

	  private:
		struct backing_buffer {
			std::unique_ptr<buffer_storage> storage = nullptr;
			cl::sycl::id<3> offset;

			backing_buffer(std::unique_ptr<buffer_storage> storage, cl::sycl::id<3> offset) : storage(std::move(storage)), offset(offset) {}
			backing_buffer() : backing_buffer(nullptr, cl::sycl::id<3>{0, 0, 0}) {}

			bool is_allocated() const { return storage != nullptr; }

			/**
			 * A backing buffer is often smaller than the "virtual" buffer that Celerity applications operate on.
			 * Given an offset in the virtual buffer, this function returns the local offset, relative to the backing buffer.
			 */
			cl::sycl::id<3> get_local_offset(const cl::sycl::id<3>& virtual_offset) const { return virtual_offset - offset; }
		};

		struct virtual_buffer {
			cl::sycl::range<3> range;
			backing_buffer device_buf;
			backing_buffer host_buf;
		};

		struct transfer {
			raw_buffer_data data;
			cl::sycl::id<3> target_offset;
		};

		struct resize_info {
			bool resize_required = false;
			cl::sycl::id<3> new_offset = {};
			cl::sycl::range<3> new_range = {};
		};

		enum class data_location { NOWHERE, HOST, DEVICE, HOST_AND_DEVICE };

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		struct buffer_type_guard_base {
			virtual ~buffer_type_guard_base(){};
			template <typename DataT, int Dims>
			bool has_type() const {
				return dynamic_cast<const buffer_type_guard<DataT, Dims>*>(this) != nullptr;
			}
		};

		template <typename DataT, int Dims>
		struct buffer_type_guard : buffer_type_guard_base {};
#endif

		struct buffer_lock_info {
			bool is_locked = false;

			// For lack of a better name, this stores *an* access mode that has already been used during this lock.
			// While it initially stores whatever is first used to access the buffer, it will always be overwritten
			// by subsequent pure producer accesses, as those are the only ones we really care about.
			std::optional<cl::sycl::access::mode> earlier_access_mode = std::nullopt;
		};

	  private:
		device_queue& queue;
		buffer_lifecycle_callback lifecycle_cb;
		size_t buffer_count = 0;
		mutable std::shared_mutex mutex;
		std::unordered_map<buffer_id, buffer_info> buffer_infos;
		std::unordered_map<buffer_id, virtual_buffer> buffers;
		std::unordered_map<buffer_id, std::vector<transfer>> scheduled_transfers;
		std::unordered_map<buffer_id, region_map<data_location>> newest_data_location;

		std::unordered_map<buffer_id, buffer_lock_info> buffer_lock_infos;
		std::unordered_map<buffer_lock_id, std::vector<buffer_id>> buffer_locks_by_id;

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		// Since we store buffers without type information (i.e., its data type and dimensionality),
		// it is the user's responsibility to only request access to a buffer using the correct type.
		// In debug builds we can help out a bit by remembering the type and asserting it on every access.
		std::unordered_map<buffer_id, std::unique_ptr<buffer_type_guard_base>> buffer_types;
#endif

		static resize_info is_resize_required(const backing_buffer& buffer, cl::sycl::range<3> request_range, cl::sycl::id<3> request_offset) {
			assert(buffer.is_allocated());
			const cl::sycl::range<3> old_abs_range = range_cast<3>(buffer.offset + buffer.storage->get_range());
			const cl::sycl::range<3> new_abs_range = range_cast<3>(request_offset + request_range);
			const bool is_inside_old_range = ((request_offset >= buffer.offset) == cl::sycl::id<3>(true, true, true))
			                                 && ((new_abs_range <= old_abs_range) == cl::sycl::range<3>(true, true, true));
			resize_info result;
			if(!is_inside_old_range) {
				result.resize_required = true;
				result.new_offset = min_id(request_offset, buffer.offset);
				result.new_range = range_cast<3>(id_cast<3>(max_range(old_abs_range, new_abs_range)) - result.new_offset);
			}
			return result;
		}

		/**
		 * Makes the contents of buffer @p target_buffer coherent within the range @p coherent_sr.
		 *
		 * This is done in three separate steps:
		 *	1) If @p mode is a consumer mode, apply all transfers that fully or partially overlap with the requested @p coherent_sr.
		 *	2) If @p mode is a consumer mode, copy newest data from H->D or D->H (depending on what type of buffer @p target_buffer is).
		 *	3) Optional: If @p previous_buffer is provided, ensure that any data that needs to be retained is copied into the new buffer.
		 *	   Importantly, this step is performed even for parts of @p previous_buffer that lie outside the requested @p coherent_sr.
		 *
		 * @param bid
		 * @param mode The access mode for which coherency needs to be established.
		 * @param target_buffer The buffer to make coherent, corresponding to @p bid.
		 * @param coherent_sr The subrange of @p target_buffer which is to be made coherent.
		 * @param previous_buffer Providing this optional argument indicates that the @p target_buffer is a newer (resized) version of this buffer,
		 *						  and that the previous contents may need to be retained.
		 *
		 * @note Calling this function has side-effects:
		 *	- Queued transfers are processed (if applicable).
		 *  - The newest data locations are updated to reflect replicated data as well as newly written ranges (depending on access mode).
		 */
		void make_buffer_subrange_coherent(buffer_id bid, cl::sycl::access::mode mode, backing_buffer& target_buffer, const subrange<3>& coherent_sr,
		    const backing_buffer& previous_buffer = backing_buffer{});

		/**
		 * Checks whether access to a currently locked buffer is safe.
		 *
		 * There's two distinct issues that can cause an access to be unsafe:
		 *	- If a buffer that has been accessed earlier needs to be resized (reallocated) now
		 *	- If a buffer was previously accessed using a discard_* mode and is now accessed using a consumer mode
		 */
		void audit_buffer_access(buffer_id bid, bool requires_allocation, cl::sycl::access::mode mode);
	};

} // namespace detail
} // namespace celerity
