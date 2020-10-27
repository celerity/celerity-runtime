#include "buffer_manager.h"

#include "buffer_storage.h"
#include "logger.h"
#include "runtime.h"

namespace celerity {
namespace detail {

	buffer_manager::buffer_manager(device_queue& queue, buffer_lifecycle_callback lifecycle_cb) : queue(queue), lifecycle_cb(std::move(lifecycle_cb)) {}

	void buffer_manager::unregister_buffer(buffer_id bid) noexcept {
		{
			std::unique_lock lock(mutex);
			assert(buffer_infos.find(bid) != buffer_infos.end());

			// Log the allocation size for host and device
			const auto& buf = buffers[bid];
			const size_t host_size = buf.host_buf.is_allocated() ? buf.host_buf.storage->get_size() : 0;
			const size_t device_size = buf.device_buf.is_allocated() ? buf.device_buf.storage->get_size() : 0;
			// During testing there might be no runtime available
			// FIXME: Pass logger instance to buffer_manager directly (at least if we want to do more logging in the future)
			if(runtime::is_initialized()) {
				const auto& logger = runtime::get_instance().get_logger();
				logger->trace(logger_map{{"event", "buffer unregister"}, {"bid", std::to_string(bid)}, {"hostSize", std::to_string(host_size)},
				    {"deviceSize", std::to_string(device_size)}});
			}
			buffers.erase(bid);
			buffer_infos.erase(bid);

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			buffer_types.erase(bid);
#endif
		}
		lifecycle_cb(buffer_lifecycle_event::UNREGISTERED, bid);
	}

	raw_buffer_data buffer_manager::get_buffer_data(buffer_id bid, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) {
		std::unique_lock lock(mutex);
		assert(buffers.count(bid) == 1 && (buffers.at(bid).device_buf.is_allocated() || buffers.at(bid).host_buf.is_allocated()));
		auto data_locations = newest_data_location.at(bid).get_region_values(subrange_to_grid_box(subrange<3>(offset, range)));

		// Slow path: We need to obtain current data from both host and device.
		if(data_locations.size() > 1) {
			auto& vb = buffers[bid];
			assert(vb.host_buf.is_allocated());

			// Make sure newest data resides on the host.
			// But first, we need to check whether the current host buffer is able to hold the full data range.
			const auto info = is_resize_required(vb.host_buf, range, offset);
			if(info.resize_required) {
				// TODO: Do we really want to allocate host memory for this..? We could also make raw_buffer_data "coherent" directly.
				backing_buffer new_buffer{std::unique_ptr<buffer_storage>(vb.host_buf.storage->make_new_of_same_type(info.new_range)), info.new_offset};
				make_buffer_subrange_coherent(bid, cl::sycl::access::mode::read, new_buffer, {offset, range}, vb.host_buf);
				vb.host_buf = std::move(new_buffer);
			} else {
				make_buffer_subrange_coherent(bid, cl::sycl::access::mode::read, vb.host_buf, {offset, range}, backing_buffer{});
			}

			data_locations = {{subrange_to_grid_box(subrange<3>(offset, range)), data_location::HOST}};
		}

		if(data_locations[0].second == data_location::HOST || data_locations[0].second == data_location::HOST_AND_DEVICE) {
			return buffers.at(bid).host_buf.storage->get_data(buffers.at(bid).host_buf.get_local_offset(offset), range);
		}

		return buffers.at(bid).device_buf.storage->get_data(buffers.at(bid).device_buf.get_local_offset(offset), range);
	}

	void buffer_manager::set_buffer_data(buffer_id bid, cl::sycl::id<3> offset, raw_buffer_data&& data) {
		std::unique_lock lock(mutex);
		assert(buffer_infos.count(bid) == 1);
		scheduled_transfers[bid].push_back({std::move(data), offset});
	}

	bool buffer_manager::try_lock(buffer_lock_id id, const std::unordered_set<buffer_id>& buffers) {
		assert(buffer_locks_by_id.count(id) == 0);
		for(auto bid : buffers) {
			if(buffer_lock_infos[bid].is_locked) return false;
		}
		buffer_locks_by_id[id].reserve(buffers.size());
		for(auto bid : buffers) {
			buffer_lock_infos[bid] = {true, std::nullopt};
			buffer_locks_by_id[id].push_back(bid);
		}
		return true;
	}

	void buffer_manager::unlock(buffer_lock_id id) {
		assert(buffer_locks_by_id.count(id) != 0);
		for(auto bid : buffer_locks_by_id[id]) {
			buffer_lock_infos[bid] = {};
		}
		buffer_locks_by_id.erase(id);
	}

	bool buffer_manager::is_locked(buffer_id bid) const {
		if(buffer_lock_infos.count(bid) == 0) return false;
		return buffer_lock_infos.at(bid).is_locked;
	}

	// TODO: Something we could look into is to dispatch all memory copies concurrently and wait for them in the end.
	void buffer_manager::make_buffer_subrange_coherent(
	    buffer_id bid, cl::sycl::access::mode mode, backing_buffer& target_buffer, const subrange<3>& coherent_sr, const backing_buffer& previous_buffer) {
		assert(!previous_buffer.is_allocated() || target_buffer.storage->get_type() == previous_buffer.storage->get_type());
		const auto target_buffer_location = target_buffer.storage->get_type() == buffer_type::HOST_BUFFER ? data_location::HOST : data_location::DEVICE;

		const auto coherent_box = subrange_to_grid_box(coherent_sr);

		// If a previous buffer is provided, we may have to retain some or all of the existing data.
		const GridRegion<3> retain_region = ([&]() {
			GridRegion<3> result = coherent_box;
			if(previous_buffer.is_allocated()) {
				result = GridRegion<3>::merge(result, subrange_to_grid_box({previous_buffer.offset, previous_buffer.storage->get_range()}));
			}
			return result;
		})(); // IIFE

		// Sanity check: Retain region must be at least as large as coherence box (and fully overlap).
		assert(coherent_box.area() <= retain_region.area());
		assert(GridRegion<3>::difference(coherent_box, retain_region).empty());
		// Also check that the new target buffer could actually fit the entire retain region.
		assert((grid_box_to_subrange(retain_region.boundingBox()).offset >= target_buffer.offset) == cl::sycl::id<3>(true, true, true));
		assert((grid_box_to_subrange(retain_region.boundingBox()).offset + grid_box_to_subrange(retain_region.boundingBox()).range
		           <= target_buffer.offset + target_buffer.storage->get_range())
		       == cl::sycl::id<3>(true, true, true));

		// Check whether we have any scheduled transfers that overlap with the requested subrange, and if so, apply them.
		// For this, we are not interested in the retain region (but we need to remember what parts will NOT have to be retained afterwards).
		auto remaining_region_after_transfers = retain_region;
#if !defined(CELERITY_DETAIL_ENABLE_DEBUG)
		// There should only be queued transfers for this buffer iff this is a consumer mode.
		// To assert this we check for bogus transfers for other modes in debug builds.
		if(detail::access::mode_traits::is_consumer(mode))
#endif
		{
			GridRegion<3> updated_region;
			std::vector<transfer> remaining_transfers;
			auto& scheduled_buffer_transfers = scheduled_transfers[bid];
			remaining_transfers.reserve(scheduled_buffer_transfers.size() / 2);
			for(auto& t : scheduled_buffer_transfers) {
				auto t_region = subrange_to_grid_box({t.target_offset, t.data.get_range()});

				// Check whether this transfer applies to the current request.
				auto t_minus_coherent_region = GridRegion<3>::difference(t_region, coherent_box);
				if(!t_minus_coherent_region.empty()) {
					// Check if transfer applies partially.
					// This might happen in certain situations, when two different commands partially overlap in their required buffer ranges.
					// We currently handle this by only copying the part of the transfer that intersects with the requested buffer range
					// into the buffer, leaving the transfer around for future requests.
					//
					// NOTE: We currently assume that one of the requests will consume the FULL transfer. Only then we discard it.
					// This assumption is valid right now, as the graph generator will not consolidate adjacent PUSHes for two (or more)
					// separate commands. This might however change in the future.
					if(t_minus_coherent_region != t_region) {
						assert(detail::access::mode_traits::is_consumer(mode));
						auto intersection = GridRegion<3>::intersect(t_region, coherent_box);
						remaining_region_after_transfers = GridRegion<3>::difference(remaining_region_after_transfers, intersection);
						intersection.scanByBoxes([&](const GridBox<3>& box) {
							auto sr = grid_box_to_subrange(box);
							auto partial_t = t.data.copy(sr.offset - t.target_offset, sr.range);
							target_buffer.storage->set_data(target_buffer.get_local_offset(sr.offset), std::move(partial_t));
							updated_region = GridRegion<3>::merge(updated_region, box);
						});
					}
					// Transfer only applies partially, or not at all - which means we have to keep it around.
					remaining_transfers.emplace_back(std::move(t));
					continue;
				}

				// Transfer applies fully.
				assert(detail::access::mode_traits::is_consumer(mode));
				remaining_region_after_transfers = GridRegion<3>::difference(remaining_region_after_transfers, t_region);
				target_buffer.storage->set_data(target_buffer.get_local_offset(t.target_offset), std::move(t.data));
				updated_region = GridRegion<3>::merge(updated_region, t_region);
			}
			// The target buffer now has the newest data in this region.
			newest_data_location.at(bid).update_region(updated_region, target_buffer_location);
			scheduled_buffer_transfers = std::move(remaining_transfers);
		}

		if(!remaining_region_after_transfers.empty()) {
			const auto maybe_retain_box = [&](const GridBox<3>& box) {
				if(detail::access::mode_traits::is_consumer(mode)) {
					// If we are accessing the buffer using a consumer mode, we have to retain the full previous contents, otherwise...
					const auto box_sr = grid_box_to_subrange(box);
					target_buffer.storage->copy(
					    *previous_buffer.storage, previous_buffer.get_local_offset(box_sr.offset), target_buffer.get_local_offset(box_sr.offset), box_sr.range);
				} else {
					// ...check if there are parts of the previous buffer that we are not going to overwrite (and thus have to retain).
					// If so, copy only those parts.
					const auto remaining_region = GridRegion<3>::difference(box, coherent_box);
					remaining_region.scanByBoxes([&](const GridBox<3>& small_box) {
						const auto small_box_sr = grid_box_to_subrange(small_box);
						target_buffer.storage->copy(*previous_buffer.storage, previous_buffer.get_local_offset(small_box_sr.offset),
						    target_buffer.get_local_offset(small_box_sr.offset), small_box_sr.range);
					});
				}
			};

			GridRegion<3> replicated_region;
			auto& buffer_data_locations = newest_data_location.at(bid);
			const auto data_locations = buffer_data_locations.get_region_values(remaining_region_after_transfers);
			for(auto& dl : data_locations) {
				// Note that this assertion can fail in legitimate cases, e.g.
				// when users manually handle uninitialized reads in the first iteration of some loop.
				// assert(!previous_buffer.is_allocated() || dl.second != data_location::NOWHERE);

				if(target_buffer.storage->get_type() == buffer_type::DEVICE_BUFFER) {
					// Copy from device in case we are resizing an existing buffer
					if((dl.second == data_location::DEVICE || dl.second == data_location::HOST_AND_DEVICE) && previous_buffer.is_allocated()) {
						maybe_retain_box(dl.first);
					}
					// Copy from host, unless we are using a pure producer mode
					else if(dl.second == data_location::HOST && detail::access::mode_traits::is_consumer(mode)) {
						assert(buffers[bid].host_buf.is_allocated());
						const auto box_sr = grid_box_to_subrange(dl.first);
						const auto& host_buf = buffers[bid].host_buf;
						target_buffer.storage->copy(
						    *host_buf.storage, host_buf.get_local_offset(box_sr.offset), target_buffer.get_local_offset(box_sr.offset), box_sr.range);
						replicated_region = GridRegion<3>::merge(replicated_region, dl.first);
					}
				} else if(target_buffer.storage->get_type() == buffer_type::HOST_BUFFER) {
					// Copy from device, unless we are using a pure producer mode
					if(dl.second == data_location::DEVICE && detail::access::mode_traits::is_consumer(mode)) {
						assert(buffers[bid].device_buf.is_allocated());
						const auto box_sr = grid_box_to_subrange(dl.first);
						const auto& device_buf = buffers[bid].device_buf;
						target_buffer.storage->copy(
						    *device_buf.storage, device_buf.get_local_offset(box_sr.offset), target_buffer.get_local_offset(box_sr.offset), box_sr.range);
						replicated_region = GridRegion<3>::merge(replicated_region, dl.first);
					}
					// Copy from host in case we are resizing an existing buffer
					else if((dl.second == data_location::HOST || dl.second == data_location::HOST_AND_DEVICE) && previous_buffer.is_allocated()) {
						maybe_retain_box(dl.first);
					}
				}
			}

			// Finally, remember the fact that we replicated some regions to the new target location.
			buffer_data_locations.update_region(replicated_region, data_location::HOST_AND_DEVICE);
		}

		if(detail::access::mode_traits::is_producer(mode)) { newest_data_location.at(bid).update_region(coherent_box, target_buffer_location); }
	}

	void buffer_manager::audit_buffer_access(buffer_id bid, bool requires_allocation, cl::sycl::access::mode mode) {
		auto& lock_info = buffer_lock_infos[bid];

		// Buffer locking is currently opt-in, so if this buffer isn't locked, we won't check anything else.
		if(!lock_info.is_locked) return;

		if(lock_info.earlier_access_mode == std::nullopt) {
			// First access, all good.
			lock_info.earlier_access_mode = mode;
			return;
		}

		if(requires_allocation) {
			// Re-allocation of a buffer that is currently being accessed never works.
			throw std::runtime_error("You are requesting multiple accessors for the same buffer, with later ones requiring a larger part of the buffer, "
			                         "causing a backing buffer reallocation. "
			                         "This is currently unsupported. Try changing the order of your calls to buffer::get_access.");
		}

		if(!access::mode_traits::is_consumer(*lock_info.earlier_access_mode) && access::mode_traits::is_consumer(mode)) {
			// Accessing a buffer using a pure producer mode followed by a consumer mode breaks our coherence bookkeeping.
			throw std::runtime_error("You are requesting multiple accessors for the same buffer, using a discarding access mode first, followed by a "
			                         "non-discarding mode. This is currently unsupported. Try changing the order of your calls to buffer::get_access.");
		}

		// We only need to remember pure producer accesses.
		if(!access::mode_traits::is_consumer(mode)) { lock_info.earlier_access_mode = mode; }
	}

} // namespace detail
} // namespace celerity
