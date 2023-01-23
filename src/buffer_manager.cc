#include "buffer_manager.h"

#include "buffer_storage.h"
#include "log.h"
#include "runtime.h"

namespace celerity {
namespace detail {

	buffer_manager::buffer_manager(local_devices& devices, buffer_lifecycle_callback lifecycle_cb)
	    : m_local_devices(devices), m_lifecycle_cb(std::move(lifecycle_cb)) {}

	void buffer_manager::unregister_buffer(buffer_id bid) noexcept {
		{
			std::unique_lock lock(m_mutex);
			assert(m_buffer_infos.find(bid) != m_buffer_infos.end());

			// Log the allocation size for host and device
			const auto& buf = m_buffers.at(bid);
			const size_t host_size =
			    buf.get(m_local_devices.get_host_memory_id()).is_allocated() ? buf.get(m_local_devices.get_host_memory_id()).storage->get_size() : 0;
			// const size_t device_size = buf.device_buf.is_allocated() ? buf.device_buf.storage->get_size() : 0;
			const auto device_size = "NYI NOCOMMIT";

			CELERITY_TRACE("Unregistering buffer {}. host size = {} B, device size = {} B", bid, host_size, device_size);
			m_buffers.erase(bid);
			m_buffer_infos.erase(bid);

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			m_buffer_types.erase(bid);
#endif
		}
		m_lifecycle_cb(buffer_lifecycle_event::unregistered, bid);
	}

	backend::async_event buffer_manager::get_buffer_data(buffer_id bid, const subrange<3>& sr, void* out_linearized) {
		std::unique_lock lock(m_mutex);
		// assert(m_buffers.count(bid) == 1 && (m_buffers.at(bid).device_buf.is_allocated() || m_buffers.at(bid).host_buf.is_allocated())); // NOCOMMIT
		auto data_locations = m_newest_data_location.at(bid).get_region_values(subrange_to_grid_box(sr));

		// Slow path: We (may) need to obtain current data from multiple memories.
		// FIXME: We probably run into this more frequently with multi-GPU support. In particular after a horizon.
		if(data_locations.size() > 1) {
			auto& existing_buf = m_buffers.at(bid).get(m_local_devices.get_host_memory_id());

			// Make sure newest data resides on the host.
			// But first, we need to check whether the current host buffer is able to hold the full data range.
			const auto info = is_resize_required(existing_buf, sr.range, sr.offset);
			backing_buffer replacement_buf;
			if(info.resize_required) {
				// TODO: Do we really want to allocate host memory for this..? We could also make the buffer storage "coherent" directly.
				replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_host(info.new_range), info.new_offset};
			}
			auto [coherent_buf, pending_transfers] = make_buffer_subrange_coherent(
			    m_local_devices.get_host_memory_id(), bid, access_mode::read, std::move(existing_buf), sr, std::move(replacement_buf));
			existing_buf = std::move(coherent_buf);
			while(!pending_transfers.is_done()) {} // NOCOMMIT Add wait()?

			data_locations = {{subrange_to_grid_box(sr), data_location{}.set(m_local_devices.get_host_memory_id())}};
		}

		// get_buffer_data will race with pending transfers for the same subrange. In case there are pending transfers and a host buffer does not exist yet,
		// these transfers cannot easily be flushed here as creating a host buffer requires a templated context that knows about DataT.
		assert(std::none_of(m_scheduled_transfers[bid].begin(), m_scheduled_transfers[bid].end(),
		    [&](const transfer& t) { return !GridRegion<3>::intersect(subrange_to_grid_box(sr), t.unconsumed).empty(); }));

		if(data_locations[0].second.test(m_local_devices.get_host_memory_id())) {
			return m_buffers.at(bid)
			    .get(m_local_devices.get_host_memory_id())
			    .storage->get_data({m_buffers.at(bid).get(m_local_devices.get_host_memory_id()).get_local_offset(sr.offset), sr.range}, out_linearized);
		}

		const memory_id source_mid = ([this, &data_locations] {
			for(memory_id mid = 0; mid < m_local_devices.num_memories(); ++mid) {
				// FIXME: We currently choose the first available memory as a source, should use a better strategy here (maybe random or based on current load)
				if(data_locations[0].second.test(mid)) return mid;
			}
			assert(false && "Data region requested that is not available locally");
			return memory_id(-1);
		})(); // IIFE

		return m_buffers.at(bid).get(source_mid).storage->get_data({m_buffers.at(bid).get(source_mid).get_local_offset(sr.offset), sr.range}, out_linearized);
	}

	buffer_manager::staging_buffer& buffer_manager::get_free_staging_buffer(const size_t size) {
		for(auto& sb : m_staging_buffers) {
			// TODO: At least do best fit...
			if(sb->is_free && sb->buffer.get_size() >= size) {
				sb->is_free = false;
				// CELERITY_WARN("Returning free staging buffer {}", (void*)sb->buffer.get_pointer());
				return *sb;
			}
		}
		CELERITY_WARN("Allocating new staging buffer");
		// FIXME: Not memory aware
		auto& device = m_local_devices.get_device_queue(m_next_staging_allocation_device);
		// Do at least 10 MiB
		auto buf = std::make_unique<staging_buffer>(std::max(size, size_t(10) * 1024 * 1024), device);
		m_staging_buffers.emplace_back(std::move(buf));
		return *m_staging_buffers.back();
	}

	void buffer_manager::set_buffer_data(buffer_id bid, const subrange<3>& sr, unique_payload_ptr in_linearized) {
		std::unique_lock lock(m_mutex);
		assert(m_buffer_infos.count(bid) == 1);

		// Find free staging buffer
		auto& staging_buf = get_free_staging_buffer(sr.range.size() * m_buffer_infos[bid].element_size);
		auto evt = backend::memcpy_strided_device(staging_buf.buffer.get_owning_queue(), in_linearized.get_pointer(), staging_buf.buffer.get_pointer(),
		    m_buffer_infos[bid].element_size, range<1>(sr.range.size()), id<1>{}, range<1>(sr.range.size()), id<1>{}, range<1>(sr.range.size()));
		evt.wait(); // FIXME This should be async as well...
		m_scheduled_transfers[bid].push_back(buffer_manager::transfer{staging_buf, sr});
	}

	buffer_manager::access_info buffer_manager::access_device_buffer(
	    const memory_id mid, buffer_id bid, cl::sycl::access::mode mode, const cl::sycl::range<3>& range, const cl::sycl::id<3>& offset) {
		std::unique_lock lock(m_mutex);
		assert((range_cast<3>(offset + range) <= m_buffer_infos.at(bid).range) == cl::sycl::range<3>(true, true, true));

		auto& device_queue = m_local_devices.get_close_device_queue(mid);

		auto& existing_buf = m_buffers.at(bid).get(mid);
		assert(!existing_buf.is_allocated() || existing_buf.storage->get_type() == buffer_type::device_buffer);
		backing_buffer replacement_buf;

		[[maybe_unused]] const auto die = [&](const size_t allocation_size) {
			std::string msg = fmt::format("Unable to allocate buffer {} of size {} on device {} (memory {}).\n", bid, allocation_size, device_queue.get_id(),
			    device_queue.get_memory_id());
			fmt::format_to(std::back_inserter(msg), "\nCurrent allocations on device {}:\n", device_queue.get_id());
			size_t total_bytes = 0;
			for(const auto& [bid, b] : m_buffers) {
				const auto& bb = b.get(mid);
				if(bb.is_allocated()) {
					fmt::format_to(std::back_inserter(msg), "\tBuffer {}: {} bytes\n", bid, bb.storage->get_size());
					total_bytes += bb.storage->get_size();
				}
			}
			fmt::format_to(std::back_inserter(msg), "Total usage: {} / {} bytes ({:.1f}%).\n", total_bytes, device_queue.get_global_memory_size(),
			    100 * static_cast<double>(total_bytes) / device_queue.get_global_memory_size());
			throw allocation_error(msg);
		};

		if(!existing_buf.is_allocated()) {
#if USE_NDVBUFFER
			// Construct buffer with full virtual buffer range (this only allocates the address space, no memory)
			// NOCOMMIT FIXME: Unfortunately this breaks our memory management (for OOM error message).
			// Possible solution would be to add an allocator interface to ndv that is then implemented by device_queue.
			replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_device(m_buffer_infos.at(bid).range, device_queue), {}};
			replacement_buf.storage->allocate(subrange{offset, range});
#else
			const auto allocation_size = range.size() * m_buffer_infos.at(bid).element_size;
			if(!can_allocate(device_queue.get_memory_id(), allocation_size)) {
				// TODO: Unless this single allocation exceeds the total available memory on the device we don't need to abort right away,
				// could evict other buffers first.
				die(allocation_size);
			}
			replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_device(range, device_queue), offset};
#endif
		} else {
#if USE_NDVBUFFER
			existing_buf.storage->allocate(subrange{offset, range});
#else
			if(const auto info = is_resize_required(existing_buf, range, offset); info.resize_required) {
				if(NOMERGE_warn_on_device_buffer_resize) {
					CELERITY_WARN("Resizing buffer {} on memory {} due to access {}, from {} to {}.\n.", bid, mid, subrange<3>(offset, range),
					    subrange<3>(existing_buf.offset, existing_buf.storage->get_range()), subrange<3>(info.new_offset, info.new_range));
				}

				assert(!existing_buf.storage->supports_dynamic_allocation());
				const auto element_size = m_buffer_infos.at(bid).element_size;
				const auto allocation_size = info.new_range.size() * element_size;
				if(can_allocate(device_queue.get_memory_id(), allocation_size)) {
					// Easy path: We can just do the resize on the device directly
					replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_device(info.new_range, device_queue), info.new_offset};
				} else {
					ZoneScopedN("slow path: reallocate through host");

					bool spill_to_host = false;
					// Check if we can do the resize by going through host first (see if we'll be able to fit just the added elements of the resized buffer).
					if(!can_allocate(device_queue.get_memory_id(), allocation_size - (existing_buf.storage->get_range().size() * element_size))) {
						// Final attempt: Check if we can create a new buffer with the requested size if we spill everything else to the host.
						if(can_allocate(device_queue.get_memory_id(), range.size() * element_size, existing_buf.storage->get_range().size() * element_size)) {
							CELERITY_WARN("Spilling buffer {} on memory {}, existing size {}, to enable acecss {}\n", bid, mid,
							    subrange<3>(existing_buf.offset, existing_buf.storage->get_range()), subrange<3>(offset, range));
							spill_to_host = true;
						} else {
							// TODO: Same thing as above
							die(allocation_size);
						}
					}

					// Use faux host accesses retain all data from the device (except what is going to be discarded anyway).
					// TODO: This could be made more efficient, currently it may cause multiple consecutive resizes.
					GridRegion<3> retain_region = subrange_to_grid_box(subrange<3>{existing_buf.offset, existing_buf.storage->get_range()});
					if(!access::mode_traits::is_consumer(mode)) {
						retain_region = GridRegion<3>::difference(retain_region, subrange_to_grid_box(subrange<3>{offset, range}));
					}
					retain_region.scanByBoxes([&](const GridBox<3>& box) {
						const auto sr = grid_box_to_subrange(box);
						access_host_buffer_impl(bid, access_mode::read, sr.range, sr.offset);
					});

					// We now have all data "backed up" on the host, so we may deallocate the device buffer (via destructor).
					const auto existing_buf_sr = subrange<3>{existing_buf.offset, existing_buf.storage->get_range()};
					existing_buf = backing_buffer{};
					auto locations = m_newest_data_location.at(bid).get_region_values(subrange_to_grid_box(existing_buf_sr));
					for(auto& [box, locs] : locations) {
						m_newest_data_location.at(bid).update_region(box, locs.reset(mid));
					}

					// Finally create the new device buffer. It will be made coherent with data from the host below.
					// If we have to spill to host, only allocate the currently requested subrange. Otherwise use bounding box of existing and new range.
					replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_device(spill_to_host ? range : info.new_range, device_queue),
					    spill_to_host ? offset : info.new_offset};
				}
			}
#endif
		}

		audit_buffer_access(bid, mid, replacement_buf.is_allocated(), mode);

		if(m_test_mode && replacement_buf.is_allocated()) {
			auto* ptr = replacement_buf.storage->get_pointer();
			const auto bytes = replacement_buf.storage->get_size();
			device_queue.get_sycl_queue().submit([&](cl::sycl::handler& cgh) { cgh.memset(ptr, test_mode_pattern, bytes); }).wait();
		}

		auto [coherent_buf, pending_transfers] =
		    make_buffer_subrange_coherent(mid, bid, mode, std::move(existing_buf), {offset, range}, std::move(replacement_buf));
		existing_buf = std::move(coherent_buf);

		return {existing_buf.storage->get_pointer(), existing_buf.storage->get_range(), existing_buf.offset, std::move(pending_transfers)};
	}

	buffer_manager::access_info buffer_manager::access_host_buffer(
	    buffer_id bid, cl::sycl::access::mode mode, const cl::sycl::range<3>& range, const cl::sycl::id<3>& offset) {
		std::unique_lock lock(m_mutex);
		return access_host_buffer_impl(bid, mode, range, offset);
	}

	buffer_manager::access_info buffer_manager::access_host_buffer_impl(
	    buffer_id bid, cl::sycl::access::mode mode, const cl::sycl::range<3>& range, const cl::sycl::id<3>& offset) {
		assert((range_cast<3>(offset + range) <= m_buffer_infos.at(bid).range) == cl::sycl::range<3>(true, true, true));

		auto& existing_buf = m_buffers.at(bid).get(m_local_devices.get_host_memory_id());
		assert(!existing_buf.is_allocated() || existing_buf.storage->get_type() == buffer_type::host_buffer);
		backing_buffer replacement_buf;

		if(!existing_buf.is_allocated()) {
			replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_host(range), offset};
		} else {
			const auto info = is_resize_required(existing_buf, range, offset);
			if(info.resize_required) { replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_host(info.new_range), info.new_offset}; }
		}

		audit_buffer_access(bid, m_local_devices.get_host_memory_id(), replacement_buf.is_allocated(), mode);

		if(m_test_mode && replacement_buf.is_allocated()) {
			auto* ptr = replacement_buf.storage->get_pointer();
			const auto size = replacement_buf.storage->get_size();
			std::memset(ptr, test_mode_pattern, size);
		}

		auto [coherent_buf, pending_transfers] = make_buffer_subrange_coherent(
		    m_local_devices.get_host_memory_id(), bid, mode, std::move(existing_buf), {offset, range}, std::move(replacement_buf));
		existing_buf = std::move(coherent_buf);

		return {existing_buf.storage->get_pointer(), existing_buf.storage->get_range(), existing_buf.offset, std::move(pending_transfers)};
	}

	bool buffer_manager::try_lock(const buffer_lock_id id, const memory_id mid, const std::unordered_set<buffer_id>& buffers) {
		// NOCOMMIT EXTREME HACK
		if(NOMERGE_warn_on_device_buffer_resize) { return true; }

		assert(m_buffer_locks_by_id.count(id) == 0);
		for(auto bid : buffers) {
			if(m_buffer_lock_infos[std::pair{bid, mid}].is_locked) return false;
		}
		m_buffer_locks_by_id[id].reserve(buffers.size());
		for(auto bid : buffers) {
			m_buffer_lock_infos[std::pair{bid, mid}] = {true, std::nullopt};
			m_buffer_locks_by_id[id].push_back(std::pair{bid, mid});
		}
		return true;
	}

	void buffer_manager::unlock(buffer_lock_id id) {
		// NOCOMMIT EXTREME HACK
		if(NOMERGE_warn_on_device_buffer_resize) {
			// We probably didn't lock in the first place, avoid running into assertion below
			return;
		}

		assert(m_buffer_locks_by_id.count(id) != 0);
		for(auto bid : m_buffer_locks_by_id[id]) {
			m_buffer_lock_infos[bid] = {};
		}
		m_buffer_locks_by_id.erase(id);
	}

	bool buffer_manager::is_locked(const buffer_id bid, const memory_id mid) const {
		if(m_buffer_lock_infos.count(std::pair{bid, mid}) == 0) return false;
		return m_buffer_lock_infos.at(std::pair{bid, mid}).is_locked;
	}

	// TODO: Something we could look into is to dispatch all memory copies concurrently and wait for them in the end.
	std::pair<buffer_manager::backing_buffer, backend::async_event> buffer_manager::make_buffer_subrange_coherent(const memory_id mid, buffer_id bid,
	    cl::sycl::access::mode mode, backing_buffer existing_buffer, const subrange<3>& coherent_sr, backing_buffer replacement_buffer) {
		ZoneScopedN("make_buffer_subrange_coherent");
		backing_buffer target_buffer, previous_buffer;
		if(replacement_buffer.is_allocated()) {
			assert(!existing_buffer.is_allocated() || replacement_buffer.storage->get_type() == existing_buffer.storage->get_type());
			target_buffer = std::move(replacement_buffer);
			previous_buffer = std::move(existing_buffer);
		} else {
			assert(existing_buffer.is_allocated());
			target_buffer = std::move(existing_buffer);
			previous_buffer = {};
		}

		if(coherent_sr.range.size() == 0) { return std::pair{std::move(target_buffer), backend::async_event{}}; }

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

		backend::async_event pending_transfers;

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
			auto& scheduled_buffer_transfers = m_scheduled_transfers[bid];
			remaining_transfers.reserve(scheduled_buffer_transfers.size() / 2);
			for(auto& t : scheduled_buffer_transfers) {
				// Check whether this transfer applies to the current request.
				const auto unconsumed_minus_coherent_region = GridRegion<3>::difference(t.unconsumed, coherent_box);

				if(unconsumed_minus_coherent_region == t.unconsumed) {
					// Transfer does not apply - continue.
					remaining_transfers.emplace_back(std::move(t));
					continue;
				}

				// We can't be receiving something that we are going to overwrite immediately.
				assert(detail::access::mode_traits::is_consumer(mode));

				const bool is_partially_consumed = t.unconsumed != subrange_to_grid_box(t.sr);

				//
				//
				// There was a bug. It caused me great pain.
				//
				//
				// TOOD: Partial ingestion w/ multiple devices can lead to the same portion being ingested multiple times,
				//       overriding newer data in the process. I'm still not entirely sure how it happens => Needs further investigation.
				//       Surfaced in Cahn-Hilliard, 2D split 4x oversubscribed with 2 nodes and at least 2 GPUs each.
				//
				// Current workaround is to keep track of which parts of a transfer have already been ingested. This is not ideal
				// b/c it means that we are likely to have a lot more partial transfers, which are SLOW.
				//
				if(!unconsumed_minus_coherent_region.empty() || is_partially_consumed) {
					assert(detail::access::mode_traits::is_consumer(mode));
					const auto intersection = GridRegion<3>::intersect(t.unconsumed, coherent_box);
					remaining_region_after_transfers = GridRegion<3>::difference(remaining_region_after_transfers, intersection);
					// const auto element_size = m_buffer_infos.at(bid).element_size;
					intersection.scanByBoxes([&](const GridBox<3>& box) {
						ZoneScopedN("ingest transfer (partial)");
						const auto sr = grid_box_to_subrange(box);
						// TODO can this temp buffer be avoided?
						// auto tmp = make_uninitialized_payload<std::byte>(sr.range.size() * element_size);
						// FIXME: THIS MUST BE AVOIDED AT ALL COSTS! On Marconi-100 I've seen this take ~100ms for a 16 MiB buffer ?!
						// linearize_subrange(t.linearized.get_pointer(), tmp.get_pointer(), element_size, t.sr.range, {sr.offset - t.sr.offset, sr.range});
						// auto evt = target_buffer.storage->set_data({target_buffer.get_local_offset(sr.offset), sr.range}, tmp.get_pointer());
						// target_buffer.storage->copy(, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range)
						auto evt = target_buffer.storage->copy_from_device_raw(t.get_buffer().get_owning_queue(), t.get_buffer().get_pointer(), t.sr.range,
						    sr.offset - t.sr.offset, target_buffer.get_local_offset(sr.offset), sr.range);
						// auto evt = target_buffer.storage->copy(t.get_buffer(), sr.offset - t.sr.offset, target_buffer.get_local_offset(sr.offset), sr.range);
						// evt.hack_attach_payload(std::move(tmp)); // FIXME
						pending_transfers.merge(std::move(evt));
						updated_region = GridRegion<3>::merge(updated_region, box);
					});

					t.unconsumed = GridRegion<3>::difference(t.unconsumed, intersection);

					if(!t.unconsumed.empty()) {
						// Transfer only applied partially - which means we have to keep it around.
						remaining_transfers.emplace_back(std::move(t));
					}
					continue;
				}

				ZoneScopedN("ingest transfer (full)");
				// Transfer applies fully.
				remaining_region_after_transfers = GridRegion<3>::difference(remaining_region_after_transfers, t.unconsumed);
				// auto evt = target_buffer.storage->set_data({target_buffer.get_local_offset(t.sr.offset), t.sr.range}, t.linearized.get_pointer());
				auto evt = target_buffer.storage->copy_from_device_raw(t.get_buffer().get_owning_queue(), t.get_buffer().get_pointer(), t.sr.range, id<3>{},
				    target_buffer.get_local_offset(t.sr.offset), t.sr.range);
				// auto evt = target_buffer.storage->copy(t.get_buffer(), id<3>{}, target_buffer.get_local_offset(t.sr.offset), t.sr.range);
				// evt.hack_attach_payload(std::move(t.linearized)); // FIXME
				pending_transfers.merge(std::move(evt));
				updated_region = GridRegion<3>::merge(updated_region, t.unconsumed);
			}
			// The target buffer now has the newest data in this region.
			{
				// FIXME: DRY below
				const auto locations = m_newest_data_location.at(bid).get_region_values(updated_region);
				for(auto& [box, _] : locations) {
					// NOCOMMIT TODO: Add regression test: This used to be data_location{locs}.set(mid), i.e., the previous location remained valid.
					m_newest_data_location.at(bid).update_region(box, data_location{}.set(mid));
				}
			}
			scheduled_buffer_transfers = std::move(remaining_transfers);
		}

		if(!remaining_region_after_transfers.empty()) {
			const auto maybe_retain_box = [&](const GridBox<3>& box) {
				if(detail::access::mode_traits::is_consumer(mode)) {
					// If we are accessing the buffer using a consumer mode, we have to retain the full previous contents, otherwise...
					const auto box_sr = grid_box_to_subrange(box);
					auto evt = target_buffer.storage->copy(
					    *previous_buffer.storage, previous_buffer.get_local_offset(box_sr.offset), target_buffer.get_local_offset(box_sr.offset), box_sr.range);
					pending_transfers.merge(std::move(evt));
				} else {
					// ...check if there are parts of the previous buffer that we are not going to overwrite (and thus have to retain).
					// If so, copy only those parts.
					const auto remaining_region = GridRegion<3>::difference(box, coherent_box);
					remaining_region.scanByBoxes([&](const GridBox<3>& small_box) {
						const auto small_box_sr = grid_box_to_subrange(small_box);
						auto evt = target_buffer.storage->copy(*previous_buffer.storage, previous_buffer.get_local_offset(small_box_sr.offset),
						    target_buffer.get_local_offset(small_box_sr.offset), small_box_sr.range);
						pending_transfers.merge(std::move(evt));
					});
				}
			};

			GridRegion<3> replicated_region;
			auto& buffer_data_locations = m_newest_data_location.at(bid);
			const auto data_locations = buffer_data_locations.get_region_values(remaining_region_after_transfers);
			for(auto& [box, locs] : data_locations) {
				// Note that this assertion can fail in legitimate cases, e.g.
				// when users manually handle uninitialized reads in the first iteration of some loop.
				// assert(!previous_buffer.is_allocated() || !locs.none());
				if(locs.none()) continue;

				// Copy from same memory in case we are resizing an existing buffer
				if(locs.test(mid)) {
					if(previous_buffer.is_allocated()) { maybe_retain_box(box); }
					continue;
				}

				// No need to copy data from a different memory if we are not going to read it.
				if(!detail::access::mode_traits::is_consumer(mode)) { continue; }

				const memory_id source_mid = ([this, locs = &locs] {
					for(memory_id m = m_local_devices.num_memories() - 1; m >= 0; --m) {
						// FIXME: We currently choose the first available memory as a source (starting from the back to use host memory as a last resort),
						// should use a better strategy here (maybe random or based on current load)
						if(locs->test(m)) return m;
					}
					assert(false && "Data region requested that is not available locally");
					return memory_id(-1);
				})(); // IIFE

				assert(source_mid != mid);
				assert(m_buffers.at(bid).get(source_mid).is_allocated());
				const auto box_sr = grid_box_to_subrange(box);
				const auto& source_buffer = m_buffers.at(bid).get(source_mid);
				auto evt = target_buffer.storage->copy(
				    *source_buffer.storage, source_buffer.get_local_offset(box_sr.offset), target_buffer.get_local_offset(box_sr.offset), box_sr.range);
				pending_transfers.merge(std::move(evt));
				replicated_region = GridRegion<3>::merge(replicated_region, box);
			}

			// Finally, remember the fact that we replicated some regions to the new target location.
			{
				// NOCOMMIT DRY above, could do this in a single update step. Also can we skip if producer mode?
				const auto locations = m_newest_data_location.at(bid).get_region_values(replicated_region);
				for(auto& [box, locs] : locations) {
					m_newest_data_location.at(bid).update_region(box, data_location{locs}.set(mid));
				}
			}
		}

		if(detail::access::mode_traits::is_producer(mode)) { m_newest_data_location.at(bid).update_region(coherent_box, data_location{}.set(mid)); }

		return std::pair{std::move(target_buffer), std::move(pending_transfers)};
	}

	void buffer_manager::audit_buffer_access(const buffer_id bid, const memory_id mid, const bool requires_allocation, const access_mode mode) {
		auto& lock_info = m_buffer_lock_infos[std::pair{bid, mid}];

		// Buffer locking is currently opt-in, so if this buffer isn't locked, we won't check anything else.
		if(!lock_info.is_locked) return;

		if(lock_info.earlier_access_mode == std::nullopt) {
			// First access, all good.
			lock_info.earlier_access_mode = mode;
			return;
		}

		// NOCOMMIT TODO: We can get rid of this with removal of multi-pass, and even more so with ndvbuffers.
		// if(requires_allocation) {
		// 	// Re-allocation of a buffer that is currently being accessed never works.
		// 	throw std::runtime_error("You are requesting multiple accessors for the same buffer, with later ones requiring a larger part of the buffer, "
		// 	                         "causing a backing buffer reallocation. "
		// 	                         "This is currently unsupported. Try changing the order of your calls to buffer::get_access.");
		// }

		// NOCOMMIT TODO: We can get rid of this with removal of multi-pass!
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
