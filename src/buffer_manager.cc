#include "buffer_manager.h"

#include "buffer_storage.h"
#include "log.h"
#include "runtime.h"

namespace celerity {
namespace detail {

	buffer_manager::buffer_manager(local_devices& devices, buffer_lifecycle_callback lifecycle_cb)
	    : m_local_devices(devices), m_lifecycle_cb(std::move(lifecycle_cb)) {
		// NOCOMMIT
		for(device_id did = 0; did < devices.num_compute_devices(); ++did) {
			cudaStream_t s;
			CELERITY_CUDA_CHECK(cudaSetDevice, did);
			CELERITY_CUDA_CHECK(cudaStreamCreate, &s);
			m_cuda_copy_streams.emplace(did, std::unique_ptr<CUstream_st, delete_cuda_stream>(s));
		}
	}

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
		ZoneScoped;

		std::unique_lock lock(m_mutex);
		const auto& buffer_info = m_buffer_infos.at(bid);

#if TRACY_ENABLE
		auto zone_sr_text = fmt::format("B{} {}", bid, sr);
#endif

		backend::async_event pending_fast_transfers;
		std::vector<std::tuple<subrange<3>, backend::async_event, unique_payload_ptr>> pending_slow_transfers;
		for(const auto& [box, source_mid] : m_authoritative_source.at(bid).get_region_values(subrange_to_grid_box(sr))) {
			const auto part_sr = grid_box_to_subrange(box);

			auto& source = m_buffers.at(bid).get(source_mid.value());
			// TODO if source_mid == host_memory_id, async transfers won't actually be async - we want to instead begin start all async d2h transfers first

			const bool part_has_output_stride =
			    part_sr.offset[1] == sr.offset[1] && part_sr.range[1] == sr.range[1] && part_sr.offset[2] == sr.offset[2] && part_sr.range[2] == sr.range[2];

			if(part_has_output_stride) {
				// Fast path: We can linearize this box right into the output
				const auto part_offset = (part_sr.offset[0] - sr.offset[0]) * sr.range[1] * sr.range[2];
				const auto part_linearized = static_cast<std::byte*>(out_linearized) + part_offset * buffer_info.element_size;
				pending_fast_transfers.merge(source.storage->get_data({source.get_local_offset(part_sr.offset), part_sr.range}, part_linearized));
			} else {
				// Slow path: Copy into a temporary buffer, linearize afterwards
				// TODO evolve get_data() to deal with output strides, otherwise we give up too much memory bandwidth for these temporary copies
				auto temp = make_uninitialized_payload<std::byte>(part_sr.range.size() * buffer_info.element_size);
				auto done = source.storage->get_data({source.get_local_offset(part_sr.offset), part_sr.range}, temp.get_pointer());
				pending_slow_transfers.emplace_back(part_sr, std::move(done), std::move(temp));
			}
#if TRACY_ENABLE
			fmt::format_to(std::back_inserter(zone_sr_text), "\n{} path from M{}: {}", part_has_output_stride ? "fast" : "slow", source_mid.value(), part_sr);
#endif
		}

#if TRACY_ENABLE
		ZoneText(zone_sr_text.data(), zone_sr_text.size());
#endif

		while(!pending_slow_transfers.empty()) {
			if(const auto done_it = std::find_if(pending_slow_transfers.begin(), pending_slow_transfers.end(), //
			       [](const auto& tuple) { return std::get<backend::async_event>(tuple).is_done(); });
			    done_it != pending_slow_transfers.end()) {
				auto& [part_sr, done, temp] = *done_it;

#if TRACY_ENABLE
				ZoneScopedN("linearize");
				const auto zone_linearize_text = fmt::to_string(part_sr);
				ZoneText(zone_linearize_text.data(), zone_linearize_text.size());
#endif

				switch(buffer_info.dims) {
				case 1:
					memcpy_strided(temp.get_pointer(), out_linearized, buffer_info.element_size, range_cast<1>(part_sr.range), id<1>(0),
					    range_cast<1>(sr.range), id_cast<1>(part_sr.offset - sr.offset), range_cast<1>(part_sr.range));
					break;
				case 2:
					memcpy_strided(temp.get_pointer(), out_linearized, buffer_info.element_size, range_cast<2>(part_sr.range), id<2>(0, 0),
					    range_cast<2>(sr.range), id_cast<2>(part_sr.offset - sr.offset), range_cast<2>(part_sr.range));
					break;
				case 3:
					memcpy_strided(temp.get_pointer(), out_linearized, buffer_info.element_size, part_sr.range, id<3>(0, 0, 0), sr.range,
					    part_sr.offset - sr.offset, part_sr.range);
					break;
				}

				pending_slow_transfers.erase(done_it);
			}
		}

		return pending_fast_transfers;
	}

	void buffer_manager::set_buffer_data(buffer_id bid, const subrange<3>& sr, shared_payload_ptr in_linearized) {
		std::unique_lock lock(m_mutex);
		assert(m_buffer_infos.count(bid) == 1);
		m_scheduled_transfers[bid].push_back(buffer_manager::transfer{std::move(in_linearized), sr});
	}

	bool buffer_manager::is_broadcast_possible(buffer_id bid, const subrange<3>& sr) const {
		std::unique_lock lock(m_mutex);

		auto& buff = m_buffers.at(bid);
		auto num_devices = m_local_devices.num_compute_devices();

		// check whether we can do this without resizing
		for(size_t device_id = 0; device_id < num_devices; ++device_id) {
			auto mem_id = m_local_devices.get_memory_id(device_id);
			if(is_resize_required(buff.get(mem_id), sr.range, sr.offset).resize_required) { return false; }
		}
		return true;
	}

	void buffer_manager::immediately_broadcast_data(buffer_id bid, const subrange<3>& sr, void* const in_linearized) {
		ZoneScoped;
#if TRACY_ENABLE
		const auto msg = fmt::format("broadcast bid {} - {}", bid, sr);
		ZoneText(msg.c_str(), msg.size());
#endif

		std::unique_lock lock(m_mutex);

		auto& buff = m_buffers.at(bid);
		auto num_devices = m_local_devices.num_compute_devices();
		std::vector<backend::async_event> transfer_events(num_devices);
		data_location all_loc;
		// upload to all devices
		for(size_t device_id = 0; device_id < num_devices; ++device_id) {
			auto mem_id = m_local_devices.get_memory_id(device_id);
			all_loc.set(mem_id);
			auto& buffer = buff.get(mem_id);
			transfer_events[device_id] = buffer.storage->set_data(in_linearized, sr.range, id<3>(0, 0, 0), buffer.get_local_offset(sr.offset), sr.range);
#if TRACY_ENABLE
			const auto msg = fmt::format("Broadcast set_data started for device {}", device_id);
			TracyMessage(msg.c_str(), msg.size());
#endif
		}

		// wait for uploads to complete
		for(size_t device_id = 0; device_id < num_devices; ++device_id) {
			transfer_events[device_id].wait();
#if TRACY_ENABLE
			const auto msg = fmt::format("Broadcast set_data completed for device {}", device_id);
			TracyMessage(msg.c_str(), msg.size());
#endif
		}

		// update data availability
		m_newest_data_location.at(bid).update_region(subrange_to_grid_box(sr), all_loc);
		m_authoritative_source.at(bid).update_region(subrange_to_grid_box(sr), m_local_devices.get_memory_id(0) /* arbitrary */);
	}

	buffer_manager::access_info buffer_manager::access_device_buffer(
	    const memory_id mid, buffer_id bid, cl::sycl::access::mode mode, const cl::sycl::range<3>& range, const cl::sycl::id<3>& offset) {
#if TRACY_ENABLE
		ZoneScopedN("access_device_buffer");
		const auto zone_text = fmt::format("B{} on M{}, mode {}, {}", bid, mid, (int)mode, subrange<3>{offset, range});
		ZoneText(zone_text.data(), zone_text.size());
#endif

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
			replacement_buf = backing_buffer{m_buffer_infos.at(bid).construct_device(range, device_queue, m_cuda_copy_streams.at(mid - 1).get()), offset};
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
					replacement_buf = backing_buffer{
					    m_buffer_infos.at(bid).construct_device(info.new_range, device_queue, m_cuda_copy_streams.at(mid - 1).get()), info.new_offset};
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
						access_host_buffer_impl(bid, access_mode::read, sr.range, sr.offset).pending_transfers.wait();
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
					replacement_buf = backing_buffer{
					    m_buffer_infos.at(bid).construct_device(spill_to_host ? range : info.new_range, device_queue, m_cuda_copy_streams.at(mid - 1).get()),
					    spill_to_host ? offset : info.new_offset};

					// We have wait()ed for the copy to host, so the host can be safely copied from later
					auto& authorative_source = m_authoritative_source.at(bid);
					for(auto& [box, source] : authorative_source.get_region_values(retain_region)) {
						if(source == mid) { authorative_source.update_region(retain_region, host_memory_id); }
					}
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
#if TRACY_ENABLE
		ZoneScopedN("make_buffer_subrange_coherent");
		{
			const auto zone_text = fmt::format("B{} on M{}, mode {}, {}", bid, mid, (int)mode, coherent_sr);
			ZoneText(zone_text.data(), zone_text.size());
		}
#endif
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
					intersection.scanByBoxes([&](const GridBox<3>& box) {
						const auto sr = grid_box_to_subrange(box);
#if TRACY_ENABLE
						ZoneScopedN("ingest transfer (partial)");
						const auto zone_text = fmt::format("{}", sr);
						ZoneText(zone_text.data(), zone_text.size());
#endif
						auto evt = target_buffer.storage->set_data(
						    t.linearized.get_pointer(), t.sr.range, sr.offset - t.sr.offset, target_buffer.get_local_offset(sr.offset), sr.range);
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

#if TRACY_ENABLE
				ZoneScopedN("ingest transfer (full)");
				const auto zone_text = fmt::format("{}", t.sr);
				ZoneText(zone_text.data(), zone_text.size());
#endif
				// Transfer applies fully.
				remaining_region_after_transfers = GridRegion<3>::difference(remaining_region_after_transfers, t.unconsumed);
				auto evt = target_buffer.storage->set_data(
				    t.linearized.get_pointer(), t.sr.range, id<3>(0, 0, 0), target_buffer.get_local_offset(t.sr.offset), t.sr.range);
				// auto evt = target_buffer.storage->copy_from_device_raw(t.get_buffer().get_owning_queue(), t.get_buffer().get_pointer(), t.sr.range, id<3>{},
				//     target_buffer.get_local_offset(t.sr.offset), t.sr.range, t.get_buffer().m_did, t.get_buffer().m_copy_stream);
				// auto evt = target_buffer.storage->copy(t.get_buffer(), id<3>{}, target_buffer.get_local_offset(t.sr.offset), t.sr.range);
				evt.hack_attach_payload(std::move(t.linearized)); // FIXME
				pending_transfers.merge(std::move(evt));
				updated_region = GridRegion<3>::merge(updated_region, t.unconsumed);
			}
			// The target buffer now has the newest data in this region.
			{
				// NOCOMMIT TODO: Add regression test: This used to be data_location{locs}.set(mid), i.e., the previous location remained valid.
				m_newest_data_location.at(bid).update_region(updated_region, data_location{}.set(mid));
				m_authoritative_source.at(bid).update_region(updated_region, mid);
			}
			scheduled_buffer_transfers = std::move(remaining_transfers);
		}

		if(!remaining_region_after_transfers.empty()) {
			const auto maybe_retain_box = [&](const GridBox<3>& box) {
				if(detail::access::mode_traits::is_consumer(mode)) {
					// If we are accessing the buffer using a consumer mode, we have to retain the full previous contents, otherwise...
					const auto box_sr = grid_box_to_subrange(box);
#if TRACY_ENABLE
					ZoneScopedN("consumer retain");
					const auto zone_text = fmt::format("copy {}", box_sr);
					ZoneText(zone_text.c_str(), zone_text.size());
#endif
					auto evt = target_buffer.storage->copy(
					    *previous_buffer.storage, previous_buffer.get_local_offset(box_sr.offset), target_buffer.get_local_offset(box_sr.offset), box_sr.range);
					pending_transfers.merge(std::move(evt));
				} else {
					// ...check if there are parts of the previous buffer that we are not going to overwrite (and thus have to retain).
					// If so, copy only those parts.
					const auto remaining_region = GridRegion<3>::difference(box, coherent_box);
					remaining_region.scanByBoxes([&](const GridBox<3>& small_box) {
						const auto small_box_sr = grid_box_to_subrange(small_box);
#if TRACY_ENABLE
						ZoneScopedN("pure-producer retain");
						const auto zone_text = fmt::format("copy {}", small_box_sr);
						ZoneText(zone_text.c_str(), zone_text.size());
#endif
						auto evt = target_buffer.storage->copy(*previous_buffer.storage, previous_buffer.get_local_offset(small_box_sr.offset),
						    target_buffer.get_local_offset(small_box_sr.offset), small_box_sr.range);
						pending_transfers.merge(std::move(evt));
					});
				}
			};

			GridRegion<3> replicated_region;
			const auto data_locations = m_newest_data_location.at(bid).get_region_values(remaining_region_after_transfers);
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

				replicated_region = GridRegion<3>::merge(replicated_region, box);
			}

			for(auto& [box, source_mid] : m_authoritative_source.at(bid).get_region_values(replicated_region)) {
				assert(source_mid.value() != mid);
				assert(m_buffers.at(bid).get(source_mid.value()).is_allocated());
				const auto box_sr = grid_box_to_subrange(box);
				const auto& source_buffer = m_buffers.at(bid).get(source_mid.value());
#if TRACY_ENABLE
				ZoneScopedN("replicate");
				const auto zone_text = fmt::format("copy {}", box_sr);
				ZoneText(zone_text.c_str(), zone_text.size());
#endif
				auto evt = target_buffer.storage->copy(
				    *source_buffer.storage, source_buffer.get_local_offset(box_sr.offset), target_buffer.get_local_offset(box_sr.offset), box_sr.range);
				pending_transfers.merge(std::move(evt));
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

		if(detail::access::mode_traits::is_producer(mode)) {
			m_newest_data_location.at(bid).update_region(coherent_box, data_location{}.set(mid));
			m_authoritative_source.at(bid).update_region(coherent_box, mid);
		}

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
