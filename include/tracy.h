#pragma once

#include <cassert>
#include <optional>
#include <vector>

#include <spdlog/fmt/fmt.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

namespace celerity::detail {

#if TRACY_ENABLE
struct lane_info {
	std::unique_ptr<std::string> name;
	bool is_free;
};

// Since we pass static data to Tracy, we need to control its lifetime manually;
// otherwise we risk running into static destruction order UB.
// TODO: Still crashes on Marconi 100? Need to investigate further.
struct tracy_static_context {
	std::vector<lane_info> lanes = {};

	tracy_static_context() { tracy::StartupProfiler(); }

	~tracy_static_context() { tracy::ShutdownProfiler(); }
};
#endif

// Workaround for https://github.com/wolfpld/tracy/issues/426
class tracy_async_lane {
  public:
	void initialize() {
#if TRACY_ENABLE
		assert(!m_started);
		m_lane_id = get_free_lane();
		m_started = true;
#endif
	}

	void destroy() {
#if TRACY_ENABLE
		assert(m_started);
		TracyFiberEnter(tracy_sctx.lanes[m_lane_id].name->c_str());
		if(m_current_zone.has_value()) { TracyCZoneEnd(*m_current_zone); }
		return_lane(m_lane_id);
		TracyFiberLeave;
		m_started = false;
#endif
	}

	void activate() {
#if TRACY_ENABLE
		assert(m_started);
		TracyFiberEnter(tracy_sctx.lanes[m_lane_id].name->c_str());
#endif
	}

	void deactivate() {
#if TRACY_ENABLE
		assert(m_started);
		TracyFiberLeave;
#endif
	}

	void begin_phase(const std::string& name, const std::string& description, const tracy::Color::ColorType color) {
#if TRACY_ENABLE
		assert(m_started);
		TracyFiberEnter(tracy_sctx.lanes[m_lane_id].name->c_str());
		if(m_current_zone.has_value()) { TracyCZoneEnd(*m_current_zone); }
		TracyCZone(t_ctx, true);
		TracyCZoneName(t_ctx, name.c_str(), name.size());
		TracyCZoneText(t_ctx, description.c_str(), description.size());
		TracyCZoneColor(t_ctx, color);
		TracyFiberLeave;
		m_current_zone = t_ctx;
#endif
	}

  private:
#if TRACY_ENABLE
	bool m_started = false;
	size_t m_lane_id = -1;
	std::optional<TracyCZoneCtx> m_current_zone;

	inline static tracy_static_context tracy_sctx;

	static size_t get_free_lane() {
		for(size_t lane = 0; lane < tracy_sctx.lanes.size(); ++lane) {
			if(tracy_sctx.lanes[lane].is_free) {
				tracy_sctx.lanes[lane].is_free = false;
				return lane;
			}
		}
		tracy_sctx.lanes.push_back({std::make_unique<std::string>(fmt::format("celerity async {:02}", tracy_sctx.lanes.size())), false});
		return tracy_sctx.lanes.size() - 1;
	}

	static void return_lane(size_t lane_id) {
		assert(!tracy_sctx.lanes.at(lane_id).is_free);
		tracy_sctx.lanes[lane_id].is_free = true;
	}
#endif
};

} // namespace celerity::detail
