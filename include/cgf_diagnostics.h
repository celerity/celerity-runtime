#pragma once

#include <optional>

#include "task.h"

namespace celerity::detail {

class cgf_diagnostics {
  public:
	static void make_available() {
		assert(m_instance == nullptr);
		m_instance = std::unique_ptr<cgf_diagnostics>(new cgf_diagnostics());
	}

	static bool is_available() { return m_instance != nullptr; }

	static void teardown() { m_instance.reset(); }

	static cgf_diagnostics& get_instance() {
		assert(m_instance != nullptr);
		return *m_instance;
	}

	cgf_diagnostics(const cgf_diagnostics&) = delete;
	cgf_diagnostics(cgf_diagnostics&&) = delete;
	cgf_diagnostics operator=(const cgf_diagnostics&) = delete;
	cgf_diagnostics operator=(cgf_diagnostics&&) = delete;
	~cgf_diagnostics() = default;

	template <target Tgt, typename Closure, std::enable_if_t<Tgt == target::device, int> = 0>
	void check(const Closure& kernel, const buffer_access_map& buffer_accesses) {
		static_assert(std::is_copy_constructible_v<std::decay_t<Closure>>);
		check(target::device, kernel, &buffer_accesses, 0);
	}

	template <target Tgt, typename Closure, std::enable_if_t<Tgt == target::host_task, int> = 0>
	void check(const Closure& kernel, const buffer_access_map& buffer_accesses, const size_t non_void_side_effects_count) {
		static_assert(std::is_copy_constructible_v<std::decay_t<Closure>>);
		check(target::host_task, kernel, &buffer_accesses, non_void_side_effects_count);
	}

	bool is_checking() const { return m_is_checking; }

	void register_accessor(const hydration_id hid, const target tgt) {
		assert(m_is_checking);
		assert(hid - 1 < m_expected_buffer_accesses->get_num_accesses());
		if(tgt != m_expected_target) {
			throw std::runtime_error(fmt::format("Accessor {} for buffer {} has wrong target ('{}' instead of '{}').", hid - 1,
			    m_expected_buffer_accesses->get_nth_access(hid - 1).first, tgt == target::device ? "device" : "host_task",
			    m_expected_target == target::device ? "device" : "host_task"));
		}
		m_registered_buffer_accesses.at(hid - 1) = true;
	}

	void register_side_effect() {
		if(!m_is_checking) return;
		if(m_expected_target != target::host_task) { throw std::runtime_error("Side effects can only be used in host tasks."); }
		m_registered_side_effect_count++;
	}

  private:
	inline static thread_local std::unique_ptr<cgf_diagnostics> m_instance; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

	bool m_is_checking = false;
	std::optional<target> m_expected_target = std::nullopt;
	const buffer_access_map* m_expected_buffer_accesses = nullptr;
	std::vector<bool> m_registered_buffer_accesses;
	size_t m_expected_side_effects_count = 0;
	size_t m_registered_side_effect_count = 0;

	cgf_diagnostics() = default;

	template <typename Closure>
	void check(const target tgt, const Closure& kernel, const buffer_access_map* const buffer_accesses, const size_t expected_side_effects_count) {
		m_expected_target = tgt;
		m_expected_buffer_accesses = buffer_accesses;
		m_registered_buffer_accesses.clear();
		m_registered_buffer_accesses.resize(m_expected_buffer_accesses->get_num_accesses());
		m_expected_side_effects_count = expected_side_effects_count;
		m_registered_side_effect_count = 0;

		m_is_checking = true;
		try {
			[[maybe_unused]] auto copy = kernel;
		} catch(...) {
			m_is_checking = false;
			throw;
		}
		m_is_checking = false;
		m_expected_target = std::nullopt;

		for(size_t i = 0; i < m_expected_buffer_accesses->get_num_accesses(); ++i) {
			if(!m_registered_buffer_accesses[i]) {
				throw std::runtime_error(fmt::format("Accessor {} for buffer {} is not being copied into the kernel. This indicates a bug. Make sure "
				                                     "the accessor is captured by value and not by reference, or remove it entirely.",
				    i, m_expected_buffer_accesses->get_nth_access(i).first));
			}
		}

		if(tgt == target::host_task) {
			if(m_registered_side_effect_count < m_expected_side_effects_count) {
				throw std::runtime_error(
				    fmt::format("The number of side effects copied into the kernel is fewer ({}) than expected ({}). This may be indicative "
				                "of a bug. Make sure all side effects are captured by value and not by reference, and remove unused ones.",
				        m_registered_side_effect_count, m_expected_side_effects_count));
			}
			// TODO: We could issue a warning here when the number of registered side effects is higher than expected (which may be legitimate, due to copies).
		}
	}
};

} // namespace celerity::detail
