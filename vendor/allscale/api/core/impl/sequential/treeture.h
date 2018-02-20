#pragma once

#include <vector>

#include "allscale/utils/assert.h"
#include "allscale/utils/printer/arrays.h"

namespace allscale {
namespace api {
namespace core {
namespace impl {
namespace sequential {


	// --------------------------------------------------------------------------------------------
	//								   sequential treeture implementation
	// --------------------------------------------------------------------------------------------


	// ------------------------------------- Declarations -----------------------------------------

	/**
	 * The actual treeture, referencing the computation of a value.
	 */
	template<typename T>
	class treeture;

	/**
	 * A treeture not yet released to the runtime system for execution.
	 */
	template<typename T>
	class unreleased_treeture;

	/**
	 * A handle for a lazily constructed unreleased treeture. This intermediate construct is utilized
	 * for writing templated code that can be optimized to overhead-less computed values and to facilitate
	 * the support of the sequence combinator.
	 */
	template<typename T, typename Gen>
	class lazy_unreleased_treeture;

	/**
	 * A class to reference tasks for synchronization purposes.
	 */
	class task_reference;

	/**
	 * A class to model task dependencies
	 */
	class dependencies;


	// ------------------------------------- Definitions ------------------------------------------

	// -- task_reference --

	class task_reference {

		bool isDone() const {
			return true;
		}

		void wait() const {
			// always done
		}

		task_reference& descentLeft() {
			return *this;
		}

		task_reference& descentRight() {
			return *this;
		}

		task_reference getLeft() const {
			return *this;
		}

		task_reference getRight() const {
			return *this;
		}

	};


	// -- treeture --

	template<>
	class treeture<void> : public task_reference {
	public:

		using value_type = void;

		treeture() {}

		template<typename Fun>
		explicit treeture(Fun&& fun) {
			fun();
		}

		template<typename T>
		treeture(const treeture<T>& /*other*/) {}

		void get() const {
			// nothing to do
		}

	};

	template<typename T>
	class treeture : public task_reference {

		T value;

	public:

		using value_type = T;

		using treeture_type = treeture<T>;

		treeture() {}

		treeture(const T& value)
			: value(value) {}

		treeture(const T&& value)
			: value(std::move(value)) {}

		template<typename Fun>
		explicit treeture(Fun&& fun)
			: value(fun()) {}

		T get() const {
			return value;
		}

	};


	template<typename Op, typename R = std::result_of_t<Op()>>
	treeture<R> make_treeture(Op&& op) {
		return treeture<R>(std::move(op));
	}

	// -- unreleased_treeture --

	template<typename T>
	class unreleased_treeture : public task_reference {

		treeture<T> res;

	public:

		using value_type = T;

		using treeture_type = treeture<T>;

		unreleased_treeture() {}

		template<typename Fun>
		explicit unreleased_treeture(Fun&& fun)
			: res(fun()) {}

		unreleased_treeture(const unreleased_treeture&) =delete;
		unreleased_treeture(unreleased_treeture&&) =default;

		unreleased_treeture& operator=(const unreleased_treeture&) =delete;
		unreleased_treeture& operator=(unreleased_treeture&&) =default;

		treeture<T> release() const && {
			return res;
		}

		operator treeture<T>() const && {
			return std::move(*this).release();
		}

		T get() const && {
			return std::move(*this).release().get();
		}

	};

	template<typename Gen, typename T = typename std::result_of_t<Gen()>::value_type>
	unreleased_treeture<T> make_unreleased_treeture(Gen&& gen) {
		return unreleased_treeture<T>(std::move(gen));
	}

	template<typename T,typename Gen>
	class lazy_unreleased_treeture {

		mutable Gen gen;

	public:

		using value_type = T;

		using treeture_type = treeture<T>;

		explicit lazy_unreleased_treeture(Gen&& gen)
			: gen(std::move(gen)) {}

		unreleased_treeture<T> toUnreleasedTreeture() const {
			return gen();
		}

		treeture<T> release() const {
			return toUnreleasedTreeture();
		}

		T get() const {
			return release().get();
		}

		operator unreleased_treeture<T>() const {
			return toUnreleasedTreeture();
		}

		operator treeture<T>() const {
			return release();
		}

	};

	template<typename Gen, typename T = typename std::result_of_t<Gen()>::value_type>
	lazy_unreleased_treeture<T,Gen> make_lazy_unreleased_treeture(Gen&& gen) {
		return lazy_unreleased_treeture<T,Gen>(std::move(gen));
	}

	/**
	 * There are no dependencies to be recorded, so this object is an empty object.
	 */
	class dependencies {};


	// -------------------------------------- Operators -------------------------------------------


	inline dependencies after() {
		return {};
	}

	template<typename ... Rest>
	dependencies after(const task_reference&, const Rest& ... rest) {
		return after(rest...);
	}

	inline dependencies after(const std::vector<task_reference>&) {
		return {};		// if it is a task_reference, it is computed
	}


	inline auto done() {
		return make_lazy_unreleased_treeture([=](){
			return make_unreleased_treeture([=](){ return treeture<void>(); });
		});
	}

	template<typename T>
	auto done(const T& value) {
		return make_lazy_unreleased_treeture([=](){
			return make_unreleased_treeture([=](){ return treeture<T>(value); });
		});
	}


	template<typename Op>
	auto spawn(dependencies&&, Op&& op) {
		return make_lazy_unreleased_treeture([=](){
			return make_unreleased_treeture([=](){ return make_treeture(std::move(op)); });
		});
	}

	template<typename Op>
	auto spawn(Op&& op) {
		return spawn(after(),std::move(op));
	}


	inline auto seq() {
		return done();
	}

	template<typename F, typename FA, typename ... R, typename ... RA>
	auto seq(dependencies&&, lazy_unreleased_treeture<F,FA>&& f, lazy_unreleased_treeture<R,RA>&& ... rest) {
		return make_lazy_unreleased_treeture([f,rest...]() mutable {
			return make_unreleased_treeture([f,rest...]() mutable {
				return make_treeture([f,rest...]() mutable {
					f.get();
					seq(std::move(rest)...).get();
				});
			});
		});
	}

	template<typename F, typename FA, typename ... R, typename ... RA>
	auto seq(lazy_unreleased_treeture<F,FA>&& f, lazy_unreleased_treeture<R,RA>&& ... rest) {
		return seq(after(), std::move(f),std::move(rest)...);
	}

	template<typename ... T, typename ... TA>
	auto par(dependencies&&, lazy_unreleased_treeture<T,TA>&& ... tasks) {
		// for the sequential implementation, parallel is the same as sequential
		return seq(std::move(tasks)...);
	}

	template<typename ... T, typename ... TA>
	auto par(lazy_unreleased_treeture<T,TA>&& ... tasks) {
		return par(after(), std::move(tasks)...);
	}


	template<typename A, typename AA, typename B, typename BA, typename M>
	auto combine(dependencies&&, lazy_unreleased_treeture<A,AA>&& a, lazy_unreleased_treeture<B,BA>&& b, M&& m, bool = true) {
		return make_lazy_unreleased_treeture([=]() {
			return make_unreleased_treeture([=]() {
				return make_treeture([=]() {
					return m(a.get(),b.get());
				});
			});
		});
	}

	template<typename A, typename AA, typename B, typename BA, typename M>
	auto combine(lazy_unreleased_treeture<A,AA>&& a, lazy_unreleased_treeture<B,BA>&& b, M&& m, bool parallel = true) {
		return sequential::combine(after(), std::move(a), std::move(b), std::move(m), parallel);
	}


} // end namespace sequential
} // end namespace impl
} // end namespace core
} // end namespace api
} // end namespace allscale

