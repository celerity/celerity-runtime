#pragma once

#include <atomic>
#include <thread>

#if defined _MSC_VER
// required for YieldProcessor macro
#define NOMINMAX
#include "windows.h"
//#elif defined (__ppc64__) || defined (_ARCH_PPC64)

#endif

namespace allscale {
namespace api {
namespace core {
inline namespace simple {

	/* Pause instruction to prevent excess processor bus usage */
	
#ifdef _MSC_VER
#define cpu_relax() YieldProcessor()
#elif defined (__ppc64__) || defined (_ARCH_PPC64)
#define __barrier() __asm__ volatile("": : :"memory")
#define __HMT_low() __asm__ volatile("or 1,1,1     # low priority")
#define __HMT_medium() __asm__ volatile("or 2,2,2     # medium priority")
#define cpu_relax() do { __HMT_low(); __HMT_medium(); __barrier(); } while (0)
#else
#define cpu_relax() __builtin_ia32_pause()
#endif

	class Waiter {
		int i;
	public:
		Waiter() : i(0) {}

		void operator()() {
			++i;
			if ((i % 1000) == 0) {
				// there was no progress => let others work
				std::this_thread::yield();
			} else {
				// relax this CPU
				cpu_relax();
			}
		}
	};



    class SpinLock {
        std::atomic<int> lck;
    public:

        SpinLock() : lck(0) {
        }

        void lock() {
            Waiter wait;
            while(!try_lock()) wait();
        }

        bool try_lock() {
            int should = 0;
            return lck.compare_exchange_weak(should, 1, std::memory_order_acquire);
        }

        void unlock() {
            lck.store(0, std::memory_order_release);
        }
    };

	/**
	 * An optimistic read/write lock.
	 */
	class OptimisticReadWriteLock {

		/**
		 * The type utilized for the version numbering.
		 */
		using version_t = std::size_t;

		/**
		 * The version number.
		 *  - even: there is no write in progress
		 *  - odd: there is a write in progress, do not allow read operations
		 */
		std::atomic<version_t> version;

	public:

		/**
		 * The lease utilized to link start and end of read phases.
		 */
		class Lease {
			friend class OptimisticReadWriteLock;
			version_t version;
		public:
			Lease(version_t version = 0) : version(version) {}
			Lease(const Lease& lease) = default;
			Lease& operator=(const Lease& other) = default;
			Lease& operator=(Lease&& other) = default;
		};

		OptimisticReadWriteLock() : version(0) {}

		/**
		 * Starts a read phase, making sure that there is currently no
		 * active concurrent modification going on. The resulting lease
		 * enables the invoking process to later-on verify that no
		 * concurrent modifications took place.
		 */
		Lease start_read() {
			Waiter wait;

			// get a snapshot of the lease version
			auto v = version.load(std::memory_order_acquire);

			// spin while there is a write in progress
			while((v & 0x1) == 1) {
				// wait for a moment
				wait();
				// get an updated version
				v = version.load(std::memory_order_acquire);
			}

			// done
			return Lease(v);
		}

		/**
		 * Tests whether there have been concurrent modifications since
		 * the given lease has been issued.
		 *
		 * @return true if no updates have been conducted, false otherwise
		 */
		bool validate(const Lease& lease) {
			// check whether version number has changed in the mean-while
			return lease.version == version.load(std::memory_order_consume);
		}

		/**
		 * Ends a read phase by validating the given lease.
		 *
		 * @return true if no updates have been conducted since the
		 *         issuing of the lease, false otherwise
		 */
		bool end_read(const Lease& lease) {
			// check lease in the end
			return validate(lease);
		}

		/**
		 * Starts a write phase on this lock be ensuring exclusive access
		 * and invalidating any existing read lease.
		 */
		void start_write() {
			Waiter wait;

			// set last bit => make it odd
			auto v = version.fetch_or(0x1, std::memory_order_acquire);

			// check for concurrent writes
			while((v & 0x1) == 1) {
				// wait for a moment
				wait();
				// get an updated version
				v = version.fetch_or(0x1, std::memory_order_acquire);
			}

			// done
		}

		/**
		 * Tries to start a write phase unless there is a currently ongoing
		 * write operation. In this case no write permission will be obtained.
		 *
		 * @return true if write permission has been granted, false otherwise.
		 */
		bool try_start_write() {
			auto v = version.fetch_or(0x1, std::memory_order_acquire);
			return !(v & 0x1);
		}

		/**
		 * Updates a read-lease to a write permission by a) validating that the
		 * given lease is still valid and b) making sure that there is no currently
		 * ongoing write operation.
		 *
		 * @return true if the lease was still valid and write permissions could
		 *      be granted, false otherwise.
		 */
		bool try_upgrade_to_write(const Lease& lease) {
			auto v = version.fetch_or(0x1, std::memory_order_acquire);

			// check whether write privileges have been gained
			if (v & 0x1) return false;// there is another writer already

			// check whether there was no write since the gain of the read lock
			if (lease.version == v) return true;

			// if there was, undo write update
			abort_write();

			// operation failed
			return false;
		}

		/**
		 * Aborts a write operation by reverting to the version number before
		 * starting the ongoing write, thereby re-validating existing leases.
		 */
		void abort_write() {
			// reset version number
			version.fetch_sub(1,std::memory_order_release);
		}

		/**
		 * Ends a write operation by giving up the associated exclusive access
		 * to the protected data and abandoning the provided write permission.
		 */
		void end_write() {
			// update version number another time
			version.fetch_add(1,std::memory_order_release);
		}

		/**
		 * Tests whether currently write permissions have been granted to any
		 * client by this lock.
		 *
		 * @return true if so, false otherwise
		 */
		bool is_write_locked() const {
			return version & 0x1;
		}

	};

} // end namespace simple
} // end namespace core
} // end namespace api
} // end namespace allscale
