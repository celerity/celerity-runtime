#pragma once

#include <array>
#include <chrono>
#include <vector>
#include <list>
#include <thread>
#include <fstream>

#include "allscale/api/core/impl/reference/task_id.h"

namespace allscale {
namespace api {
namespace core {
namespace impl {
namespace reference {

	/**
	 * A log entry within the performance log.
	 */
	class ProfileLogEntry {

	public:

		/**
		 * Codes enumerating possible events.
		 */
		enum Kind {
			// worker events
			WorkerCreated,			// < the first event to be logged
			WorkerSuspended,		// < a worker thread is suspended
			WorkerResumed,			// < a worker thread is resumed
			WorkerDestroyed,		// < a worker thread is terminated

			// task events
			TaskStolen,				// < a task got stolen
			TaskSplit,				// < a task got split
			TaskStarted,			// < a task processing got started
			TaskEnded,				// < a task processing finished

			// control events
			EndOfStream,			// < the last event, to mark the end of a stream
		};

	private:

		uint64_t time;

		Kind kind;

		TaskID task;

		ProfileLogEntry(uint64_t time, Kind kind)
			: time(time), kind(kind), task() {}

		ProfileLogEntry(uint64_t time, Kind kind, TaskID task)
			: time(time), kind(kind), task(task) {}

	public:

		ProfileLogEntry() =default;

		// -- observers --

		uint64_t getTimestamp() const {
			return time;
		}

		Kind getKind() const {
			return kind;
		}

		TaskID getTask() const {
			return task;
		}

		// -- factories --

		static ProfileLogEntry createWorkerCreatedEntry() {
			return ProfileLogEntry(getCurrentTime(), WorkerCreated);
		}

		static ProfileLogEntry createWorkerDestroyedEntry() {
			return ProfileLogEntry(getCurrentTime(), WorkerDestroyed);
		}

		static ProfileLogEntry createWorkerSuspendedEntry() {
			return ProfileLogEntry(getCurrentTime(), WorkerSuspended);
		}

		static ProfileLogEntry createWorkerResumedEntry() {
			return ProfileLogEntry(getCurrentTime(), WorkerResumed);
		}

		static ProfileLogEntry createTaskStolenEntry(const TaskID& task) {
			return ProfileLogEntry(getCurrentTime(), TaskStolen, task);
		}

		static ProfileLogEntry createTaskStartedEntry(const TaskID& task) {
			return ProfileLogEntry(getCurrentTime(), TaskStarted, task);
		}

		static ProfileLogEntry createTaskEndedEntry(const TaskID& task) {
			return ProfileLogEntry(getCurrentTime(), TaskEnded, task);
		}

		// -- utility functions --

		bool operator<(const ProfileLogEntry& other) {
			// sort events by time
			return time < other.time;
		}

		friend std::ostream& operator<<(std::ostream& out, const ProfileLogEntry& entry) {

			out << "@" << entry.time << ":";

			switch(entry.kind) {
			// worker events
			case WorkerCreated:   return out << "Worker created";
			case WorkerSuspended: return out << "Worker suspended";
			case WorkerResumed:   return out << "Worker resumed";
			case WorkerDestroyed: return out << "Worker destroyed";

			// task events
			case TaskStolen:      return out << "Task " << entry.task << " stolen";
			case TaskSplit:       return out << "Task " << entry.task << " split";
			case TaskStarted:     return out << "Task " << entry.task << " started";
			case TaskEnded:       return out << "Task " << entry.task << " ended";

			// everything else
			default:              return out << "Unknown event!";
			}
		}

	private:

		/**
		 * A utility to retrieve a timestamp for events.
		 */
		static uint64_t getCurrentTime() {
			static thread_local uint64_t last = 0;

			// get current time
			uint64_t cur = std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::high_resolution_clock::now().time_since_epoch()
			).count();

			// make sure time is progressing
			if (cur > last) {
				last = cur;
				return cur;
			}

			// increase by at least one time step
			return last+1;
		}

	};



	class ProfileLog {

	public:

		// the block size of the log
		enum { BATCH_SIZE = 100000 };

	private:

		using block_t = std::array<ProfileLogEntry,BATCH_SIZE>;
		using block_list_t = std::list<block_t>;

		using block_const_iter = block_list_t::const_iterator;
		using block_iter = block_list_t::iterator;

		using entry_const_iter = block_t::const_iterator;
		using entry_iter = block_t::iterator;

		// the log entries, organized in blocks of N entries
		block_list_t data;

		entry_iter next;
		entry_iter endOfBlock;

	public:

		ProfileLog() : next(nullptr), endOfBlock(nullptr) {}

		void addEntry(const ProfileLogEntry& entry) {
			// create a new block if necessary
			if (next == endOfBlock) {
				data.emplace_back();
				next = data.back().begin();
				endOfBlock = data.back().end();
			}

			// insert entry
			*next = entry;
			++next;
		}

		ProfileLog& operator<<(const ProfileLogEntry& entry) {
			addEntry(entry);
			return *this;
		}


		// -- log entry iteration --

		class iterator : public std::iterator<std::input_iterator_tag,ProfileLogEntry> {

			block_const_iter b_cur;
			block_const_iter b_end;

			entry_const_iter e_cur;
			entry_const_iter e_end;

			entry_const_iter log_end;

		public:

			static iterator begin(const block_list_t& blocks, const entry_const_iter& log_end) {
				iterator res;
				res.b_cur = blocks.begin();
				res.b_end = blocks.end();
				if (res.isEnd()) return res;
				res.e_cur = res.b_cur->begin();
				res.e_end = res.b_cur->end();
				res.log_end = log_end;
				return res;
			}

			static iterator end(const block_list_t& blocks) {
				iterator res;
				res.b_cur = blocks.end();
				res.b_end = blocks.end();
				return res;
			}

			bool operator==(const iterator& other) const {
				return isEnd() && other.isEnd();
			}

			bool operator!=(const iterator& other) const {
				return !(*this == other);
			}

			const ProfileLogEntry& operator*() const {
				return *e_cur;
			}

			iterator& operator++() {
				// go to next entry
				++e_cur;

				// if it is the end of the log => jump to end of iterator range
				if (e_cur == log_end) {
					b_cur = b_end;
					return *this;
				}

				// if not end of current block is reached, continue
				if (e_cur != e_end) return *this;

				// go to next block
				b_cur++;

				// if there is none, mark as done
				if (b_cur == b_end) return *this;

				// walk into next block
				e_cur = b_cur->begin();
				e_end = b_cur->end();
				return *this;
			}

		private:

			bool isEnd() const {
				return b_cur == b_end;
			}

		};

		iterator begin() const {
			return iterator::begin(data,next);
		}

		iterator end() const {
			return iterator::end(data);
		}


		void saveTo(std::ostream& out) {
			// save the number of blocks
			std::size_t num_blocks = data.size();
			out.write((char*)&num_blocks,sizeof(num_blocks));

			// save the offset of the last block
			std::size_t offset = 0;
			if (num_blocks > 0) {
				offset = next - data.back().begin();
			}
			out.write((char*)&offset,sizeof(offset));

			// save all blocks
			for(const auto& cur : data) {
				out.write((char*)&cur,sizeof(block_t));
			}
		}

		void saveTo(const std::string& file) {
			std::fstream trg(file.c_str(), std::ios::out | std::ios::binary);
			saveTo(trg);
		}

		static ProfileLog loadFrom(std::istream& in) {
			// load the number of blocks
			std::size_t num_blocks;
			in.read((char*)&num_blocks,sizeof(num_blocks));

			// load the offset for the last block
			std::size_t offset;
			in.read((char*)&offset,sizeof(offset));

			ProfileLog log;
			for(std::size_t i = 0; i<num_blocks; i++) {
				log.data.emplace_back();
				in.read((char*)&log.data.back(),sizeof(block_t));
			}

			// move next pointer to last position
			if (num_blocks > 0) {
				log.next = log.data.back().begin() + offset;
			}

			// done
			return log;
		}

		static ProfileLog loadFrom(const std::string& file) {
			std::fstream src(file.c_str(), std::ios::in | std::ios::binary);
			return loadFrom(src);
		}

	};

	inline std::string getLogFileNameForWorker(int id) {
		// create the filename
		char filename[17];
		assert_lt(id, 10000) << "Unexpectedly larger number of workers";
		snprintf(filename, 17, "profile_log.%04d", ((unsigned)id)%10000);
		return filename;
	}

	static inline int& getCurrentWorkerID() {
		static thread_local int workerID;
		return workerID;
	}

	static inline void setCurrentWorkerID(int id) {
		getCurrentWorkerID() = id;
	}

	namespace detail {

		struct ProfileLogHandler {
			ProfileLog log;

			~ProfileLogHandler() {
				// save log to the chosen filename
				log.saveTo(getLogFileNameForWorker(getCurrentWorkerID()));
			}
		};

		inline ProfileLog& getProfileLog() {
			static thread_local ProfileLogHandler logHandler;
			return logHandler.log;
		}

		inline void logProfilerEventInternal(const ProfileLogEntry& entry) {
			getProfileLog() << entry;
		}

	}


	#ifdef ENABLE_PROFILING

		const bool PROFILING_ENABLED = true;

		#define logProfilerEvent(EVENT) \
			allscale::api::core::impl::reference::detail::logProfilerEventInternal(EVENT)

	#else

		const bool PROFILING_ENABLED = false;

		#define logProfilerEvent(EVENT) /* ignore */

	#endif



} // end namespace reference
} // end namespace impl
} // end namespace core
} // end namespace api
} // end namespace allscale
