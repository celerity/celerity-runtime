#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <type_traits>

#ifdef __linux__
	#include <pthread.h>
#endif

#include "allscale/utils/assert.h"
#include "allscale/utils/bitmanipulation.h"

#include "allscale/api/core/impl/reference/lock.h"
#include "allscale/api/core/impl/reference/profiling.h"
#include "allscale/api/core/impl/reference/queue.h"
#include "allscale/api/core/impl/reference/runtime_predictor.h"

namespace allscale {
namespace api {
namespace core {
namespace impl {
namespace reference {

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
	 * A reference to a task to synchronize upon it.
	 */
	class task_reference;

	/**
	 * A class to model task dependencies
	 */
	template<typename size>
	class dependencies;



	// ---------------------------------------------------------------------------------------------
	//								    Internal Forward Declarations
	// ---------------------------------------------------------------------------------------------


	class TaskBase;

	template<typename T>
	class Task;


	// ---------------------------------------------------------------------------------------------
	//											  Debugging
	// ---------------------------------------------------------------------------------------------


	// -- Declarations --

	const bool REFERENCE_RUNTIME_DEBUG = false;

	inline std::mutex& getLogMutex() {
		static std::mutex m;
		return m;
	}

	#define LOG(MSG) \
		{  \
			if (REFERENCE_RUNTIME_DEBUG) { \
				std::thread::id this_id = std::this_thread::get_id(); \
				std::lock_guard<std::mutex> lock(getLogMutex()); \
				std::cerr << "Thread " << this_id << ": " << MSG << "\n"; \
			} \
		}

	const bool DEBUG_SCHEDULE = false;

	#define LOG_SCHEDULE(MSG) \
		{  \
			if (DEBUG_SCHEDULE) { \
				std::thread::id this_id = std::this_thread::get_id(); \
				std::lock_guard<std::mutex> lock(getLogMutex()); \
				std::cerr << "Thread " << this_id << ": " << MSG << "\n"; \
			} \
		}

	const bool DEBUG_TASKS = false;

	#define LOG_TASKS(MSG) \
		{  \
			if (DEBUG_TASKS) { \
				std::thread::id this_id = std::this_thread::get_id(); \
				std::lock_guard<std::mutex> lock(getLogMutex()); \
				std::cerr << "Thread " << this_id << ": " << MSG << "\n"; \
			} \
		}



	// -----------------------------------------------------------------
	//						  Monitoring (for Debugging)
	// -----------------------------------------------------------------


	const bool MONITORING_ENABLED = false;

	namespace monitoring {

		enum class EventType {
			Run, RunDirect, Split, Wait, DependencyWait
		};

		struct Event {

			EventType type;

			const TaskBase* task;

			TaskID taskId;

			bool operator==(const Event& other) const {
				return other.type == type && other.task == task && other.taskId == taskId;
			}

			friend std::ostream& operator<<(std::ostream& out, const Event& e);
		};


		class ThreadState {

			using guard = std::lock_guard<std::mutex>;

			std::thread::id thread_id;

			std::mutex lock;

			std::vector<Event> eventStack;

		public:

			ThreadState() : thread_id(std::this_thread::get_id()) {
				guard g(getStateLock());
				getStates().push_back(this);
			}

			~ThreadState() {
				assert_true(eventStack.empty());
			}

			void pushEvent(const Event& e) {
				guard g(lock);
				eventStack.push_back(e);
			}

			void popEvent(__allscale_unused const Event& e) {
				guard g(lock);
				assert_eq(e,eventStack.back());
				eventStack.pop_back();
			}

			void dumpState(std::ostream& out) {
				guard g(lock);
				out << "\nThread: " << thread_id << "\n";
				out << "\tStack:\n";
				for(const auto& cur : eventStack) {
					out << "\t\t" << cur << "\n";
				}
				out << "\t\t -- top of stack --\n";
				out << "\n";
			}

			static void dumpStates(std::ostream& out) {
				// lock states
				std::lock_guard<std::mutex> g(getStateLock());

				// provide a hint if there is no information
				if (getStates().empty()) {
					out << "No thread states recorded.";
					if (!MONITORING_ENABLED) {
						out << " You can enable it by setting the MONITORING_ENABLED flag in the code base.";
					}
					out << "\n";
					return;
				}

				// print all current states
				for(const auto& cur : getStates()) {
					cur->dumpState(out);
				}
			}

		private:

			static std::mutex& getStateLock() {
				static std::mutex state_lock;
				return state_lock;
			}

			static std::vector<ThreadState*>& getStates() {
				static std::vector<ThreadState*> states;
				return states;
			}

		};

		thread_local static ThreadState tl_thread_state;


		struct Action {

			bool active;
			Event e;

			Action() : active(false) {}

			Action(const Event& e) : active(true), e(e) {
				// register action
				tl_thread_state.pushEvent(e);
			}

			Action(Action&& other) : active(other.active), e(other.e) {
				other.active = false;
			}

			Action(const Action&) = delete;

			Action& operator=(const Action&) = delete;
			Action& operator=(Action&&) = delete;

			~Action() {
				if (!active) return;
				// remove action from action stack
				tl_thread_state.popEvent(e);
			}

		};

		inline Action log(EventType type, const TaskBase* task) {
			assert_true(type != EventType::DependencyWait);
			if (!MONITORING_ENABLED) return {};
			return Event{type,task,TaskID()};
		}

		inline Action log(EventType type, const TaskID& task) {
			assert_true(type == EventType::DependencyWait);
			if (!MONITORING_ENABLED) return {};
			return Event{type,nullptr,task};
		}

	}




	// ---------------------------------------------------------------------------------------------
	//								 	Task Dependency Manager
	// ---------------------------------------------------------------------------------------------

	template<std::size_t max_depth>
	class TaskDependencyManager {

		// dependencies are stored in a linked list
		struct Entry {
			TaskBase* task;
			Entry* next;
		};

		using cell_type = std::atomic<Entry*>;

		enum { num_entries = 1<<(max_depth+1) };

		// an epoch counter to facilitate re-use
		std::atomic<std::size_t> epoch;

		// the container for storing task dependencies, pointer tagging is used to test for completeness
		cell_type data[num_entries];

	public:

		TaskDependencyManager(std::size_t epoch = 0) : epoch(epoch) {
			for(auto& cur : data) cur = nullptr;
		}

		~TaskDependencyManager() {
			for(auto& cur : data) {
				if (!isDone(cur)) delete cur;
			}
		}

		TaskDependencyManager(const TaskDependencyManager&) = delete;
		TaskDependencyManager(TaskDependencyManager&&) = delete;

		TaskDependencyManager& operator=(const TaskDependencyManager&) = delete;
		TaskDependencyManager& operator=(TaskDependencyManager&&) = delete;

		std::size_t getEpoch() const {
			return epoch.load();
		}

		void startEpoch(std::size_t newEpoch) {
			// make sure there is a change
			assert_ne(epoch.load(),newEpoch);

			// re-set state
			epoch = newEpoch;
			for(auto& cur : data) {
				// there should not be any dependencies left
				assert_true(cur == nullptr || isDone(cur));

				// reset dependencies
				cur = nullptr;
			}
		}


		/**
		 * Adds a dependency between the given tasks such that
		 * task x depends on the completion of the task y.
		 */
		void addDependency(TaskBase* x, const TaskPath& y);

		void markComplete(const TaskPath& task);

		bool isComplete(const TaskPath& path) const {
			return isDone(data[getPosition(path)]);
		}

	private:

		std::size_t getPosition(const TaskPath& path) const {

			// get length and path
			auto l = path.getLength();
			auto p = path.getPath();

			// limit length to max_depth
			if (l > max_depth) {
				p = p >> (l - max_depth);	// effective path
				l = max_depth;				// effective depth
			}

			// compute result
			return (1 << l) | p;
		}

		bool isDone(const Entry* ptr) const {
			// if the last bit is set, the task already finished
			return (intptr_t)(ptr) & 0x1;
		}

	};



	// ---------------------------------------------------------------------------------------------
	//											 Task Family
	// ---------------------------------------------------------------------------------------------


	/**
	 * A task family is a collection of tasks descending from a common (single) ancestor.
	 * Task families are created by root-level prec operator calls, and manage the dependencies
	 * of all its members.
	 *
	 * Tasks being created through recursive or combine calls are initially not members of
	 * any family, but may get adapted (by being the result of a split operation).
	 */
	class TaskFamily {

		// TODO: make task dependency manager depth target system dependent

		using DependencyManager = TaskDependencyManager<6>;

		// the manager of all dependencies on members of this family
		DependencyManager dependencies;

		// a flag determining whether this is a top-level task family
		// (it is not created nested by a treeture but by the main thread)
		bool top_level;

	public:

		/**
		 * Creates a new family, using a new ID.
		 */
		TaskFamily(bool top_level = false) : dependencies(getNextID()), top_level(top_level) {}

		/**
		 * Obtain the family ID.
		 */
		std::size_t getId() const {
			return dependencies.getEpoch();
		}

		/**
		 * Tests whether this task family is a top-level family (not nested).
		 */
		bool isTopLevel() const {
			return top_level;
		}

		/**
		 * Tests whether the given sub-task is complete.
		 */
		bool isComplete(const TaskPath& path) const {
			return dependencies.isComplete(path);
		}

		/**
		 * Register a dependency ensuring that a task x is depending on a task y.
		 */
		void addDependency(TaskBase* x, const TaskPath& y) {
			dependencies.addDependency(x,y);
		}

		/**
		 * Mark the given task as being finished.
		 */
		void markDone(const TaskPath& x) {
			dependencies.markComplete(x);
		}

		/**
		 * A family ID generator.
		 */
		static unsigned getNextID() {
			static std::atomic<int> counter(0);
			return ++counter;
		}

	};


	// the pointer type to reference task families
	using TaskFamilyPtr = TaskFamily*;

	/**
	 * A manager keeping track of created families.
	 */
	class TaskFamilyManager {

		SpinLock lock;

		std::vector<std::unique_ptr<TaskFamily>> families;

	public:

		TaskFamilyPtr getFreshFamily(bool topLevel) {
			std::lock_guard<SpinLock> lease(lock);

			// TODO: replace this by a re-use based solution

			// gradually drain old family references
			/*
			if (families.size() > 20000) {
				families.erase(families.begin(),families.begin() + families.size()/2);
			}
			*/

			// create a new family
			families.push_back(std::make_unique<TaskFamily>(topLevel));
			return families.back().get();
		}

	};


	// a factory for a new task family
	inline TaskFamilyPtr createFamily(bool topLevel = false) {
		static TaskFamilyManager familyManager;
		return familyManager.getFreshFamily(topLevel);
	}



	// ---------------------------------------------------------------------------------------------
	//										task reference
	// ---------------------------------------------------------------------------------------------


	/**
	 * A reference to a task utilized for managing task synchronization. Tasks may
	 * only be synchronized on if they are members of a task family.
	 */
	class task_reference {

		// a weak reference to a task's family
		TaskFamilyPtr family;

		TaskPath path;

		task_reference(const TaskFamilyPtr& family, const TaskPath& path)
			: family(family), path(path) {}

	public:

		task_reference() : family(nullptr), path(TaskPath::root()) {}

		task_reference(const TaskBase& task);

		task_reference(const task_reference&) = default;

		task_reference(task_reference&& other) : family(other.family), path(other.path) {
			other.family = nullptr;
		}

		task_reference& operator=(const task_reference& other) = default;

		task_reference& operator=(task_reference&& other) {
			family = other.family;
			path = other.path;
			other.family = nullptr;
			return *this;
		}

		bool isDone() const {
			return (!family || family->isComplete(path));
		}

		bool valid() const {
			return family;
		}

		void wait() const;

		task_reference getLeft() const {
			return task_reference ( family, path.getLeftChildPath() );
		}

		task_reference getRight() const {
			return task_reference ( family, path.getRightChildPath() );
		}

		task_reference& descentLeft() {
			path.descentLeft();
			return *this;
		}

		task_reference& descentRight() {
			path.descentRight();
			return *this;
		}

		// -- implementation details --

		TaskFamilyPtr getFamily() const {
			return family;
		}

		const TaskPath& getPath() const {
			return path;
		}

	};


	template<std::size_t size>
	struct fixed_sized {};

	struct dynamic_sized {};

	/**
	 * A class to aggregate task dependencies.
	 */
	template<typename size>
	class dependencies;

	/**
	 * A specialization for empty task dependencies.
	 */
	template<>
	class dependencies<fixed_sized<0>> {

	public:

		bool empty() const {
			return true;
		}

		std::size_t size() const {
			return 0;
		}

		const task_reference* begin() const {
			return nullptr;
		}

		const task_reference* end() const {
			return nullptr;
		}

	};


	/**
	 * A specialization for fixed-sized task dependencies.
	 */
	template<std::size_t Size>
	class dependencies<fixed_sized<Size>> {

		template<std::size_t A, std::size_t B>
		friend dependencies<fixed_sized<A+B>> concat(const dependencies<fixed_sized<A>>&, const dependencies<fixed_sized<B>>&);

		std::array<task_reference,Size> list;

	public:

		template<typename ... Args>
		dependencies(const Args& ... args) : list({{args...}}) {}

		dependencies(const dependencies&) = default;
		dependencies(dependencies&&) = default;

		dependencies& operator=(const dependencies&) = default;
		dependencies& operator=(dependencies&&) = default;

		bool empty() const {
			return Size == 0;
		}

		std::size_t size() const {
			return Size;
		}

		const task_reference* begin() const {
			return &(list[0]);
		}

		const task_reference* end() const {
			return begin()+Size;
		}

	};

	/**
	 * Enables the concatentation of two fixed-sized dependencies lists.
	 */
	template<std::size_t A, std::size_t B>
	dependencies<fixed_sized<A+B>> concat(const dependencies<fixed_sized<A>>& a, const dependencies<fixed_sized<B>>& b) {
		dependencies<fixed_sized<A+B>> res;
		for(std::size_t i=0; i<A; i++) {
			res.list[i] = a.list[i];
		}
		for(std::size_t i=0; i<B; i++) {
			res.list[A+i] = b.list[i];
		}
		return res;
	}

	/**
	 * A specialization for dynamically sized task dependencies.
	 */
	template<>
	class dependencies<dynamic_sized> {

		using list_type = std::vector<task_reference>;

		list_type* list;

	public:

		dependencies() : list(nullptr) {}

		dependencies(std::vector<task_reference>&& deps)
			: list(new list_type(std::move(deps))) {}

		dependencies(const dependencies&) = delete;

		dependencies(dependencies&& other) : list(other.list){
			other.list = nullptr;
		}

		~dependencies() {
			delete list;
		}

		dependencies& operator=(const dependencies&) = delete;

		dependencies& operator=(dependencies&& other) {
			if (list == other.list) return *this;
			delete list;
			list = other.list;
			other.list = nullptr;
			return *this;
		}

		bool empty() const {
			return list == nullptr;
		}

		std::size_t size() const {
			return (list) ? list->size() : 0;
		}

		void add(const task_reference& ref) {
			if (!list) list = new list_type();
			list->push_back(ref);
		}

		const task_reference* begin() const {
			return (list) ? &list->front() : nullptr;
		}

		const task_reference* end() const {
			return (list) ? (&list->back()) + 1 : nullptr;
		}

	};


	// ---------------------------------------------------------------------------------------------
	//										      promise
	// ---------------------------------------------------------------------------------------------


	/**
	 * A promise, forming the connection between a task and a treeture
	 * waiting for the task's result.
	 */
	template<typename T>
	class Promise {

		// a marker for delivered values
		std::atomic<bool> ready;

		// the delivered value
		T value;

	public:

		Promise() : ready(false) {}

		Promise(const T& value)
			: ready(true), value(value) {}

		bool isReady() const {
			return ready;
		}

		const T& getValue() const {
			return value;
		}

		void setValue(const T& newValue) {
			value = newValue;
			ready = true;
		}
	};

	/**
	 * A specialization for void promises.
	 */
	template<>
	class Promise<void> {

		// a marker for delivered promises
		std::atomic<bool> ready;

	public:

		Promise(bool ready = false)
			: ready(ready) {}

		bool isReady() const {
			return ready;
		}

		void setReady() {
			ready = true;
		}

	};


	template<typename T>
	using PromisePtr = std::shared_ptr<Promise<T>>;


	// ---------------------------------------------------------------------------------------------
	//											  Tasks
	// ---------------------------------------------------------------------------------------------


	// the RT's interface to a task
	class TaskBase {

	public:

		enum class State {
			New,          // < this task has been created, but not processed by a worker yet
			Blocked,      // < this task has unfinished dependencies
			Ready,        // < this task may be processed (scheduled in work queues)
			Running,      // < this task is running
			Aggregating,  // < this split task is aggregating results (skipped if not split)
			Done          // < this task is completed
		};

		friend std::ostream& operator<<(std::ostream& out, const State& state) {
			switch(state) {
				case State::New:           return out << "New";
				case State::Blocked:       return out << "Blocked";
				case State::Ready:         return out << "Ready";
				case State::Running:       return out << "Running";
				case State::Aggregating:   return out << "Aggregating";
				case State::Done:          return out << "Done";
			}
			return out << "Invalid";
		}

	private:

		// the family this task belongs to, if null, this task is an orphan task.
		TaskFamilyPtr family;

		// the position of this task within its family
		TaskPath path;

		// A cached version of the task ID. This id
		// is only valid if this task is not an orphan
		TaskID id;

		// the current state of this task
		std::atomic<State> state;

		/**
		 * the number of active dependencies keeping this object alive and
		 * blocking its execution. Those dependencies include
		 *   +1 for the unreleased treeture, subtracted once the task is released
		 *   +1 for the parent, released once the parent is no longer interested in this task
		 *   +1 for each task this task is waiting for, thus for each dependency
		 *
		 * Initially, there are 2 dependencies -- one for the parent, one for the release.
		 *
		 * Actions:
		 *   1 ... this task is started
		 *   0 ... this task is destroyed
		 */
		std::atomic<int> num_active_dependencies;

		// indicates whether this task can be split
		bool splitable;

		// split task data
		TaskBase* left;
		TaskBase* right;

		// for the mutation from a simple to a split task
		TaskBase* substitute;

		// TODO: get rid of this
		bool parallel;

		// for the processing of split tasks
		TaskBase* parent;                      // < a pointer to the parent to be notified upon completion
		std::atomic<int> alive_child_counter;  // < the number of active child tasks

		// a flag to remember that this task got a substitute, even after the
		// substitute got cut lose
		std::atomic<bool> substituted;

	public:

		TaskBase(bool done = false)
			: family(), path(TaskPath::root()), id(TaskFamily::getNextID()),
			  state(done ? State::Done : State::New),
			  // one initial control flow dependency, released by treeture release
			  num_active_dependencies(done ? 1 : 2),
			  splitable(false),
			  left(nullptr), right(nullptr), substitute(nullptr),
			  parallel(false), parent(nullptr),
			  substituted(false) {

			LOG_TASKS( "Created " << *this );

			// register this task
			if (MONITORING_ENABLED) registerTask(*this);
		}

		TaskBase(TaskBase* left, TaskBase* right, bool parallel)
			: family(),
			  path(TaskPath::root()), id(TaskFamily::getNextID()),
			  state(State::New),
			  // one initial control flow dependency, released by treeture release
			  num_active_dependencies(2),
			  splitable(false),
			  left(left), right(right), substitute(nullptr),
			  parallel(parallel),
			  parent(nullptr), alive_child_counter(0),
			  substituted(false) {

			LOG_TASKS( "Created " << *this );
			assert(this->left);
			assert(this->right);

			// fix the parent pointer
			this->left->parent = this;
			this->right->parent = this;

			// register this task
			if (MONITORING_ENABLED) registerTask(*this);
		}

	protected:

		// make the destructor private, such that only this class can destroy itself
		virtual ~TaskBase() {
			if (MONITORING_ENABLED) unregisterTask(*this);
			LOG_TASKS( "Destroying Task " << *this );
			assert_true(isDone()) << getId() << " - " << getState();
		};

	public:

		// -- observers --

		const TaskFamilyPtr& getTaskFamily() const {
			return family;
		}

		const TaskPath& getTaskPath() const {
			return path;
		}

		TaskID getId() const {
			return id;
		}

		bool isOrphan() const {
			return !family;
		}

		std::size_t getDepth() const {
			return path.getLength();
		}

		State getState() const {
			// the substitute takes over the control of the state
			if (substitute) return substitute->state;
			return state;
		}

		// each implementation is required to provide a runtime predictor
		virtual RuntimePredictor& getRuntimePredictor() const = 0;

		// -- mutators --

		void addDependency(const task_reference& ref) {
			addDependencies(&ref,&ref+1);
		}

		template<typename Iter>
		void addDependencies(const Iter& begin, const Iter& end) {

			// ignore empty dependencies
			if (begin == end) return;

			// we must still be in the new state
			assert_eq(getState(),State::New);

			// this task must not yet be started nor must the parent be lost
			assert_le(2,num_active_dependencies);

			// increase the number of active dependencies
			num_active_dependencies += (int)(end - begin);

			// register dependencies
			for(auto it = begin; it != end; ++it) {
				const auto& cur = *it;

				// filter out already completed tasks (some may be orphans)
				if (cur.isDone()) {
					// notify that one dependency more is completed
					dependencyDone();
					// continue with next
					continue;
				}

				// add dependency
				assert_true(cur.getFamily());
				cur.getFamily()->addDependency(this,cur.getPath());
			}

		}

		void adopt(const TaskFamilyPtr& family, const TaskPath& path = TaskPath()) {
			// check that this task is not member of another family
			assert_true(isOrphan()) << "Can not adopt a member of another family.";

			// check whether there is an actual family
			if (!family) return;

			// join the family
			this->family = family;
			this->path = path;

			// update the id
			this->id = TaskID(family->getId(),path);

			// mark as complete, if already complete
			if(isDone()) family->markDone(path);

			// propagate adoption to descendants
			if (substitute) substitute->adopt(family,path);
			if (left)  left->adopt(family, path.getLeftChildPath());
			if (right) right->adopt(family, path.getRightChildPath());
		}


		// -- state transitions --

		// New -> Blocked
		void start();

		// Blocked -> Ready transition is triggered by the last dependency

		// Ready -> Running - finish() ->  Done
		void run() {

			// log this event
			auto action = monitoring::log(monitoring::EventType::Run, this);

			// process substituted tasks
			if (substituted) {
				// there is nothing to do
				return;
			}


			LOG_TASKS( "Running Task " << *this );

			// check that it is allowed to run
			assert_eq(state, State::Ready);
			assert_eq(1,num_active_dependencies);

			// update state
			setState(State::Running);

			// process split tasks
			if (isSplit()) {					// if there is a left, it is a split task

				// check some assumptions
				assert(left && right);

				State lState = left->state;
				State rState = right->state;

				assert(lState == State::New || lState == State::Done);
				assert(rState == State::New || rState == State::Done);

				// run task sequentially if requested
				if (!parallel) {

					// TODO: implement sequential execution dependency based
					alive_child_counter = 2;

					// process left first
					if (lState != State::Done) {
						left->start();
					} else {
						// notify that this child is done
						childDone(*left);
					}

					// right child is started by childDone once left is finished

					// done
					return;

				}

				// count number of sub-tasks to be started
				assert_eq(0,alive_child_counter);

				// check which child tasks need to be started
				if (lState == State::New && rState == State::New) {

					// both need to be started
					alive_child_counter = 2;
					left->start();
					right->start();

				} else if (lState == State::New) {

					// only left has to be started
					alive_child_counter = 1;
					left->start();

				} else if (rState == State::New) {

					// only left has to be started
					alive_child_counter = 1;
					right->start();

				} else {

					// perform reduction immediately since sub-tasks are done
					finish();

					// done
					return;
				}

				// processing complete

			} else {

				// run computation
				execute();

				// finish task
				finish();

			}
		}

		// Ready -> Split (if supported, otherwise remains Ready)
		virtual bool split() {
			// by default, no splitting is supported
			assert_fail() << "This should not be reachable!";
			return false;
		}

		// wait for the task completion
		void wait();

		bool isDone() const {
			// simply check the state of this task
			return state == State::Done;
		}

		const TaskBase* getLeft() const {
			// forward call to substitute if present
			if (substitute) return substitute->getLeft();
			return left;
		}

		const TaskBase* getRight() const {
			// forward call to substitute if present
			if (substitute) return substitute->getRight();
			return right;
		}

		bool isSplitable() const {
			return splitable;
		}

		bool isSplit() const {
			return (bool)left;
		}

		bool isSubstituted() const {
			return substituted;
		}

		bool isReady() const {
			if (substitute) return substitute->isReady();
			return state == State::Ready;
		}

		void dependencyDone();

	protected:

		/**
		 * A hook to define the operations to be conducted by this
		 * task instance. This function will only be triggered
		 * for non-split tasks.
		 */
		virtual void execute() =0;

		/**
		 * A hook to define post-operation operations triggered after
		 * the completion of this task or the completion of its child
		 * tasks. It should be utilized to retrieve results from
		 * substitutes or child tasks and aggregate those.
		 */
		virtual void aggregate() =0;

		void setSplitable(bool value = true) {
			splitable = value && getDepth() < 60;
		}

		void setSubstitute(TaskBase* newSub) {

			// must only be set once!
			assert_false(substitute);

			// can only happen if this task is in blocked or ready state
			assert_true(state == State::Blocked || state == State::Ready)
					<< "Actual state: " << state;

			// and the substitute must be valid
			assert_true(newSub);

			// the substitute must be new
			assert_true(newSub->state == State::New || newSub->state == State::Done);

			// link substitute -- with this responsibilities are transfered
			substitute = newSub;

			// connect substitute to parent
			substitute->parent = this;

			// remember that a substitute has been assigned
			substituted = true;

			// if the split task is done, this one is done
			if (substitute->isDone()) {

				// update state
				if (state == State::Blocked) setState(State::Ready);

				// pass through running
				setState(State::Running);

				// finish this task
				finish();

				// done
				return;
			}

			// adapt substitute
			substitute->adopt(this->family, this->path);

			// and update this state to ready
			if (state == State::Blocked) setState(State::Ready);

			// since the substitute may be processed any time, this may finish
			// any time => thus it is in the running state
			setState(State::Running);

			// start the substitute
			substitute->start();

		}

	private:

		bool isValidTransition(State from, State to) {
			return (from == State::New         && to == State::Blocked     ) ||
				   (from == State::Blocked     && to == State::Ready       ) ||
				   (from == State::Ready       && to == State::Running     ) ||
				   (from == State::Running     && to == State::Aggregating ) ||
				   (from == State::Aggregating && to == State::Done        ) ;
		}

		void setState(State newState) {

			// check correctness of state transitions
			assert_true(isValidTransition(state,newState))
				<< "Illegal state transition from " << state << " to " << newState;

			// make sure that the task is not released with active dependencies
			assert_true(newState != State::Ready || num_active_dependencies == 1 || substituted)
				<< "Active dependencies: " << num_active_dependencies;

			// update the state
			state = newState;
			LOG_TASKS( "Updated state: " << *this );
		}

		void childDone(const TaskBase& child) {

			// this task must not be done yet
			assert_ne(state,State::Done);

			// check whether it is the substitute
			if (substitute == &child) {

				// check state of this task
				assert_true(State::Ready == state || State::Running == state)
					<< "Actual state: " << state;

				// log state change
				LOG_TASKS( "Substitute " << *substitute << " of " << *this << " done");

				// trigger completion of task
				finish();
				return;
			}

			// make sure this task is still running
			assert_eq(State::Running, state)
				<< "\tis substitute:  " << (substitute == &child) << "\n"
				<< "\tis child left:  " << (left == &child) << "\n"
				<< "\tis child right: " << (right == &child) << "\n";

			// process a split-child
			LOG_TASKS( "Child " << child << " of " << *this << " done" );

			// if this is a sequential node, start next child
			if (!parallel && &child == left) {

				// continue with the right child
				if (right->getState() != State::Done) {
					right->start();
				} else {
					// notify that the right child is also done
					childDone(*right);
				}

			}

			// decrement active child count
			unsigned old_child_count = alive_child_counter.fetch_sub(1);

			// log alive counter
			LOG_TASKS( "Child " << child << " of " << *this << " -- alive left: " << (old_child_count - 1) );

			// check whether this was the last child
			if (old_child_count != 1) return;

			// the last child finished => finish this task
			finish();

			// LOG_TASKS( "Child " << child << " of " << *this << " done - processing complete" );
		}

		void parentDone() {

			// check that there is a parent
			assert_true(parent);

			// signal that one more dependency is satisfied
			dependencyDone();

		}

		// Running -> Aggregating -> Done
		void finish() {

			LOG_TASKS( "Finishing task " << *this );

			// check precondition
			assert_true(state == State::Running)
					<< "Actual State: " << state << "\nTask: " << *this;


			// update state to aggregation
			setState(State::Aggregating);

			// log aggregation step
			LOG( "Aggregating task " << *this );

			// aggregate result (collect results)
			aggregate();

			// a tool to release dependent tasks
			auto release = [](TaskBase* task) {
				assert_true(!task || task->isDone());
				if (!task) return;
				task->parentDone();
			};

			// cut lose children
			release(left);
			release(right);

			// cut lose substitutes
			release(substitute);

			// log completion
			LOG( "Aggregating task " << *this << " complete" );

			// job is done
			setState(State::Done);

			// copy parent pointer to stack, since the markDone may release this task
			TaskBase* locParent = parent;

			// inform the family that the job is done
			if (!parent || parent->substitute != this) {
				// only due this if you are not the substitute
				if (family) family->markDone(path);

				// if there is no parent, don't wait for it to signal its release
				if (!parent) dependencyDone();
			}

			// notify parent
			if (locParent) {

				// notify parents
				parent->childDone(*this);

			}

		}

		// -- support printing of tasks for debugging --

		friend std::ostream& operator<<(std::ostream& out, const TaskBase& task) {

			// if substituted, print the task and its substitute
			if (task.substitute) {
				out << task.getId() << " -> " << *task.substitute;
				return out;
			}

			// if split, print the task and its children
			if (task.isSplit()) {
				out << task.getId() << " : " << task.state;
				if (task.state == State::Done) return out;

				out << " = " << (task.parallel ? "parallel" : "sequential") << " [";
				if (task.left) out << *task.left; else out << "nil";
				out << ",";
				if (task.right) out << *task.right; else out << "nil";
				out << "]";
				return out;
			}

			// in all other cases, just print the id
			out << task.getId() << " : " << task.state;

			// get the total number of dependencies
			std::size_t numDependencies = task.num_active_dependencies;

			// remove release dependency
			if (task.state == State::New) numDependencies -= 1;

			// remove delete dependency
			numDependencies -= 1;

			// print number of task dependencies
			if (task.state <= State::Blocked) {
				out << " waiting for " << numDependencies << " task(s)";
			}

			return out;
		}

		template<typename Process, typename Split, typename R>
		friend class SplitableTask;

		// --- debugging ---

	private:

		static std::mutex& getTaskRegisterLock() {
			static std::mutex lock;
			return lock;
		}

		static std::set<const TaskBase*>& getTaskRegister() {
			static std::set<const TaskBase*> instances;
			return instances;
		}

		static void registerTask(const TaskBase& task) {
			std::lock_guard<std::mutex> g(getTaskRegisterLock());
			getTaskRegister().insert(&task);
		}

		static void unregisterTask(const TaskBase& task) {
			std::lock_guard<std::mutex> g(getTaskRegisterLock());
			auto pos = getTaskRegister().find(&task);
			assert_true(pos!=getTaskRegister().end());
			getTaskRegister().erase(pos);
		}

	public:

		static void dumpAllTasks(std::ostream& out) {
			std::lock_guard<std::mutex> g(getTaskRegisterLock());

			// check whether monitoring is enabled
			if (!MONITORING_ENABLED) {
				out << " -- task tracking disabled, enable by setting MONITORING_ENABLED to true --\n";
				return;
			}

			// list active tasks
			std::cout << "List of all tasks:\n";
			for(const auto& cur : getTaskRegister()) {
				std::cout << "\t" << *cur << "\n";
			}
		}

	};


	// ----------- Task Dependency Manager Implementations ---------------

	template<std::size_t max_depth>
	void TaskDependencyManager<max_depth>::addDependency(TaskBase* x, const TaskPath& y) {

		// locate entry
		std::size_t pos = getPosition(y);

		// load epoch
		auto curEpoch = epoch.load();

		// load the head
		Entry* head = data[pos].load();

		// check whether we are still in the same epoch
		if (curEpoch != epoch.load()) {
			// the epoch has changed, the previous is gone
			x->dependencyDone();
			return;
		}

		// check whether this task is already completed
		if (isDone(head)) {
			// signal that this dependency is done
			x->dependencyDone();
			return;
		}

		// insert element
		Entry* entry = new Entry();
		entry->task = x;
		entry->next = head;

		// update entry pointer lock-free
		while (!data[pos].compare_exchange_weak(entry->next,entry)) {

			// check whether the task has been completed in the meanwhile
			if (isDone(entry->next)) {
				delete entry;
				// signal that this dependency is done
				x->dependencyDone();
				return;
			}

			// otherwise, repeat until it worked
		}

		// successfully inserted
	}

	template<std::size_t max_depth>
	void TaskDependencyManager<max_depth>::markComplete(const TaskPath& task) {

		// ignore tasks that are too small
		if (task.getLength() > max_depth) return;

		// mark as complete and obtain head of depending list
		auto pos = getPosition(task);
		Entry* cur = data[pos].exchange((Entry*)0x1);

		// do not process list twice (may be called multiple times due to substitutes)
		if (isDone(cur)) return;

		// signal the completion of this task
		while(cur) {

			// signal a completed dependency
			cur->task->dependencyDone();

			// move on to next entry
			Entry* next = cur->next;
			delete cur;
			cur = next;
		}

		// and its children
		if (pos >= num_entries/2) return;
		markComplete(task.getLeftChildPath());
		markComplete(task.getRightChildPath());
	}

	// -------------------------------------------------------------------



	// ------------------------- Task Reference --------------------------

	inline task_reference::task_reference(const TaskBase& task)
		: family(task.getTaskFamily()), path(task.getTaskPath()) {
		assert_false(task.isOrphan()) << "Unable to reference an orphan task!";
	}

	// -------------------------------------------------------------------


	// a task computing a value of type T
	template<typename T>
	class Task : public TaskBase {

		T value;

		mutable PromisePtr<T> promise;

	public:

		Task() : TaskBase(), promise(nullptr) {}

		Task(const T& value)
			: TaskBase(true), value(value), promise(nullptr) {}

		Task(TaskBase* left, TaskBase* right, bool parallel)
			: TaskBase(left, right, parallel), promise(nullptr) {}


		virtual ~Task(){};

		const T& getValue() const {
			assert_true(isDone()) << this->getState();
			return value;
		}

		void setPromise(const PromisePtr<T>& newPromise) const {

			// this task must not be started yet
			assert_eq(State::New,this->getState());

			// there must not be a previous promise
			assert_false(promise);

			// register promise
			promise = newPromise;
		}

	protected:

		void execute() override {
			value = computeValue();
		}

		void aggregate() override {
			value = computeAggregate();
			if(promise) {
				promise->setValue(value);
			}
		}

		virtual T computeValue() {
			// the default does nothing
			return value;
		};

		virtual T computeAggregate() {
			// nothing to do by default
			return value;
		};

		virtual RuntimePredictor& getRuntimePredictor() const override {
			assert_fail() << "Should not be reachable, predictions only intresting for splitable tasks!";
			return reference::getRuntimePredictor<void>();
		}
	};

	template<>
	class Task<void> : public TaskBase {

		mutable PromisePtr<void> promise;

	public:

		Task(bool done = false) : TaskBase(done) {}

		Task(TaskBase* left, TaskBase* right, bool parallel)
			: TaskBase(left,right,parallel) {}

		virtual ~Task(){};

		void getValue() const {
		}

		void setPromise(const PromisePtr<void>& newPromise) const {

			// this task must not be started yet
			assert_eq(State::New,this->getState());

			// there must not be a previous promise
			assert_false(promise);

			// register promise
			promise = newPromise;
		}

	protected:

		void execute() override {
			computeValue();
		}

		void aggregate() override {
			computeAggregate();
			if(promise) {
				promise->setReady();
			}
		}

		virtual void computeValue() {};

		virtual void computeAggregate() {};

		virtual RuntimePredictor& getRuntimePredictor() const override {
			assert_fail() << "Should not be reachable, predictions only intresting for splitable tasks!";
			return reference::getRuntimePredictor<void>();
		}
	};


	template<
		typename Process,
		typename R = std::result_of_t<Process()>
	>
	class SimpleTask : public Task<R> {

		Process task;

	public:

		SimpleTask(const Process& task)
			: Task<R>(), task(task) {}

		R computeValue() override {
			return task();
		}

		virtual RuntimePredictor& getRuntimePredictor() const override {
			return reference::getRuntimePredictor<Process>();
		}

	};


	template<
		typename Process,
		typename Split,
		typename R = std::result_of_t<Process()>
	>
	class SplitableTask : public Task<R> {

		Process task;
		Split decompose;

		Task<R>* subTask;

	public:

		SplitableTask(const Process& c, const Split& d)
			: Task<R>(), task(c), decompose(d), subTask(nullptr) {
			// mark this task as one that can be split
			TaskBase::setSplitable();
		}

		R computeValue() override {
			// this should not be called if split
			assert_false(subTask);
			return task();
		}

		R computeAggregate() override {
			// the aggregated value depends on whether it was split or not
			return (subTask) ? subTask->getValue() : Task<R>::computeAggregate();
		}

		bool split() override;

		virtual RuntimePredictor& getRuntimePredictor() const override {
			return reference::getRuntimePredictor<Process>();
		}

	};

	template<typename R, typename A, typename B, typename C>
	class SplitTask : public Task<R> {

		const Task<A>& left;
		const Task<B>& right;

		C merge;

	public:

		SplitTask(Task<A>* left, Task<B>* right, C&& merge, bool parallel)
			: Task<R>(left,right,parallel),
			  left(*left),
			  right(*right),
			  merge(merge) {}


		R computeValue() override {
			// should not be reached
			assert_fail() << "Should always be split!";
			return {};
		}

		R computeAggregate() override {
			return merge(left.getValue(),right.getValue());
		}

		virtual RuntimePredictor& getRuntimePredictor() const override {
			assert_fail() << "Should not be reachable, predictions only intresting for splitable tasks!";
			return reference::getRuntimePredictor<void>();
		}
	};

	template<typename A, typename B>
	class SplitTask<void,A,B,void> : public Task<void> {
	public:

		SplitTask(TaskBase* left, TaskBase* right, bool parallel)
			: Task<void>(left,right,parallel) {}

		void computeValue() override {
			// should not be reached
			assert_fail() << "Should always be split!";
		}

		void computeAggregate() override {
			// nothing to do
		}

		virtual RuntimePredictor& getRuntimePredictor() const override {
			assert_fail() << "Should not be reachable, predictions only intresting for splitable tasks!";
			return reference::getRuntimePredictor<void>();
		}
	};

	template<typename Deps, typename A, typename B, typename C, typename R = std::result_of_t<C(A,B)>>
	Task<R>* make_split_task(Deps&& deps, Task<A>* left, Task<B>* right, C&& merge, bool parallel) {
		Task<R>* res = new SplitTask<R,A,B,C>(left, right, std::move(merge), parallel);
		res->addDependencies(deps.begin(), deps.end());
		return res;
	}

	template<typename Deps>
	Task<void>* make_split_task(Deps&& deps, TaskBase* left, TaskBase* right, bool parallel) {
		Task<void>* res = new SplitTask<void,void,void,void>(left, right, parallel);
		res->addDependencies(deps.begin(), deps.end());
		return res;
	}





	// ---------------------------------------------------------------------------------------------
	//											Treetures
	// ---------------------------------------------------------------------------------------------


	namespace detail {

		/**
		 * A common base class for all treetures, providing common functionality.
		 */
		template<typename T>
		class treeture_base {

			template<typename Process, typename Split, typename R>
			friend class SplitableTask;

		protected:

			task_reference taskRef;

			PromisePtr<T> promise;

			treeture_base() : promise() {}

			treeture_base(const Task<T>& task) : promise(std::make_shared<Promise<T>>()) {

				// make sure task has not been started yet
				assert_eq(TaskBase::State::New, task.getState());

				// register the promise
				task.setPromise(promise);

				// also create task reference if available
				if (!task.isOrphan()) {
					taskRef = task_reference(task);
				}
			}

			treeture_base(PromisePtr<T>&& promise)
				: promise(std::move(promise)) {

				// make sure the promise is valid and set
				assert_true(this->promise);
				assert_true(this->promise->isReady());

			}

		public:

			using value_type = T;

			treeture_base(const treeture_base&) = delete;
			treeture_base(treeture_base&& other) = default;

			treeture_base& operator=(const treeture_base&) = delete;
			treeture_base& operator=(treeture_base&& other) = default;

			void wait() const;

			bool isDone() const {
				return !promise || promise->isReady();
			}

			bool isValid() const {
				return (bool)promise;
			}

			task_reference getLeft() const {
				return getTaskReference().getLeft();
			}

			task_reference getRight() const {
				return getTaskReference().getRight();
			}

			task_reference getTaskReference() const {
				return taskRef;
			}

			operator task_reference() const {
				return getTaskReference();
			}

		};

	}

	/**
	 * A treeture, providing a reference to the state of a task as well as to
	 * the computed value upon completion.
	 */
	template<typename T>
	class treeture : public detail::treeture_base<T> {

		using super = detail::treeture_base<T>;

		friend class unreleased_treeture<T>;

	protected:

		treeture(const Task<T>& task) : super(task) {}

	public:

		using treeture_type = treeture<T>;

		treeture() {}

		treeture(const T& value)
			: super(std::make_shared<Promise<T>>(value)) {}

		treeture(const treeture&) = delete;
		treeture(treeture&& other) = default;

		treeture& operator=(const treeture&) = delete;
		treeture& operator=(treeture&& other) = default;

		const T& get() {
			static const T defaultValue = T();
			if (!this->promise) return defaultValue;
			super::wait();
			return this->promise->getValue();
		}

	};

	/**
	 * A specialization of the general value treeture for the void type, exhibiting
	 * a modified signature for the get() member function.
	 */
	template<>
	class treeture<void> : public detail::treeture_base<void> {

		using super = detail::treeture_base<void>;

		friend class unreleased_treeture<void>;

	protected:

		treeture(const Task<void>& task) : super(task) {}

	public:

		treeture() : super() {}

		treeture(const treeture&) = delete;
		treeture(treeture&& other) = default;

		treeture& operator=(const treeture&) = delete;
		treeture& operator=(treeture&& other) = default;

		void get() {
			wait();
		}

	};



	template<typename Process, typename Split, typename R>
	bool SplitableTask<Process,Split,R>::split() {
		// do not split a second time
		if (!TaskBase::isSplitable()) return false;

		assert_true(TaskBase::State::Blocked == this->state || TaskBase::State::Ready == this->state)
				<< "Actual state: " << this->state;

		// decompose this task
		Task<R>* substitute = decompose().toTask();
		assert_true(substitute);
		assert_true(substitute->state == TaskBase::State::New || substitute->state == TaskBase::State::Done);

		// record reference to sub-task
		subTask = substitute;

		// mark as no longer splitable
		TaskBase::setSplitable(false);

		// mutate to new task
		Task<R>::setSubstitute(substitute);

		// done
		return true;
	}



	// ---------------------------------------------------------------------------------------------
	//										 Unreleased Treetures
	// ---------------------------------------------------------------------------------------------

	namespace detail {

		template<typename T>
		struct done_task_to_treeture {
			treeture<T> operator()(const Task<T>& task) {
				return treeture<T>(task.getValue());
			}
		};

		template<>
		struct done_task_to_treeture<void> {
			treeture<void> operator()(const Task<void>&) {
				return treeture<void>();
			}
		};
	}


	/**
	 * A handle to a yet unreleased task.
	 */
	template<typename T>
	class unreleased_treeture {

		Task<T>* task;

	public:

		using value_type = T;

		using treeture_type = treeture<T>;

		unreleased_treeture(Task<T>* task)
			: task(task) {}

		unreleased_treeture(const unreleased_treeture&) =delete;

		unreleased_treeture(unreleased_treeture&& other) : task(other.task) {
			other.task = nullptr;
		}

		unreleased_treeture& operator=(const unreleased_treeture&) =delete;

		unreleased_treeture& operator=(unreleased_treeture&& other) {
			std::swap(task,other.task);
			return *this;
		}

		~unreleased_treeture() {
			if(task) {
				assert_fail()
						<< "Did you forget to release a treeture?";
			}
		}

		treeture<T> release() && {

			// there has to be a task
			assert_true(task);

			// special case for completed tasks
			if (task->isDone()) {
				auto res = detail::done_task_to_treeture<T>()(*task);
				task->dependencyDone();	// remove one dependency for the lose of the owner
				task = nullptr;
				return res;
			}

			// the referenced task has not been released yet
			assert_eq(TaskBase::State::New,task->getState());

			// create the resulting treeture
			treeture<T> res(*task);

			// start the task -- the actual release
			task->start();

			// reset the task pointer
			task = nullptr;

			// return the resulting treeture
			return res;
		}

		operator treeture<T>() && {
			return std::move(*this).release();
		}

		T get() && {
			return std::move(*this).release().get();
		}

		Task<T>* toTask() && {
			auto res = task;
			task = nullptr;
			return res;
		}

	};



	// ---------------------------------------------------------------------------------------------
	//										   Operators
	// ---------------------------------------------------------------------------------------------



	inline dependencies<fixed_sized<0>> after() {
		return dependencies<fixed_sized<0>>();
	}

	template<typename ... Rest>
	auto after(const task_reference& r, const Rest& ... rest) {
		return dependencies<fixed_sized<1+sizeof...(Rest)>>(r,rest...);
	}

	inline dependencies<dynamic_sized> after(std::vector<task_reference>&& refs) {
		return std::move(refs);
	}


	template<typename DepsKind>
	unreleased_treeture<void> done(dependencies<DepsKind>&& deps) {
		auto res = new Task<void>(true);
		res->addDependencies(deps.begin(),deps.end());
		return res;
	}

	inline unreleased_treeture<void> done() {
		return done(after());
	}

	template<typename DepsKind, typename T>
	unreleased_treeture<T> done(dependencies<DepsKind>&& deps, const T& value) {
		auto res = new Task<T>(value);
		res->addDependencies(deps.begin(),deps.end());
		return res;
	}

	template<typename T>
	unreleased_treeture<T> done(const T& value) {
		return done(after(),value);
	}

	namespace runtime {

		// determines whether this thread is running in a nested context
		bool isNestedContext();

	}

	namespace detail {

		template<bool root, typename Deps, typename T>
		unreleased_treeture<T> init(Deps&& deps, Task<T>* task) {

			// add dependencies
			task->addDependencies(deps.begin(),deps.end());

			// create task family if requested
			if (root) {
				task->adopt(createFamily(!runtime::isNestedContext()));
			}

			// done
			return task;
		}

	}


	template<bool root, typename DepsKind, typename Action, typename T = std::result_of_t<Action()>>
	unreleased_treeture<T> spawn(dependencies<DepsKind>&& deps, Action&& op) {
		// create and initialize the task
		return detail::init<root>(std::move(deps), (Task<T>*)(new SimpleTask<Action>(std::move(op))));
	}

	template<bool root, typename Action>
	auto spawn(Action&& op) {
		return spawn<root>(after(),std::move(op));
	}

	template<bool root, typename Deps, typename Action, typename Split, typename T = std::result_of_t<Action()>>
	unreleased_treeture<T> spawn(Deps&& deps, Action&& op, Split&& split) {
		// create and initialize the task
		return detail::init<root>(std::move(deps), (Task<T>*)(new SplitableTask<Action,Split>(std::move(op),std::move(split))));
	}

	template<bool root, typename Action, typename Split>
	auto spawn(Action&& op, Split&& split) {
		return spawn<root>(after(),std::move(op),std::move(split));
	}

	template<typename Deps>
	unreleased_treeture<void> seq(Deps&& deps) {
		return done(std::move(deps));
	}

	inline unreleased_treeture<void> seq() {
		return done();
	}

	template<typename DepsKind, typename A, typename B>
	unreleased_treeture<void> seq(dependencies<DepsKind>&& deps, unreleased_treeture<A>&& a, unreleased_treeture<B>&& b) {
		return make_split_task(std::move(deps),std::move(a).toTask(),std::move(b).toTask(),false);
	}

	template<typename A, typename B>
	unreleased_treeture<void> seq(unreleased_treeture<A>&& a, unreleased_treeture<B>&& b) {
		return seq(after(),std::move(a),std::move(b));
	}

	template<typename DepsKind, typename F, typename ... R>
	unreleased_treeture<void> seq(dependencies<DepsKind>&& deps, unreleased_treeture<F>&& f, unreleased_treeture<R>&& ... rest) {
		// TODO: conduct a binary split to create a balanced tree
		return make_split_task(std::move(deps),std::move(f).toTask(),seq(std::move(rest)...).toTask(),false);
	}

	template<typename F, typename ... R>
	unreleased_treeture<void> seq(unreleased_treeture<F>&& f, unreleased_treeture<R>&& ... rest) {
		return seq(after(), std::move(f),std::move(rest)...);
	}

	template<typename Deps>
	unreleased_treeture<void> par(Deps&& deps) {
		return done(std::move(deps));
	}

	inline unreleased_treeture<void> par() {
		return done();
	}

	template<typename DepsKind, typename A, typename B>
	unreleased_treeture<void> par(dependencies<DepsKind>&& deps, unreleased_treeture<A>&& a, unreleased_treeture<B>&& b) {
		return make_split_task(std::move(deps),std::move(a).toTask(),std::move(b).toTask(),true);
	}

	template<typename A, typename B>
	unreleased_treeture<void> par(unreleased_treeture<A>&& a, unreleased_treeture<B>&& b) {
		return par(after(),std::move(a),std::move(b));
	}

	template<typename DepsKind, typename F, typename ... R>
	unreleased_treeture<void> par(dependencies<DepsKind>&& deps, unreleased_treeture<F>&& f, unreleased_treeture<R>&& ... rest) {
		// TODO: conduct a binary split to create a balanced tree
		return make_split_task(std::move(deps),std::move(f).toTask(),par(std::move(deps),std::move(rest)...).toTask(),true);
	}

	template<typename F, typename ... R>
	unreleased_treeture<void> par(unreleased_treeture<F>&& f, unreleased_treeture<R>&& ... rest) {
		return par(after(), std::move(f),std::move(rest)...);
	}



	template<typename DepsKind, typename A, typename B, typename M, typename R = std::result_of_t<M(A,B)>>
	unreleased_treeture<R> combine(dependencies<DepsKind>&& deps, unreleased_treeture<A>&& a, unreleased_treeture<B>&& b, M&& m, bool parallel = true) {
		return make_split_task(std::move(deps),std::move(a).toTask(),std::move(b).toTask(),std::move(m),parallel);
	}

	template<typename A, typename B, typename M, typename R = std::result_of_t<M(A,B)>>
	unreleased_treeture<R> combine(unreleased_treeture<A>&& a, unreleased_treeture<B>&& b, M&& m, bool parallel = true) {
		return reference::combine(after(),std::move(a),std::move(b),std::move(m),parallel);
	}


	// ---------------------------------------------------------------------------------------------
	//											Runtime
	// ---------------------------------------------------------------------------------------------

	namespace runtime {



		// -----------------------------------------------------------------
		//						    Worker Pool
		// -----------------------------------------------------------------

		class Worker;

		thread_local static Worker* tl_worker = nullptr;

		static void setCurrentWorker(Worker& worker) {
			tl_worker = &worker;
		}

		static Worker& getCurrentWorker();

		namespace detail {

			/**
			 * A utility to fix the affinity of the current thread to the given core.
			 * Does not do anything on operating systems other than linux.
			 */
			#ifdef __linux__
				inline void fixAffinity(int core) {
					// fix affinity if user does not object
					if(std::getenv("NO_AFFINITY") == nullptr) {
						int num_cores = std::thread::hardware_concurrency();
						cpu_set_t mask;
						CPU_ZERO(&mask);
						CPU_SET(core % num_cores, &mask);
						pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
					}
				}
			#else
				inline void fixAffinity(int) { }
			#endif

		}

		class WorkerPool;


		class Worker {

			using duration = RuntimePredictor::duration;

			// the targeted maximum queue length
			// (more like a guideline, may be exceeded due to high demand)
			enum { max_queue_length = 8 };

			WorkerPool& pool;

			volatile bool alive;

			// list of tasks ready to run
			OptimisticUnboundQueue<TaskBase*> queue;

			std::thread thread;

			unsigned id;

			// the list of workers to attempt to steel from, in order
			std::vector<Worker*> stealingOrder;

		public:

			Worker(WorkerPool& pool, unsigned id)
				: pool(pool), alive(true), id(id) { }

			Worker(const Worker&) = delete;
			Worker(Worker&&) = delete;

			Worker& operator=(const Worker&) = delete;
			Worker& operator=(Worker&&) = delete;

			void start() {
				thread = std::thread([&](){ run(); });
			}

			void poison() {
				alive = false;
			}

			void join() {
				thread.join();
			}

			void dumpState(std::ostream& out) const {
				out << "Worker " << id << " / " << thread.get_id() << ":\n";
				out << "\tQueue:\n";
				for(const auto& cur : queue.getSnapshot()) {
					out << "\t\t" << *cur << "\n";
				}
			}

		private:

			void run();

			void runTask(TaskBase& task);

			bool splitTask(TaskBase& task);

			duration estimateRuntime(const TaskBase& task) {
				return task.getRuntimePredictor().predictTime(task.getDepth());
			}

		public:

			void schedule(TaskBase& task);

			bool schedule_step();

		};

		class WorkerPool {

			std::vector<Worker*> workers;

			// tools for managing idle threads
			std::mutex m;
			std::condition_variable cv;

		public:

			WorkerPool() {

				int numWorkers = std::thread::hardware_concurrency();

				// parse environment variable
				if (char* val = std::getenv("NUM_WORKERS")) {
					auto userDef = std::atoi(val);
					if (userDef != 0) numWorkers = userDef;
				}

				// there must be at least one worker
				if (numWorkers < 1) numWorkers = 1;

				// create workers
				for(int i=0; i<numWorkers; ++i) {
					workers.push_back(new Worker(*this,i));
				}

				// start additional workers (worker 0 is main thread)
				for(int i=1; i<numWorkers; ++i) {
					workers[i]->start();
				}

				// make worker 0 being linked to the main thread
				setCurrentWorker(*workers.front());

				// fix affinity of main thread
				detail::fixAffinity(0);

				// fix worker id of main thread
				setCurrentWorkerID(0);

			}

			~WorkerPool() {
				// shutdown threads

				{
					// poison all workers
					std::lock_guard<std::mutex> guard(m);
					for(auto& cur : workers) {
						cur->poison();
					}

					// make work available
					workAvailable();
				}

				// wait for their death
				for(std::size_t i=1; i<workers.size(); ++i) {
					workers[i]->join();
				}

				// free resources
				for(auto& cur : workers) {
					delete cur;
				}

			}

			static WorkerPool& getInstance() {
				static WorkerPool pool;
				return pool;
			}

			int getNumWorkers() const {
				return (int)workers.size();
			}

		private:

			mutable std::size_t initialLimit = std::numeric_limits<std::size_t>::max();

		public:

			std::size_t getInitialSplitDepthLimit() const {
			    if (initialLimit == std::numeric_limits<std::size_t>::max()) {
			        std::size_t i = 0;
	                auto num_workers = getNumWorkers();
                    while ((1<<i) < (num_workers * 2)) {
                        i++;
                    }
                    initialLimit = i;
			    }
			    return initialLimit;
			}

			Worker& getWorker(int i) {
				assert_le(0,i);
				assert_lt(i,(int)workers.size());
				return *workers[i];
			}

			const std::vector<Worker*>& getWorkers() const {
				return workers;
			}

			Worker& getWorker() {
				return getWorker(0);
			}

			void dumpState(std::ostream& out) {
				for(const auto& cur : workers) {
					cur->dumpState(out);
				}
			}

		protected:

			friend Worker;

			void waitForWork(volatile bool& alive) {
				std::unique_lock<std::mutex> lk(m);
				if (!alive) return;
				LOG_SCHEDULE("Going to sleep");
				cv.wait(lk);
				LOG_SCHEDULE("Woken up again");
			}

			void workAvailable() {
				// wake up all workers
				cv.notify_all();
			}

		};

		static Worker& getCurrentWorker() {
			if (tl_worker) return *tl_worker;
			return WorkerPool::getInstance().getWorker();
		}

		inline void Worker::run() {

			// fix worker ID
			setCurrentWorkerID(id);

			// copy worker list
			auto allWorkers = pool.getWorkers();

			// a utility to add new steel targets
			auto addStealTarget = [&](std::size_t idx) {
				if (idx == id) return;
				stealingOrder.push_back(allWorkers[idx]);
			};

			// create list of workers to steel from
			auto numWorkers = allWorkers.size();
			for(std::size_t d=1; d<numWorkers; ++d) {
				addStealTarget((id + d) % numWorkers);
				addStealTarget((id - d + numWorkers) % numWorkers);
			}

			// log creation of worker event
			logProfilerEvent(ProfileLogEntry::createWorkerCreatedEntry());

			// fix affinity
			detail::fixAffinity(id);

			// register worker
			setCurrentWorker(*this);

			// start processing loop
			while(alive) {

				// count number of idle cycles
				int idle_cycles = 0;

				// conduct a schedule step
				while(alive && !schedule_step()) {
					// increment idle counter
					++idle_cycles;

					// wait a moment
					cpu_relax();

					// if there was no work for quite some time
					if(idle_cycles > 100000) {

						// report sleep event
						logProfilerEvent(ProfileLogEntry::createWorkerSuspendedEntry());

						// wait for work by putting thread to sleep
						pool.waitForWork(alive);

						// report awakening
						logProfilerEvent(ProfileLogEntry::createWorkerResumedEntry());

						// reset cycles counter
						idle_cycles = 0;
					}
				}
			}

			// log worker termination event
			logProfilerEvent(ProfileLogEntry::createWorkerDestroyedEntry());

			// done

		}

		inline bool& getIsNestedFlag() {
			static thread_local bool nested = false;
			return nested;
		}

		inline bool isNestedContext() {
			return getIsNestedFlag();
		}

		inline void Worker::runTask(TaskBase& task) {

			// the splitting of a task may provide a done substitute => skip those
			if (task.isDone()) return;

			LOG_SCHEDULE("Starting task " << task);

			// no substituted task may be processed
			assert_false(task.isSubstituted());

			// make sure this is a ready task
			assert_eq(TaskBase::State::Ready,task.getState());

			// mark as nested
			bool& nestedContextFlag = getIsNestedFlag();
			bool old = nestedContextFlag;
			nestedContextFlag = true;

			// process the task
			if (task.isSplit()) {
				task.run();
			} else {

				__allscale_unused auto taskId = task.getId();
				logProfilerEvent(ProfileLogEntry::createTaskStartedEntry(taskId));

				// check whether this run needs to be sampled
				auto level = task.getDepth();
				if (level == 0) {

					// level 0 does not need to be recorded (orphans)
					task.run();

				} else {

					// get predictor before task by be gone (as part of the processing)
					RuntimePredictor& predictor = task.getRuntimePredictor();

					// take the time to make predictions
					auto start = RuntimePredictor::clock::now();
					task.run();
					auto time = RuntimePredictor::clock::now() - start;

					predictor.registerTime(level,time);

				}

				logProfilerEvent(ProfileLogEntry::createTaskEndedEntry(taskId));

			}

			// reset old nested context state
			nestedContextFlag = old;

			LOG_SCHEDULE("Finished task " << task);
		}

		inline bool Worker::splitTask(TaskBase& task) {
			using namespace std::chrono_literals;

			// the threshold for estimated task to be split
			static const auto taskTimeThreshold = CycleCount(3*1000*1000);

			// only split the task if it is estimated to exceed a threshold
			if (task.isSplitable() && (task.getDepth() == 0 || estimateRuntime(task) > taskTimeThreshold)) {

				// split this task
				return task.split();

			}

			// no split happend
			return false;
		}

		inline void Worker::schedule(TaskBase& task) {

			// assert that task has no unfinished dependencies
			assert_true(task.isReady());

			// no task that is substituted shall be scheduled
			assert_false(task.isSubstituted());


			// actively distribute initial tasks, by assigning them to different workers

			// TODO: do the following only for top-level tasks!!

			if (!task.isOrphan() && task.getTaskFamily()->isTopLevel()) {

				// get the limit for initial decomposition
				auto split_limit = pool.getInitialSplitDepthLimit();

				// if below this limit, split the task
				if (task.isSplitable() && task.getDepth() < split_limit) {

					// if splitting worked => we are done
					if (task.split()) return;

				}

				// the depth limit for task being actively distributed
				auto distribution_limit = split_limit + 2;

				// actively distribute tasks throughout the pool
				if (task.getDepth() < distribution_limit) {

					// actively select the worker to issue the task to
					std::size_t num_workers = pool.getNumWorkers();
					auto path = task.getTaskPath().getPath();
					auto depth = task.getDepth();

					auto trgWorker = (depth==0) ? 0 : (path * num_workers) / ((uint64_t)1 << depth);

					// check the computation of the target worker
					assert_lt(trgWorker,(std::size_t)pool.getNumWorkers())
						<< "Error in target worker computation:\n"
						<< "\tNumWorkers: " << num_workers << "\n"
						<< "\tPath:       " << path << "\n"
						<< "\tDepth:      " << depth << "\n"
						<< "\tTarget:     " << trgWorker << "\n";


					// if the target is another worker => send the task there
					if (trgWorker != id) {

						// submit this task to the selected worker
						pool.getWorker((int)trgWorker).schedule(task);

						// done
						return;

					}
				}
			}

			// add task to queue
			LOG_SCHEDULE( "Queue size before: " << queue.size() );

			// no task that is substituted shall be scheduled
			assert_false(task.isSubstituted());

			// add task to queue
			queue.push_back(&task);

			// signal available work
			pool.workAvailable();

			// log new queue length
			LOG_SCHEDULE( "Queue size after: " << queue.size() );

		}


		inline bool Worker::schedule_step() {

			// process a task from the local queue
			if (TaskBase* t = queue.pop_front()) {

				// the task should not have a substitute
				assert_false(t->isSubstituted());

				// check precondition of task
				assert_true(t->isReady()) << "Actual state: " << t->getState();

				// if the queue is not full => create more tasks
				if (queue.size() < (max_queue_length*3)/4) {

					LOG_SCHEDULE( "Splitting tasks @ queue size: " << queue.size() );



					// split task and be done
					if (splitTask(*t)) return true;

					// the task should not have a substitute
					assert_false(t->isSubstituted());

				}

				// process this task
				runTask(*t);
				return true;
			}

			// look through potential targets to steel a task
			for(const auto& cur : stealingOrder) {

				// otherwise, steal a task from another worker
				Worker& other = *cur;

				// try to steal a task from another queue
				if (TaskBase* t = other.queue.try_pop_back()) {

					// the task should not have a substitute
					assert_false(t->isSubstituted());

					// log creation of worker event
					logProfilerEvent(ProfileLogEntry::createTaskStolenEntry(t->getId()));

					LOG_SCHEDULE( "Stolen task: " << t );

					// split task the task (since there is not enough work in the queue)
					if (splitTask(*t)) return true;

					// the task should not have a substitute
					assert_false(t->isSubstituted());

					// process task
					runTask(*t);
					return true;	// successfully completed a task
				}

			}

			// no task found => wait a moment
			cpu_relax();

			// report back the failed steal attempt
			return false;
		}

	}

	namespace monitoring {

		inline std::ostream& operator<<(std::ostream& out, const Event& e) {
			switch(e.type) {
			case EventType::Run:    		return out << "Running task            " << *e.task;
			case EventType::RunDirect:  	return out << "Running direct task     " << *e.task;
			case EventType::Split:  		return out << "Splitting task          " << *e.task;
			case EventType::Wait:   		return out << "Waiting for task        " << *e.task;
			case EventType::DependencyWait: return out << "Waiting for dependency: " << e.taskId;
			}
			return out << "Unknown Event";
		}

	}// end namespace monitoring


	inline void TaskBase::start() {
		LOG_TASKS("Starting " << *this );

		// check that the given task is a new task
		assert_eq(TaskBase::State::New, state);

		// move to next state
		setState(State::Blocked);

		// if below the initial split limit, split this task
		if (!isOrphan() && getTaskFamily()->isTopLevel() && isSplitable() && getDepth() < runtime::WorkerPool::getInstance().getInitialSplitDepthLimit()) {

			// attempt to split this task
			split();

		}

		// release dummy-dependency to get task started
		dependencyDone();
	}

	inline void TaskBase::dependencyDone() {

		// keep a backup in case the object is destroyed asynchronously
		auto substitutedLocalCopy = substituted.load();

		// decrease the number of active dependencies
		int oldValue = num_active_dependencies.fetch_sub(1);

		// compute the new value
		int newValue = oldValue - 1;

		// make sure there are no releases that should not be
		assert_le(0,newValue);

		// if we are down to 0 => destroy this task
		if (newValue == 0) {

			// at this point this task must be done
			assert_eq(State::Done,state);

			// destroy this object, and be done
			delete this;
			return;
		}

		// if the new value is not 1 => ignore
		if (newValue != 1) return;

		// if the value is 1, we release this task for computation
		assert_eq(1,newValue);

		// handle substituted instances by ignoring the message
		if (substitutedLocalCopy || substituted) return;

		// make sure that at this point there is still a parent left
		assert_eq(num_active_dependencies, 1);

		// at this point the state must not be new
		assert_ne(State::New, state)
			<< "A new task must not reach a state where its last dependency is released.";

		// actually, every task here must be in blocked state
		assert_eq(State::Blocked, state) << *this << "\t" << substitutedLocalCopy << "\n";

		// update the state to ready
		// (this can only be reached by one thread)
		setState(State::Ready);

		// schedule task
		runtime::getCurrentWorker().schedule(*this);

	}

	inline void TaskBase::wait() {
		// log this event
		// auto action = monitoring::log(monitoring::EventType::Wait, this);

		LOG_TASKS("Waiting for " << *this );

		// check that this task has been started before
		assert_lt(State::New,state);

		// wait until this task is finished
		while(!isDone()) {
			// make some progress
			runtime::getCurrentWorker().schedule_step();
		}
	}

	inline void task_reference::wait() const {
		// log this event
		// auto action = monitoring::log(monitoring::EventType::DependencyWait, TaskID(family->getId(),path));

		// wait until the referenced task is done
		while(!isDone()) {
			// but while doing so, do useful stuff
			runtime::getCurrentWorker().schedule_step();
		}
	}

	namespace detail {

		template<typename T>
		void treeture_base<T>::wait() const {
			// wait for completion
			while (promise && !promise->isReady()) {
				// make some progress
				runtime::getCurrentWorker().schedule_step();
			}
		}

	}

} // end namespace reference
} // end namespace impl
} // end namespace core
} // end namespace api
} // end namespace allscale


inline void __dumpRuntimeState() {
	std::cout << "\n ------------------------- Runtime State Dump -------------------------\n";
	allscale::api::core::impl::reference::monitoring::ThreadState::dumpStates(std::cout);
	allscale::api::core::impl::reference::runtime::WorkerPool::getInstance().dumpState(std::cout);
	allscale::api::core::impl::reference::TaskBase::dumpAllTasks(std::cout);
	std::cout << "\n ----------------------------------------------------------------------\n";
}
