#pragma once

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "command.h"
#include "types.h"

namespace celerity {
namespace detail {

	class logger;

	// TODO: Could be extended (using SFINAE) to support additional iterator types (e.g. random access)
	template <typename Iterator, typename PredicateFn>
	class filter_iterator {
	  public:
		using value_type = typename std::iterator_traits<Iterator>::value_type;
		using difference_type = typename std::iterator_traits<Iterator>::difference_type;
		using reference = typename std::iterator_traits<Iterator>::reference;
		using pointer = typename std::iterator_traits<Iterator>::pointer;
		using iterator_category = std::forward_iterator_tag;

		filter_iterator(Iterator begin, Iterator end, PredicateFn fn) : it(begin), end(end), fn(fn) { advance(); }

		bool operator!=(const filter_iterator& rhs) { return it != rhs.it; }

		reference operator*() { return *it; }
		reference operator->() { return *it; }

		filter_iterator& operator++() {
			if(it != end) {
				++it;
				advance();
			}
			return *this;
		}

	  private:
		Iterator it;
		const Iterator end;
		PredicateFn fn;

		void advance() {
			while(it != end && !fn(*it)) {
				++it;
			}
		}
	};

	template <typename Iterator, typename PredicateFn>
	filter_iterator<Iterator, PredicateFn> make_filter_iterator(Iterator begin, Iterator end, PredicateFn fn) {
		return filter_iterator<Iterator, PredicateFn>(begin, end, fn);
	}

	// TODO: Could be extended (using SFINAE) to support additional iterator types (e.g. random access)
	template <typename Iterator, typename TransformFn>
	class transform_iterator {
	  public:
		using value_type = decltype(std::declval<TransformFn>()(std::declval<typename std::iterator_traits<Iterator>::reference>()));
		using difference_type = typename std::iterator_traits<Iterator>::difference_type;
		using reference = value_type; // We cannot return a reference (but this is OK according to the standard)
		using pointer = value_type*;
		using iterator_category = std::forward_iterator_tag;

		transform_iterator(Iterator it, TransformFn fn) : it(it), fn(fn) {}

		bool operator!=(const transform_iterator& rhs) { return it != rhs.it; }

		reference operator*() { return fn(*it); }
		reference operator->() { return fn(*it); }

		transform_iterator& operator++() {
			++it;
			return *this;
		}

	  private:
		Iterator it;
		TransformFn fn;
	};

	template <typename Iterator, typename TransformFn>
	transform_iterator<Iterator, TransformFn> make_transform_iterator(Iterator it, TransformFn fn) {
		return transform_iterator<Iterator, TransformFn>(it, fn);
	}

	class command_graph {
	  public:
		void record_command(nop_command*) {}
		void record_command(push_command*) {}
		void record_command(await_push_command*) {}
		void record_command(execution_command* tcmd) { by_task[tcmd->get_tid()].emplace_back(tcmd); }
		void record_command(horizon_command* hcmd) {
			by_task[hcmd->get_tid()].emplace_back(hcmd);
			active_horizons.emplace_back(hcmd);
		}
		void record_command(reduction_command*) {}

		template <typename T, typename... Args>
		T* create(Args... args) {
			static_assert(std::is_base_of<abstract_command, T>::value, "T must be derived from abstract_command");
			command_id cid = next_cmd_id++;
			auto result = commands.emplace(std::make_pair(cid, new T(cid, std::forward<Args>(args)...)));
			auto cmd = result.first->second.get();
			if(!std::is_same<T, nop_command>::value) execution_fronts[cmd->get_nid()].insert(cmd);
			auto ret = static_cast<T*>(cmd);
			record_command(ret);
			return ret;
		}

		void erase(abstract_command* cmd);

		void erase_if(std::function<bool(abstract_command*)> condition);

		bool has(command_id cid) const { return commands.count(cid) == 1; }

		abstract_command* get(command_id cid) { return commands.at(cid).get(); }

		template <typename T>
		T* get(command_id cid) {
			// dynamic_cast with reference to force bad_cast to be thrown if type mismatches
			return &dynamic_cast<T&>(*commands.at(cid));
		}

		size_t command_count() const { return commands.size(); }
		size_t task_command_count(task_id tid) const { return by_task.at(tid).size(); }

		auto all_commands() const {
			const auto transform = [](auto& uptr) { return uptr.second.get(); };
			return iterable_range{make_transform_iterator(commands.cbegin(), transform), make_transform_iterator(commands.cend(), transform)};
		}

		auto& task_commands(task_id tid) { return by_task.at(tid); }

		void print_graph(logger& graph_logger) const;

		// TODO unify dependency terminology to this
		void add_dependency(abstract_command* depender, abstract_command* dependee, dependency_kind kind = dependency_kind::TRUE_DEP) {
			assert(depender->get_nid() == dependee->get_nid()); // We cannot depend on commands executed on another node!
			assert(dependee != depender);
			depender->add_dependency({dependee, kind});
			execution_fronts[depender->get_nid()].erase(dependee);
			max_pseudo_critical_path_length = std::max(max_pseudo_critical_path_length, depender->get_pseudo_critical_path_length());
		}

		void remove_dependency(abstract_command* depender, abstract_command* dependee) { depender->remove_dependency(dependee); }

		const std::unordered_set<abstract_command*>& get_execution_front(node_id nid) const { return execution_fronts.at(nid); }

		int get_max_pseudo_critical_path_length() const { return max_pseudo_critical_path_length; }

		std::vector<horizon_command*>& get_active_horizons() { return active_horizons; }

	  private:
		command_id next_cmd_id = 0;
		// TODO: Consider storing commands in a contiguous memory data structure instead
		std::unordered_map<command_id, std::unique_ptr<abstract_command>> commands;
		std::unordered_map<task_id, std::vector<task_command*>> by_task;

		// Set of per-node commands with no dependents
		std::unordered_map<node_id, std::unordered_set<abstract_command*>> execution_fronts;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		int max_pseudo_critical_path_length = 0;

		// Active horizons (created but not flushed)
		std::vector<horizon_command*> active_horizons;
	};

} // namespace detail
} // namespace celerity
