#pragma once

#include <algorithm>
#include <iterator>
#include <ostream>
#include <tuple>

#include <bitset>
#include <cstring>

#include "allscale/utils/assert.h"
#include "allscale/utils/bitmanipulation.h"
#include "allscale/utils/io_utils.h"
#include "allscale/utils/range.h"
#include "allscale/utils/raw_buffer.h"
#include "allscale/utils/serializer.h"
#include "allscale/utils/static_map.h"
#include "allscale/utils/table.h"
#include "allscale/utils/array_utils.h"
#include "allscale/utils/tuple_utils.h"

#include "allscale/utils/printer/vectors.h"

#include "allscale/api/core/data.h"
#include "allscale/api/core/prec.h"

namespace allscale {
namespace api {
namespace user {
namespace data {


	// --------------------------------------------------------------------
	//							  Declarations
	// --------------------------------------------------------------------



	// --- mesh type parameter constructs ---

	/**
	 * The base type of edges connecting nodes of kind A with nodes of kind B
	 * on the same level.
	 */
	template<typename A, typename B>
	struct edge {
		using src_node_kind = A;
		using trg_node_kind = B;
	};


	/**
	 * The base type of edges connecting nodes of kind A with nodes of kind B
	 * on adjacent levels.
	 */
	template<typename A, typename B>
	struct hierarchy {
		using parent_node_kind = A;
		using child_node_kind = B;
	};

	/**
	 * The constructor for the list of node kinds to be included in a mesh structure.
	 */
	template<typename ... Nodes>
	struct nodes {
		enum { size = sizeof...(Nodes) };
	};

	/**
	 * The constructor for the list of edge kinds to be included in a mesh structure.
	 */
	template<typename ... Edges>
	struct edges {
		enum { size = sizeof...(Edges) };
	};

	/**
	 * The constructor for the list of hierarchies to be included in a mesh structure.
	 */
	template<typename ... Hierarchies>
	struct hierarchies {
		enum { size = sizeof...(Hierarchies) };
	};


	// --- mesh type parameter constructs ---


	/**
	 * The type used for addressing nodes within meshes.
	 */
	template<typename Kind, unsigned Level = 0>
	struct NodeRef;

	/**
	 * The type used for iterating over lists of nodes, e.g. a list of adjacent nodes.
	 */
	template<typename Kind,unsigned Level>
	using NodeList = utils::range<const NodeRef<Kind,Level>*>;


	/**
	 * The type for representing the topological information of a hierarchical mesh.
	 */
	template<
		typename NodeKinds,						// < list of node types in each level
		typename EdgeKinds,						// < list of edge types connecting nodes within levels
		typename Hierarchies = hierarchies<>,	// < list of edge types connecting nodes between adjacent levels
		unsigned Levels = 1,					// < number of levels in the hierarchy
		unsigned PartitionDepth = 0				// < number of partitioning level
	>
	class Mesh;


	/**
	 * The type for associating (dynamic) information to nodes within a mesh.
	 */
	template<
		typename NodeKind,				// < the type of node to be annotated
		typename ElementType,			// < the type of value to be associated to each node on the given level
		unsigned Level,					// < the level of the mesh to be annotated
		typename PartitionTree			// < the type of the partition tree indexing the associated mesh
	>
	class MeshData;


	/**
	 * A utility to construct meshes.
	 */
	template<
		typename NodeKinds,						// < list of node types in each level
		typename EdgeKinds,						// < list of edge types connecting nodes within levels
		typename Hierarchies = hierarchies<>,	// < list of edge types connecting nodes between adjacent levels
		unsigned Levels = 1						// < number of levels in the hierarchy
	>
	class MeshBuilder;


	// -- mesh attributes --

	/**
	 * The base type for mesh property kinds.
	 */
	template<typename NodeKind, typename ValueType>
	struct mesh_property {
		using node_kind = NodeKind;
		using value_type = ValueType;
	};

	/**
	 * A container for a collection of mesh properties. A mesh property is
	 * a value associated to a certain kind of node on each level of a mesh.
	 * The MeshProperties container allows multiple properties to be managed
	 * within a single, consistent entity.
	 *
	 * To create an instance, the factory function "createProperties" of
	 * the Mesh structure has to be utilized.
	 */
	template<unsigned Levels, typename PartitionTree, typename ... Properties>
	class MeshProperties;



	// --------------------------------------------------------------------
	//							  Definitions
	// --------------------------------------------------------------------

	// The type used for indexing nodes in meshes
	using node_index_t = uint64_t;

	// The type used for identifying nodes within meshes.
	struct NodeID {

		node_index_t id;

		NodeID() = default;

		constexpr explicit NodeID(node_index_t id) : id(id) {}

		operator node_index_t() const {
			return id;
		}

		node_index_t getOrdinal() const {
			return id;
		}

		bool operator==(const NodeID& other) const {
			return id == other.id;
		}

		bool operator!=(const NodeID& other) const {
			return id != other.id;
		}

		bool operator<(const NodeID& other) const {
			return id < other.id;
		}

		friend std::ostream& operator<<(std::ostream& out, const NodeID& ref) {
			return out << "n" << ref.id;
		}

	};

	/**
	 * The type used for addressing nodes within meshes.
	 */
	template<typename Kind,unsigned Level>
	struct NodeRef : public NodeID {

		using node_kind = Kind;

		enum { level = Level };

		NodeRef() = default;

		constexpr explicit NodeRef(node_index_t id)
			: NodeID(id) {}

		constexpr explicit NodeRef(NodeID id)
			: NodeID(id) {}

	};


	template<typename Kind, unsigned Level>
	class NodeRange {

		NodeRef<Kind,Level> _begin;

		NodeRef<Kind,Level> _end;

	public:

		NodeRange(const NodeRef<Kind,Level>& a, const NodeRef<Kind,Level>& b) : _begin(a), _end(b) {
			assert_le(_begin.id,_end.id);
		}

		NodeRange() : _begin(), _end() {}

		NodeRef<Kind,Level> getBegin() const {
			return _begin;
		}

		NodeRef<Kind,Level> getEnd() const {
			return _end;
		}

		NodeRef<Kind,Level> operator[](std::size_t index) const {
			return NodeRef<Kind,Level>(NodeID(_begin.id + (node_index_t)index));
		}

		std::size_t size() const {
			return _end.id - _begin.id;
		}


		class const_iterator : public std::iterator<std::random_access_iterator_tag, NodeRef<Kind,Level>> {

			node_index_t cur;

		public:

			const_iterator(NodeID pos) : cur(pos) {};

			bool operator==(const const_iterator& other) const {
				return cur == other.cur;
			}

			bool operator!=(const const_iterator& other) const {
				return !(*this == other);
			}

			bool operator<(const const_iterator& other) const {
				return cur < other.cur;
			}

			bool operator<=(const const_iterator& other) const {
				return cur <= other.cur;
			}

			bool operator>=(const const_iterator& other) const {
				return cur >= other.cur;
			}

			bool operator>(const const_iterator& other) const {
				return cur > other.cur;
			}

			NodeRef<Kind,Level> operator*() const {
				return NodeRef<Kind,Level>{cur};
			}

			const_iterator& operator++() {
				++cur;
				return *this;
			}

			const_iterator operator++(int) {
				const_iterator res = *this;
				++cur;
				return res;
			}

			const_iterator& operator--() {
				--cur;
				return *this;
			}

			const_iterator operator--(int) {
				const_iterator res = *this;
				--cur;
				return res;
			}

			const_iterator& operator+=(std::ptrdiff_t n) {
				cur += n;
				return *this;
			}

			const_iterator& operator-=(std::ptrdiff_t n) {
				cur -= n;
				return *this;
			}

			friend const_iterator operator+(const_iterator& iter, std::ptrdiff_t n) {
				const_iterator res = iter;
				res.cur += n;
				return res;

			}

			friend const_iterator& operator+(std::ptrdiff_t n, const_iterator& iter) {
				const_iterator res = iter;
				res.cur += n;
				return res;
			}

			const_iterator operator-(std::ptrdiff_t n) {
				const_iterator res = *this;
				res.cur -= n;
				return res;
			}

			std::ptrdiff_t operator-(const_iterator& other) const {
				return std::ptrdiff_t(cur - other.cur);
			}

			NodeRef<Kind,Level> operator[](std::ptrdiff_t n) const {
				return *(*this + n);
			}

		};

		const_iterator begin() const {
			return const_iterator(_begin);
		}

		const_iterator end() const {
			return const_iterator(_end);
		}

		template<typename Body>
		void forAll(const Body& body) {
			for(const auto& cur : *this) {
				body(cur);
			}
		}

		friend std::ostream& operator<<(std::ostream& out, const NodeRange& range) {
			return out << "[" << range._begin.id << "," << range._end.id << ")";
		}

	};


	namespace detail {

		template<typename List>
		struct is_nodes : public std::false_type {};

		template<typename ... Kinds>
		struct is_nodes<nodes<Kinds...>> : public std::true_type {};

		template<typename List>
		struct is_edges : public std::false_type {};

		template<typename ... Kinds>
		struct is_edges<edges<Kinds...>> : public std::true_type {};

		template<typename List>
		struct is_hierarchies : public std::false_type {};

		template<typename ... Kinds>
		struct is_hierarchies<hierarchies<Kinds...>> : public std::true_type {};

		template<unsigned Level>
		struct level {
			enum { value = Level };
		};


		template<typename T>
		struct get_level;

		template<unsigned L>
		struct get_level<level<L>> {
			enum { value = L };
		};

		template<typename T>
		struct get_level<T&> : public get_level<T> {};
		template<typename T>
		struct get_level<const T> : public get_level<T> {};
		template<typename T>
		struct get_level<volatile T> : public get_level<T> {};

		template<typename T>
		using plain_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;


		template<typename Element>
		void sumPrefixes(utils::Table<Element>& list) {
			Element counter = 0;
			for(auto& cur : list) {
				auto tmp = cur;
				cur = counter;
				counter += tmp;
			}
		}


		template<std::size_t Levels, typename ... NodeKinds>
		class NodeSet {

			using LevelData = utils::StaticMap<utils::keys<NodeKinds...>,std::size_t>;

			using DataStore = std::array<LevelData,Levels>;

			static_assert(std::is_trivial<DataStore>::value, "The implementation assumes that this type is trivial!");

			DataStore data;

		public:

			NodeSet() {
				for(auto& cur : data) cur = LevelData(0);
			}

			NodeSet(const NodeSet&) = default;
			NodeSet(NodeSet&& other) = default;

			NodeSet& operator=(const NodeSet&) =default;
			NodeSet& operator=(NodeSet&&) =default;


			// -- observers and mutators --

			template<typename NodeKind,std::size_t Level = 0>
			NodeRef<NodeKind,Level> create() {
				auto& node_counter = getNodeCounter<NodeKind,Level>();
				return NodeRef<NodeKind,Level>(node_counter++);
			}

			template<typename NodeKind,std::size_t Level = 0>
			NodeRange<NodeKind,Level> create(std::size_t num) {
				auto& node_counter = getNodeCounter<NodeKind,Level>();
				NodeRef<NodeKind,Level> begin((node_index_t)node_counter);
				node_counter += num;
				NodeRef<NodeKind,Level> end((node_index_t)node_counter);
				return { begin, end };
			}

			template<typename NodeKind,std::size_t Level = 0>
			std::size_t getNumNodes() const {
				return getNodeCounter<NodeKind,Level>();
			}

			// -- IO support --

			void store(std::ostream& out) const {
				// store the number of nodes
				utils::write<DataStore>(out, data);
			}

			static NodeSet load(std::istream& in) {

				// produce result
				NodeSet res;

				// restore the number of nodes
				res.data = utils::read<DataStore>(in);

				// done
				return res;
			}

			static NodeSet interpret(utils::RawBuffer& buffer) {

				// produce result
				NodeSet res;

				// restore the number of nodes
				res.data = buffer.consume<DataStore>();

				// done
				return res;

			}

		private:

			template<typename NodeKind, std::size_t Level>
			std::size_t& getNodeCounter() {
				return data[Level].template get<NodeKind>();
			}

			template<typename NodeKind, std::size_t Level>
			const std::size_t& getNodeCounter() const {
				return data[Level].template get<NodeKind>();
			}
		};


		template<std::size_t Levels, typename ... EdgeKinds>
		class EdgeSet {

			// -- the data stored per relation --
			class Relation {

				static_assert(
					sizeof(NodeRef<int,0>) == sizeof(NodeID),
					"For this implementation to be correct node references have to be simple node IDs."
				);

				utils::Table<uint64_t> forward_offsets;
				utils::Table<NodeID> forward_targets;

				utils::Table<uint64_t> backward_offsets;
				utils::Table<NodeID> backward_targets;

				std::vector<std::pair<NodeID,NodeID>> edges;

			public:

				template<typename EdgeKind, unsigned Level>
				NodeList<typename EdgeKind::trg_node_kind,Level> getSinks(const NodeRef<typename EdgeKind::src_node_kind,Level>& src) const {
					using List = NodeList<typename EdgeKind::trg_node_kind,Level>;
					using TrgNodeRef = NodeRef<typename EdgeKind::trg_node_kind,Level>;
					assert_true(isClosed()) << "Accessing non-closed edge set!";
					if (src.id+1 >= forward_offsets.size() || forward_targets.empty()) return List{nullptr,nullptr};
					return List{
						reinterpret_cast<const TrgNodeRef*>(&forward_targets[forward_offsets[src.id]]),
						reinterpret_cast<const TrgNodeRef*>(&forward_targets[forward_offsets[src.id+1]])
					};
				}

				template<typename EdgeKind, unsigned Level>
				NodeList<typename EdgeKind::src_node_kind,Level> getSources(const NodeRef<typename EdgeKind::trg_node_kind,Level>& src) const {
					using List = NodeList<typename EdgeKind::src_node_kind,Level>;
					using SrcNodeRef = NodeRef<typename EdgeKind::src_node_kind,Level>;
					assert_true(isClosed()) << "Accessing non-closed edge set!";
					if (src.id+1 >= backward_offsets.size() || backward_targets.empty()) return List{nullptr,nullptr};
					return List{
						reinterpret_cast<const SrcNodeRef*>(&backward_targets[backward_offsets[src.id]]),
						reinterpret_cast<const SrcNodeRef*>(&backward_targets[backward_offsets[src.id+1]])
					};
				}

				void addEdge(NodeID from, NodeID to) {
					edges.push_back({from,to});
				}

				bool isClosed() const {
					return edges.empty();
				}

				void close() {

					// get maximum source and target
					std::size_t maxSourceID = 0;
					std::size_t maxTargetID = 0;
					for(const auto& cur : edges) {
						maxSourceID = std::max<std::size_t>(maxSourceID,cur.first);
						maxTargetID = std::max<std::size_t>(maxTargetID,cur.second);
					}

					// init forward / backward vectors
					forward_offsets = utils::Table<uint64_t>(maxSourceID + 2, 0);
					forward_targets = utils::Table<NodeID>(edges.size());

					backward_offsets = utils::Table<uint64_t>(maxTargetID + 2,0);
					backward_targets = utils::Table<NodeID>(edges.size());

					// count number of sources / sinks
					for(const auto& cur : edges) {
						++forward_offsets[cur.first];
						++backward_offsets[cur.second];
					}

					// compute prefix sums
					sumPrefixes(forward_offsets);
					sumPrefixes(backward_offsets);

					// fill in targets
					auto forward_pos = forward_offsets;
					auto backward_pos = backward_offsets;
					for(const auto& cur : edges) {
						forward_targets[forward_pos[cur.first]++] = cur.second;
						backward_targets[backward_pos[cur.second]++] = cur.first;
					}

					// clear edges
					edges.clear();

				}

				void store(std::ostream& out) const {
					// only allow closed sets to be stored
					assert_true(isClosed());

					// write forward edge data
					forward_offsets.store(out);
					forward_targets.store(out);

					// write backward edge data
					backward_offsets.store(out);
					backward_targets.store(out);

				}

				static Relation load(std::istream& in) {

					Relation res;

					// restore edge data
					res.forward_offsets = utils::Table<uint64_t>::load(in);
					res.forward_targets = utils::Table<NodeID>::load(in);

					res.backward_offsets = utils::Table<uint64_t>::load(in);
					res.backward_targets = utils::Table<NodeID>::load(in);

					// done
					return res;
				}

				static Relation interpret(utils::RawBuffer& buffer) {

					Relation res;

					// restore edge data
					res.forward_offsets = utils::Table<uint64_t>::interpret(buffer);
					res.forward_targets = utils::Table<NodeID>::interpret(buffer);

					res.backward_offsets = utils::Table<uint64_t>::interpret(buffer);
					res.backward_targets = utils::Table<NodeID>::interpret(buffer);

					// done
					return res;
				}

			};

			using LevelData = utils::StaticMap<utils::keys<EdgeKinds...>,Relation>;

			using EdgeData = std::array<LevelData,Levels>;

			EdgeData data;

		public:

			EdgeSet() = default;
			EdgeSet(const EdgeSet&) = default;
			EdgeSet(EdgeSet&& other) = default;

			EdgeSet& operator=(const EdgeSet&) = delete;
			EdgeSet& operator=(EdgeSet&&) = default;


			template<typename EdgeKind, unsigned Level>
			void addEdge(const NodeRef<typename EdgeKind::src_node_kind,Level>& src, const NodeRef<typename EdgeKind::trg_node_kind,Level>& trg) {
				getEdgeRelation<EdgeKind,Level>().addEdge(src,trg);
			}

			void close() {
				// for all levels
				for(auto& level : data) {
					// for all edge kinds
					for(auto& rel : level) {
						rel.close();
					}
				}
			}

			bool isClosed() const {
				// for all levels
				for(const auto& level : data) {
					// for all edge kinds
					for(const auto& rel : level) {
						// check this instance
						if (!rel.isClosed()) return false;
					}
				}
				// all are done
				return true;
			}

			template<typename EdgeKind, unsigned Level>
			NodeList<typename EdgeKind::trg_node_kind,Level> getSinks(const NodeRef<typename EdgeKind::src_node_kind,Level>& src) const {
				return getEdgeRelation<EdgeKind,Level>().template getSinks<EdgeKind>(src);
			}

			template<typename EdgeKind, unsigned Level>
			NodeList<typename EdgeKind::src_node_kind,Level> getSources(const NodeRef<typename EdgeKind::trg_node_kind,Level>& src) const {
				return getEdgeRelation<EdgeKind,Level>().template getSources<EdgeKind>(src);
			}

			// -- IO support --

			void store(std::ostream& out) const {
				// only allow closed sets to be stored
				assert_true(isClosed());

				// store each relation independently
				for(const auto& level : data) {
					for(const auto& rel : level) {
						rel.store(out);
					}
				}

			}

			static EdgeSet load(std::istream& in) {

				EdgeSet res;

				// load each relation independently
				for(auto& level : res.data) {
					for(auto& rel : level) {
						rel = Relation::load(in);
					}
				}

				// done
				return res;
			}

			static EdgeSet interpret(utils::RawBuffer& buffer) {

				EdgeSet res;

				// interpret each relation independently
				for(auto& level : res.data) {
					for(auto& rel : level) {
						rel = Relation::interpret(buffer);
					}
				}

				// done
				return res;
			}

		private:

			template<typename EdgeKind, std::size_t Level>
			Relation& getEdgeRelation() {
				return data[Level].template get<EdgeKind>();
			}

			template<typename EdgeKind, std::size_t Level>
			const Relation& getEdgeRelation() const {
				return data[Level].template get<EdgeKind>();
			}

		};


		template<unsigned Levels, typename ... HierachyKinds>
		class HierarchySet {

			class Relation {

				// -- inefficient build structures --

				std::vector<std::vector<NodeID>> children;

				std::vector<NodeID> parents;

				// -- efficient simulation structures --

				utils::Table<NodeID> parent_targets;

				utils::Table<std::size_t> children_offsets;
				utils::Table<NodeID> children_targets;

			public:

				void addChild(const NodeID& parent, const NodeID& child) {
					// a constant for an unknown parent
					static const NodeID unknownParent(std::numeric_limits<node_index_t>::max());

					assert_ne(parent,unknownParent) << "Unknown parent constant must not be used!";

					// register child as a child of parent
					if (parent >= children.size()) {
						children.resize(parent + 1);
					}
					auto& list = children[parent];
					for(auto& cur : list) if (cur == child) return;
					list.push_back(child);


					// register parent of child
					if (child >= parents.size()) {
						parents.resize(child + 1,unknownParent);
					}
					auto& trg = parents[child];
					assert_true(trg == unknownParent || trg == parent)
						<< "Double-assignment of parent for child " << child << " and parent " << parent;

					// update parent
					trg = parent;
				}

				bool isClosed() const {
					return children.empty();
				}

				void close() {
					// a constant for an unknown parent
					static const NodeID unknownParent(std::numeric_limits<node_index_t>::max());

					// get maximum index of parents
					std::size_t maxParent = 0;
					for(const auto& cur : parents) {
						maxParent = std::max<std::size_t>(maxParent,cur);
					}

					// compute total number of parent-child links
					std::size_t numParentChildLinks = 0;
					for(const auto& cur : children) {
						numParentChildLinks += cur.size();
					}

					// init forward / backward vectors
					children_offsets = utils::Table<std::size_t>(maxParent + 2, 0);
					children_targets = utils::Table<NodeID>(numParentChildLinks);

					// init child offsets
					std::size_t idx = 0;
					std::size_t offset = 0;
					for(const auto& cur : children) {
						children_offsets[idx] = offset;
						offset += cur.size();
						idx++;
						if (idx > maxParent) break;
					}
					children_offsets[idx] = offset;

					// fill in targets
					idx = 0;
					for(const auto& cur : children) {
						for(const auto& child : cur) {
							children_targets[idx++] = child;
						}
					}

					// clear edges
					children.clear();

					// init parent target table
					parent_targets = utils::Table<NodeID>(parents.size());
					for(std::size_t i=0; i<parent_targets.size(); ++i) {
						parent_targets[i] = (i < parents.size()) ? parents[i] : unknownParent;
					}

					// clear parents list
					parents.clear();
				}


				template<typename HierarchyKind, unsigned Level>
				NodeList<typename HierarchyKind::child_node_kind,Level-1> getChildren(const NodeRef<typename HierarchyKind::parent_node_kind,Level>& parent) const {
					using List = NodeList<typename HierarchyKind::child_node_kind,Level-1>;
					using ChildNodeRef = NodeRef<typename HierarchyKind::child_node_kind,Level-1>;
					assert_true(isClosed());
					if (parent.id >= children_offsets.size()-1 || children_targets.empty()) return List{nullptr,nullptr};
					return List{
						reinterpret_cast<const ChildNodeRef*>(&children_targets[children_offsets[parent.id]]),
						reinterpret_cast<const ChildNodeRef*>(&children_targets[children_offsets[parent.id+1]])
					};
				}

				template<typename HierarchyKind, unsigned Level>
				NodeRef<typename HierarchyKind::parent_node_kind,Level+1> getParent(const NodeRef<typename HierarchyKind::child_node_kind,Level>& child) const {
					using ParentNodeRef = NodeRef<typename HierarchyKind::parent_node_kind,Level+1>;
					assert_true(isClosed());
					assert_lt(child.id,parent_targets.size());
					return ParentNodeRef(parent_targets[child.id]);
				}

				// -- IO support --

				void store(std::ostream& out) const {
					// only allow closed sets to be stored
					assert_true(isClosed());

					// write parents table
					parent_targets.store(out);

					// write child lists
					children_offsets.store(out);
					children_targets.store(out);
				}

				static Relation load(std::istream& in) {

					Relation res;

					// restore parents
					res.parent_targets = utils::Table<NodeID>::load(in);

					res.children_offsets = utils::Table<std::size_t>::load(in);
					res.children_targets = utils::Table<NodeID>::load(in);

					// done
					return res;
				}

				static Relation interpret(utils::RawBuffer& buffer) {

					Relation res;

					// restore parents
					res.parent_targets = utils::Table<NodeID>::interpret(buffer);

					res.children_offsets = utils::Table<std::size_t>::interpret(buffer);
					res.children_targets = utils::Table<NodeID>::interpret(buffer);

					// done
					return res;
				}

			};

			using LevelData = utils::StaticMap<utils::keys<HierachyKinds...>,Relation>;

			using HierarchyData = std::array<LevelData,Levels-1>;

			HierarchyData data;

		public:

			template<typename HierarchyKind, unsigned Level>
			void addChild(const NodeRef<typename HierarchyKind::parent_node_kind,Level>& parent, const NodeRef<typename HierarchyKind::child_node_kind,Level-1>& child) {
				getRelation<HierarchyKind,Level-1>().addChild(parent,child);
			}

			void close() {
				for(auto& level : data) {
					for(auto& rel : level) {
						rel.close();
					}
				}
			}

			bool isClosed() const {
				for(const auto& level : data) {
					for(const auto& rel : level) {
						if (!rel.isClosed()) return false;
					}
				}
				return true;
			}

			template<typename HierarchyKind, unsigned Level>
			NodeList<typename HierarchyKind::child_node_kind,Level-1> getChildren(const NodeRef<typename HierarchyKind::parent_node_kind,Level>& parent) const {
				return getRelation<HierarchyKind,Level-1>().template getChildren<HierarchyKind>(parent);
			}

			template<typename HierarchyKind, unsigned Level>
			NodeRef<typename HierarchyKind::parent_node_kind,Level+1> getParent(const NodeRef<typename HierarchyKind::child_node_kind,Level>& child) const {
				return getRelation<HierarchyKind,Level>().template getParent<HierarchyKind>(child);
			}


			// -- IO support --

			void store(std::ostream& out) const {
				// only allow closed sets to be stored
				assert_true(isClosed());

				// store each relation independently
				for(const auto& level : data) {
					for(const auto& rel : level) {
						rel.store(out);
					}
				}

			}

			static HierarchySet load(std::istream& in) {

				HierarchySet res;

				// load each relation independently
				for(auto& level : res.data) {
					for(auto& rel : level) {
						rel = Relation::load(in);
					}
				}

				// done
				return res;
			}

			static HierarchySet interpret(utils::RawBuffer& buffer) {

				HierarchySet res;

				// interpret each relation independently
				for(auto& level : res.data) {
					for(auto& rel : level) {
						rel = Relation::interpret(buffer);
					}
				}

				// done
				return res;
			}

		private:

			template<typename HierarchyKind, std::size_t Level>
			Relation& getRelation() {
				return data[Level].template get<HierarchyKind>();
			}

			template<typename HierarchyKind, std::size_t Level>
			const Relation& getRelation() const {
				return data[Level].template get<HierarchyKind>();
			}

		};


		// -- utilities for enumerating level/kind combinations --

		template<typename ... Kinds>
		struct KindEnumerator;

		template<typename First, typename ... Rest>
		struct KindEnumerator<First,Rest...> {
			template<typename Body>
			void operator()(const Body& body) const {
				body(First());
				KindEnumerator<Rest...>()(body);
			}
		};

		template<>
		struct KindEnumerator<> {
			template<typename Body>
			void operator()(const Body&) const {}
		};


		template<std::size_t Level>
		struct LevelEnumerator {
			template<typename Body>
			void operator()(const Body& body) const {
				body(level<Level>());
				LevelEnumerator<Level-1>()(body);
			}
		};

		template<>
		struct LevelEnumerator<0> {
			template<typename Body>
			void operator()(const Body& body) const {
				body(level<0>());
			}
		};

		template<std::size_t Level>
		struct HierarchyLevelEnumerator {
			template<typename Body>
			void operator()(const Body& body) const {
				body(level<Level>());
				HierarchyLevelEnumerator<Level-1>()(body);
			}
		};

		template<>
		struct HierarchyLevelEnumerator<1> {
			template<typename Body>
			void operator()(const Body& body) const {
				body(level<1>());
			}
		};

		template<>
		struct HierarchyLevelEnumerator<0> {
			template<typename Body>
			void operator()(const Body&) const {}
		};


		// -- mesh topology store --

		template<
			typename Nodes,
			typename Edges,
			typename Hierarchies,
			unsigned Levels
		>
		struct MeshTopologyData;

		template<
			typename ... Nodes,
			typename ... Edges,
			typename ... Hierarchies,
			unsigned Levels
		>
		struct MeshTopologyData<nodes<Nodes...>,edges<Edges...>,hierarchies<Hierarchies...>,Levels> {

			using NodeSetType = NodeSet<Levels,Nodes...>;
			using EdgeSetType = EdgeSet<Levels,Edges...>;
			using HierarchySetType = HierarchySet<Levels,Hierarchies...>;

			// the topological data of all the nodes, edges and hierarchy relations on all levels
			NodeSetType nodeSets;
			EdgeSetType edgeSets;
			HierarchySetType hierarchySets;

			MeshTopologyData() = default;
			MeshTopologyData(const MeshTopologyData&) = default;
			MeshTopologyData(MeshTopologyData&& other) = default;

			MeshTopologyData& operator= (MeshTopologyData&& m) = default;

			template<typename Body>
			void forAllNodeKinds(const Body& body) const {
				LevelEnumerator<Levels-1> forAllLevels;
				KindEnumerator<Nodes...> forAllKinds;
				forAllLevels([&](const auto& level){
					forAllKinds([&](const auto& kind){
						body(kind,level);
					});
				});
			}

			template<typename Body>
			void forAllEdgeKinds(const Body& body) const {
				LevelEnumerator<Levels-1> forAllLevels;
				KindEnumerator<Edges...> forAllKinds;
				forAllLevels([&](const auto& level){
					forAllKinds([&](const auto& kind){
						body(kind,level);
					});
				});
			}

			template<typename Body>
			void forAllHierarchyKinds(const Body& body) const {
				HierarchyLevelEnumerator<Levels-1> forAllLevels;
				KindEnumerator<Hierarchies...> forAllKinds;
				forAllLevels([&](const auto& level){
					forAllKinds([&](const auto& kind){
						body(kind,level);
					});
				});
			}

			template<typename Kind,unsigned Level = 0>
			std::size_t getNumNodes() const {
				return nodeSets.template getNumNodes<Kind,Level>();
			}

			void close() {
				edgeSets.close();
				hierarchySets.close();
			}

			bool isClosed() const {
				return edgeSets.isClosed() && hierarchySets.isClosed();
			}

			// -- IO support --

			void store(std::ostream& out) const {
				nodeSets.store(out);
				edgeSets.store(out);
				hierarchySets.store(out);
			}

			static MeshTopologyData load(std::istream& in) {
				MeshTopologyData res;
				res.nodeSets = NodeSetType::load(in);
				res.edgeSets = EdgeSetType::load(in);
				res.hierarchySets = HierarchySetType::load(in);
				return std::move(res);
			}

			static MeshTopologyData interpret(utils::RawBuffer& buffer) {
				MeshTopologyData res;
				res.nodeSets = NodeSetType::interpret(buffer);
				res.edgeSets = EdgeSetType::interpret(buffer);
				res.hierarchySets = HierarchySetType::interpret(buffer);
				return std::move(res);
			}

		};

		/**
		 * A common basis class for sub-tree and sub-graph references, which are both based on paths
		 * within a tree.
		 */
		template<typename Derived>
		class PathRefBase {

		protected:

			using value_t = uint32_t;

			value_t path;
			value_t mask;

			PathRefBase(value_t path, value_t mask)
				: path(path), mask(mask) {}

		public:

			static Derived root() {
				return { 0u , 0u };
			}

			value_t getPath() const {
				return path;
			}

			value_t getMask() const {
				return mask;
			}

			value_t getDepth() const {
				if (PathRefBase::mask == 0) return 0;
				return sizeof(PathRefBase::mask) * 8 - utils::countLeadingZeros(PathRefBase::mask);
			}

			bool isRoot() const {
				return PathRefBase::mask == 0;
			}

			bool isLeftChild() const {
				assert_false(isRoot());
				return !isRightChild();
			}

			bool isRightChild() const {
				assert_false(isRoot());
				return PathRefBase::path & (1 << (getDepth()-1));
			}

			Derived getLeftChild() const {
				assert_lt(getDepth(),sizeof(PathRefBase::path)*8);
				Derived res = asDerived();
				res.PathRefBase::mask = res.PathRefBase::mask | (1 << getDepth());
				return res;
			}

			Derived getRightChild() const {
				Derived res = getLeftChild();
				res.PathRefBase::path = res.PathRefBase::path | (1 << getDepth());
				return res;
			}

			bool operator==(const Derived& other) const {
				// same mask and same valid bit part
				return (PathRefBase::mask == other.PathRefBase::mask) &&
						((PathRefBase::path & PathRefBase::mask) == (other.PathRefBase::path & other.PathRefBase::mask));
			}

			bool operator!=(const Derived& other) const {
				return !(*this == other);
			}

			bool operator<(const Derived& other) const {

				auto thisMask = PathRefBase::mask;
				auto thatMask = other.PathRefBase::mask;

				auto thisPath = PathRefBase::path;
				auto thatPath = other.PathRefBase::path;

				while(true) {

					// if they are the same, we are done
					if (thisMask == thatMask && thisPath == thatPath) return false;

					// check last mask bit
					auto thisMbit = thisMask & 0x1;
					auto thatMbit = thatMask & 0x1;

					if (thisMbit < thatMbit) return true;
					if (thisMbit > thatMbit) return false;

					auto thisPbit = thisMbit & thisPath;
					auto thatPbit = thatMbit & thatPath;

					if (thisPbit < thatPbit) return true;
					if (thisPbit > thatPbit) return false;

					thisMask >>= 1;
					thatMask >>= 1;
					thisPath >>= 1;
					thatPath >>= 1;
				}
			}

			bool operator<=(const Derived& other) const {
				return *this == other || *this < other;
			}

			bool operator>=(const Derived& other) const {
				return !(asDerived() < other);
			}

			bool operator>(const Derived& other) const {
				return !(*this <= other);
			}

			bool covers(const Derived& other) const {
				if (getDepth() > other.getDepth()) return false;
				if (PathRefBase::mask != (PathRefBase::mask & other.PathRefBase::mask)) return false;
				return (PathRefBase::mask & PathRefBase::path) == (PathRefBase::mask & other.PathRefBase::path);
			}

			bool tryMerge(const Derived& other) {

				if (covers(other)) return true;

				if (other.covers(asDerived())) {
					*this = other;
					return true;
				}

				// the masks need to be identical
				auto thisMask = PathRefBase::mask;
				auto thatMask = other.PathRefBase::mask;
				if (thisMask != thatMask) return false;


				// the valid portion of the paths must only differe in one bit
				auto thisPath = PathRefBase::path;
				auto thatPath = other.PathRefBase::path;

				auto thisValid = thisPath & thisMask;
				auto thatValid = thatPath & thatMask;

				auto diff = thisValid ^ thatValid;

				// if there is more than 1 bit difference, there is nothing we can do
				if (utils::countOnes(diff) != 1) return false;

				// ignore this one bit in the mask
				PathRefBase::mask = PathRefBase::mask & (~diff);

				// done
				return true;
			}

			/**
			 * @return true if the intersection is not empty;
			 * 			in this case this instance has been updated to represent the intersection
			 * 		   false if the intersection is empty, the object has not been altered
			 */
			bool tryIntersect(const Derived& other) {

				// if the other covers this, the intersection is empty
				if (other.covers(asDerived())) return true;

				// if this one is the larger one, this one gets reduced to the smaller one
				if (covers(other)) {
					*this = other;
					return true;
				}

				// make sure common constraints are identical
				auto filterMask = PathRefBase::mask & other.PathRefBase::mask;
				auto thisFilter = PathRefBase::path & filterMask;
				auto thatFilter = other.PathRefBase::path & filterMask;
				if (thisFilter != thatFilter) return false;

				// unite (disjunction!) the constraints of both sides
				PathRefBase::path = (PathRefBase::path & PathRefBase::mask) | (other.PathRefBase::path & other.PathRefBase::mask);
				PathRefBase::mask = PathRefBase::mask | other.PathRefBase::mask;
				return true;
			}



			template<typename Body>
			void visitComplement(const Body& body, unsigned depth = 0) const {

				// when we reached the depth of this reference, we are done
				if (getDepth() == depth) return;

				auto bitMask = (1 << depth);

				// if at this depth there is no wild card
				if (PathRefBase::mask & bitMask) {

					// invert bit at this position
					Derived cpy = asDerived();
					cpy.PathRefBase::path ^= bitMask;
					cpy.PathRefBase::mask = cpy.PathRefBase::mask & ((bitMask << 1) - 1);

					// this is an element of the complement
					body(cpy);

					// continue path
					visitComplement<Body>(body,depth+1);

					return;
				}

				// follow both paths, do nothing here
				Derived cpy = asDerived();
				cpy.PathRefBase::mask = PathRefBase::mask | bitMask;

				// follow the 0 path
				cpy.PathRefBase::path = PathRefBase::path & ~bitMask;
				cpy.template visitComplement<Body>(body,depth+1);

				// follow the 1 path
				cpy.PathRefBase::path = PathRefBase::path | bitMask;
				cpy.template visitComplement<Body>(body,depth+1);

			}

			std::vector<Derived> getComplement() const {
				std::vector<Derived> res;
				visitComplement([&](const Derived& cur){
					res.push_back(cur);
				});
				return res;
			}

		private:

			Derived& asDerived() {
				return static_cast<Derived&>(*this);
			}

			const Derived& asDerived() const {
				return static_cast<const Derived&>(*this);
			}

		};


		/**
		 * A utility to address nodes in the partition tree.
		 */
		class SubTreeRef : public PathRefBase<SubTreeRef> {

			using super = PathRefBase<SubTreeRef>;

			friend super;

			friend class SubMeshRef;

			SubTreeRef(value_t path, value_t mask)
				: super(path,mask) {}

		public:

			value_t getIndex() const {
				// this is reversing the path 000ZYX to 1XYZ to get the usual
				// order of nodes within a embedded tree
				auto res = 1;
				value_t cur = path;
				for(unsigned i = 0; i<getDepth(); ++i) {
					res <<= 1;
					res += cur % 2;
					cur >>= 1;
				}
				return res;
			}


			SubTreeRef getParent() const {
				assert_false(isRoot());
				SubTreeRef res = *this;
				res.PathRefBase::mask = res.PathRefBase::mask & ~(1 << (getDepth()-1));
				return res;
			}


			template<unsigned DepthLimit, bool preOrder, typename Body>
			void enumerate(const Body& body) {

				if (preOrder) body(*this);

				if (getDepth() < DepthLimit) {
					getLeftChild().enumerate<DepthLimit,preOrder>(body);
					getRightChild().enumerate<DepthLimit,preOrder>(body);
				}

				if (!preOrder) body(*this);

			}


			friend std::ostream& operator<<(std::ostream& out, const SubTreeRef& ref) {
				out << "r";
				auto depth = ref.getDepth();
				for(value_t i = 0; i<depth; ++i) {
					out << "." << ((ref.path >> i) % 2);
				}
				return out;
			}

		};


		/**
		 * A reference to a continuously stored part of a mesh.
		 */
		class SubMeshRef : public PathRefBase<SubMeshRef> {

			using super = PathRefBase<SubMeshRef>;

			using value_t = uint32_t;

			friend super;

			SubMeshRef(value_t path, value_t mask)
				: super(path,mask) {}

		public:

			SubMeshRef(const SubTreeRef& ref)
				: super(ref.path, ref.mask) {}

			SubMeshRef getMasked(unsigned pos) const {
				assert_lt(pos,getDepth());
				SubMeshRef res = *this;
				res.super::mask = res.super::mask & ~(1<<pos);
				return res;
			}

			SubMeshRef getUnmasked(unsigned pos) const {
				assert_lt(pos,getDepth());
				SubMeshRef res = *this;
				res.super::mask = res.super::mask | (1<<pos);
				return res;
			}

			SubTreeRef getEnclosingSubTree() const {
				return SubTreeRef(
					super::path,
					(1 << utils::countTrailingZeros(~super::mask)) - 1
				);
			}

			template<typename Body>
			void scan(const Body& body) const {

				// look for last 0 in mask
				unsigned zeroPos = utils::countTrailingZeros(~super::mask);
				if (zeroPos >= getDepth()) {
					body(SubTreeRef(super::path,super::mask));
					return;
				}

				// recursive
				SubMeshRef copy = getUnmasked(zeroPos);

				// set bit to 0
				copy.super::path = copy.super::path & ~( 1 << zeroPos );
				copy.scan(body);

				// set bit to 1
				copy.super::path = copy.super::path |  ( 1 << zeroPos );
				copy.scan(body);
			}


			template<typename NodeKind, unsigned Level, typename PartitionTree, typename Body>
			void scan(const PartitionTree& ptree, const Body& body) const {
				scan([&](const SubTreeRef& ref){
					ptree.template getNodeRange<NodeKind,Level>(ref).forAll(body);
				});
			}


			friend std::ostream& operator<<(std::ostream& out, const SubMeshRef& ref) {
				out << "r";
				auto depth = ref.getDepth();
				for(value_t i = 0; i<depth; ++i) {
					if (ref.super::mask & (1 << i)) {
						out << "." << ((ref.super::path >> i) % 2);
					} else {
						out << ".*";
					}
				}
				return out;
			}

		};

		/**
		 * A union of sub mesh references.
		 */
		class MeshRegion {

			template<
				typename Nodes,
				typename Edges,
				typename Hierarchies,
				unsigned Levels,
				unsigned PartitionDepth
			>
			friend class PartitionTree;

			std::vector<SubMeshRef> refs;

			MeshRegion(const SubMeshRef* begin, const SubMeshRef* end)
				: refs(begin,end) {}

		public:

			MeshRegion() {}

			MeshRegion(const SubMeshRef& ref) {
				refs.push_back(ref);
			}

			MeshRegion(std::initializer_list<SubMeshRef> meshRefs) : refs(meshRefs) {
				restoreSet();
				compress();
			}

			MeshRegion(const std::vector<SubMeshRef>& refs) : refs(refs) {
				restoreSet();
				compress();
			}

			bool operator==(const MeshRegion& other) const {
				return this == &other || refs == other.refs || (difference(*this,other).empty() && difference(other,*this).empty());
			}

			bool operator!=(const MeshRegion& other) const {
				return !(*this == other);
			}

			const std::vector<SubMeshRef>& getSubMeshReferences() const {
				return refs;
			}

			bool empty() const {
				return refs.empty();
			}

			bool covers(const SubMeshRef& ref) const {
				// cheap: one is covering the given reference
				// expensive: the union of this and the reference is the same as this
				return std::any_of(refs.begin(),refs.end(),[&](const SubMeshRef& a) {
					return a.covers(ref);
				}) || (merge(*this,MeshRegion(ref)) == *this);
			}

			bool operator<(const MeshRegion& other) const {
				return refs < other.refs;
			}

			static MeshRegion merge(const MeshRegion& a, const MeshRegion& b) {
				MeshRegion res;
				std::set_union(
					a.refs.begin(), a.refs.end(),
					b.refs.begin(), b.refs.end(),
					std::back_inserter(res.refs)
				);
				res.compress();
				return res;
			}

			template<typename ... Rest>
			static MeshRegion merge(const MeshRegion& a, const MeshRegion& b, const Rest& ... rest) {
				return merge(merge(a,b),rest...);
			}

			static MeshRegion intersect(const MeshRegion& a, const MeshRegion& b) {

				MeshRegion res;

				// compute pairwise intersections
				for(const auto& ra : a.refs) {
					for(const auto& rb : b.refs) {
						auto tmp = ra;
						if (tmp.tryIntersect(rb)) {
							res.refs.push_back(tmp);
						}
					}
				}

				// restore set invariant
				res.restoreSet();

				// compress the set representation
				res.compress();
				return res;
			}

			static MeshRegion difference(const MeshRegion& a, const MeshRegion& b) {
				return intersect(a,complement(b));
			}

			static MeshRegion span(const MeshRegion&, const MeshRegion&) {
				std::cout << "Scan operation not yet implemented!";
				exit(1);
			}

			MeshRegion complement() const {

				MeshRegion res = SubMeshRef::root();

				// aggregate the complements of all entries
				for(const auto& cur : refs) {

					// compute the complement of the current entry
					MeshRegion tmp;
					cur.visitComplement([&](const SubMeshRef& ref) {
						tmp.refs.push_back(ref);
					});

					// restore invariant
					tmp.restoreSet();
					tmp.compress();

					// intersect current complement with running complement
					res = intersect(res,tmp);
				}

				// done
				return res;
			}

			static MeshRegion complement(const MeshRegion& region) {
				return region.complement();
			}

			/**
			 * An operator to load an instance of this region from the given archive.
			 */
			static MeshRegion load(utils::ArchiveReader&) {
				assert_not_implemented();
				return MeshRegion();
			}

			/**
			 * An operator to store an instance of this region into the given archive.
			 */
			void store(utils::ArchiveWriter&) const {
				assert_not_implemented();
				// nothing so far
			}

			template<typename Body>
			void scan(const Body& body) const {
				for(const auto& cur : refs) {
					cur.scan(body);
				}
			}

			template<typename NodeKind, unsigned Level, typename PartitionTree, typename Body>
			void scan(const PartitionTree& ptree, const Body& body) const {
				for(const auto& cur : refs) {
					cur.scan<NodeKind,Level>(ptree,body);
				}
			}


			friend std::ostream& operator<<(std::ostream& out, const MeshRegion& reg) {
				return out << reg.refs;
			}

		private:

			void compress() {

				// check precondition
				assert_true(std::is_sorted(refs.begin(),refs.end()));

				// Phase 1:  remove redundant entries
				removeCovered();

				// Phase 2:  collapse adjacent entries (iteratively)
				while (collapseSiblings()) {}
			}


			bool removeCovered() {

				// see whether any change happend
				bool changed = false;
				for(std::size_t i = 0; i<refs.size(); ++i) {

					auto& cur = refs[i];
					auto closure = cur.getEnclosingSubTree();

					std::size_t j = i+1;
					while(j < refs.size() && closure.covers(refs[j].getEnclosingSubTree())) {
						if (cur.covers(refs[j])) {
							refs[j] = cur;
							changed = true;
						}
						++j;
					}

				}

				// restore set condition
				if (changed) restoreSet();

				// report whether the content has been changed
				return changed;
			}

			bool collapseSiblings() {

				// see whether any change happend
				bool changed = false;
				auto size = refs.size();
				for(std::size_t i = 0; i<size; ++i) {
					for(std::size_t j = i+1; j<size; ++j) {
						if (refs[i].tryMerge(refs[j])) {
							refs[j] = refs[i];
							changed = true;
						}
					}
				}

				// restore set condition
				if (changed) restoreSet();

				// report whether the content has been changed
				return changed;

			}

			void restoreSet() {
				// sort elements
				std::sort(refs.begin(),refs.end());
				// remove duplicates
				refs.erase(std::unique(refs.begin(),refs.end()),refs.end());
			}

		};


		// --------------------------------------------------------------
		//					Partition Tree
		// --------------------------------------------------------------


		template<
			typename Nodes,
			typename Edges,
			typename Hierarchies = hierarchies<>,
			unsigned Levels = 1,
			unsigned depth = 12
		>
		class PartitionTree;

		template<
			typename Nodes,
			typename Edges,
			typename Hierarchies,
			unsigned Levels,
			unsigned depth
		>
		class PartitionTree {

			static_assert(detail::is_nodes<Nodes>::value,
					"First template argument of PartitionTree must be of type nodes<...>");

			static_assert(detail::is_edges<Edges>::value,
					"Second template argument of PartitionTree must be of type edges<...>");

			static_assert(detail::is_hierarchies<Hierarchies>::value,
					"Third template argument of PartitionTree must be of type hierarchies<...>");

		};

		template<
			typename ... Nodes,
			typename ... Edges,
			typename ... Hierarchies,
			unsigned Levels,
			unsigned PartitionDepth
		>
		class PartitionTree<nodes<Nodes...>,edges<Edges...>,hierarchies<Hierarchies...>,Levels,PartitionDepth> {

		public:

			enum { depth = PartitionDepth };

		private:

			// an internal construct to store node ranges
			struct RangeStore {
				NodeID begin;
				NodeID end;
			};

			// an internal construct to store regions in open and
			// closed structure
			//		- open:   the region pointer is referencing the stored region
			//		- closed: the begin and end indices reference and interval of an externally maintained
			//					list of regions
			struct RegionStore {

				// -- open --
				MeshRegion* region;			// the ownership is managed by the enclosing tree

				// -- closed --
				std::size_t offset;
				std::size_t length;

				RegionStore()
					: region(nullptr), offset(0), length(0) {}

				MeshRegion toRegion(const SubMeshRef* references) const {
					if (region) return *region;
					auto start = references + offset;
					auto end = start + length;
					return MeshRegion(start,end);
				}

				RegionStore& operator=(const MeshRegion& value) {
					if (!region) region = new MeshRegion();
					*region = value;
					return *this;
				}
			};


			static_assert(Levels > 0, "There must be at least one level!");

			struct LevelInfo {

				utils::StaticMap<utils::keys<Nodes...>,RangeStore> nodeRanges;

				utils::StaticMap<utils::keys<Edges...>,RegionStore> forwardClosure;
				utils::StaticMap<utils::keys<Edges...>,RegionStore> backwardClosure;

				utils::StaticMap<utils::keys<Hierarchies...>,RegionStore> parentClosure;
				utils::StaticMap<utils::keys<Hierarchies...>,RegionStore> childClosure;

			};

			struct Node {

				std::array<LevelInfo,Levels> data;

			};

			// some preconditions required for the implementation of this class to work
			static_assert(std::is_trivially_copyable<RangeStore>::value,  "RangeStore should be trivially copyable!");
			static_assert(std::is_trivially_copyable<RegionStore>::value, "RegionStore should be trivially copyable!");
			static_assert(std::is_trivially_copyable<LevelInfo>::value,   "LevelInfo should be trivially copyable!"  );
			static_assert(std::is_trivially_copyable<Node>::value,        "Nodes should be trivially copyable!"      );
			static_assert(std::is_trivially_copyable<SubMeshRef>::value,  "SubMeshRefs should be trivially copyable!");

			enum { num_elements = 1ul << (depth + 1) };

			bool owned;

			Node* data;

			std::size_t numReferences;

			SubMeshRef* references;

			PartitionTree(Node* data, std::size_t numReferences, SubMeshRef* references)
				: owned(false), data(data), numReferences(numReferences), references(references) {
				assert_true(data);
				assert_true(references);
			}

		public:

			PartitionTree() : owned(true), data(new Node[num_elements]), numReferences(0), references(nullptr) {}

			~PartitionTree() {
				if (owned) {
					delete [] data;
					free(references);
				}
			}

			PartitionTree(const PartitionTree&) = delete;

			PartitionTree(PartitionTree&& other)
				: owned(other.owned),
				  data(other.data),
				  numReferences(other.numReferences),
				  references(other.references) {

				// free other from ownership
				other.owned = false;
				other.data = nullptr;
				other.references = nullptr;
			}

			PartitionTree& operator=(const PartitionTree&) = delete;

			PartitionTree& operator=(PartitionTree&& other) {
				assert_ne(this,&other);

				// swap content and ownership
				std::swap(owned,other.owned);
				numReferences = other.numReferences;
				std::swap(data,other.data);
				std::swap(references,other.references);

				// done
				return *this;
			}

			bool isClosed() const {
				return references != nullptr;
			}

			void close() {
				// must not be closed for now
				assert_false(isClosed());

				// a utility to apply an operation on each mesh region
				auto forEachMeshRegion = [&](const auto& op) {
					for(std::size_t i=0; i<num_elements; ++i) {
						Node& cur = data[i];
						for(std::size_t l=0; l<Levels; ++l) {
							cur.data[l].forwardClosure .forEach(op);
							cur.data[l].backwardClosure.forEach(op);
							cur.data[l].parentClosure  .forEach(op);
							cur.data[l].childClosure   .forEach(op);
						}
					}
				};

				// count number of references required for all ranges
				numReferences = 0;
				forEachMeshRegion([&](const RegionStore& cur) {
					if (!cur.region) return;
					numReferences += cur.region->getSubMeshReferences().size();
				});

				// create reference buffer
				references = static_cast<SubMeshRef*>(malloc(sizeof(SubMeshRef) * numReferences));
				if (!references) {
					throw "Unable to allocate memory for managing references!";
				}

				// transfer ownership of SubMeshRefs to reference buffer
				std::size_t offset = 0;
				forEachMeshRegion([&](RegionStore& cur){

					// check whether there is a region
					if (!cur.region) {
						cur.offset = 0;
						cur.length = 0;
						return;
					}

					// close the region
					const auto& refs = cur.region->getSubMeshReferences();
					cur.offset = offset;
					cur.length = refs.size();
					for(auto& cur : refs) {
						// placement new for this reference
						new (&references[offset++]) SubMeshRef(cur);
					}

					// delete old region
					delete cur.region;
					cur.region = nullptr;
				});

				// make sure counting and transferring covered the same number of references
				assert_eq(numReferences, offset);
			}

			template<typename Kind, unsigned Level = 0>
			NodeRange<Kind,Level> getNodeRange(const SubTreeRef& ref = SubTreeRef::root()) const {
				assert_lt(ref.getIndex(),num_elements);
				auto range = data[ref.getIndex()].data[Level].nodeRanges.template get<Kind>();
				return {
					NodeRef<Kind,Level>{ range.begin },
					NodeRef<Kind,Level>{ range.end }
				};
			}

			template<typename Kind, unsigned Level = 0>
			void setNodeRange(const SubTreeRef& ref, const NodeRange<Kind,Level>& range) {
				auto& locRange = getNode(ref).data[Level].nodeRanges.template get<Kind>();
				locRange.begin = range.getBegin();
				locRange.end = range.getEnd();
			}

			template<typename EdgeKind, unsigned Level = 0>
			MeshRegion getForwardClosure(const SubTreeRef& ref) const {
				return getNode(ref).data[Level].forwardClosure.template get<EdgeKind>().toRegion(references);
			}

			template<typename EdgeKind, unsigned Level = 0>
			void setForwardClosure(const SubTreeRef& ref, const MeshRegion& region) {
				getNode(ref).data[Level].forwardClosure.template get<EdgeKind>() = region;
			}

			template<typename EdgeKind, unsigned Level = 0>
			MeshRegion getBackwardClosure(const SubTreeRef& ref) const {
				return getNode(ref).data[Level].backwardClosure.template get<EdgeKind>().toRegion(references);
			}

			template<typename EdgeKind, unsigned Level = 0>
			void setBackwardClosure(const SubTreeRef& ref, const MeshRegion& region) {
				getNode(ref).data[Level].backwardClosure.template get<EdgeKind>() = region;
			}

			template<typename HierarchyKind, unsigned Level = 0>
			MeshRegion getParentClosure(const SubTreeRef& ref) const {
				return getNode(ref).data[Level].parentClosure.template get<HierarchyKind>().toRegion(references);
			}

			template<typename HierarchyKind, unsigned Level = 0>
			void setParentClosure(const SubTreeRef& ref, const MeshRegion& region) {
				getNode(ref).data[Level].parentClosure.template get<HierarchyKind>() = region;
			}


			template<typename HierarchyKind, unsigned Level = 1>
			MeshRegion getChildClosure(const SubTreeRef& ref) const {
				return getNode(ref).data[Level].childClosure.template get<HierarchyKind>().toRegion(references);
			}

			template<typename HierarchyKind, unsigned Level = 1>
			void setChildClosure(const SubTreeRef& ref, const MeshRegion& region) {
				getNode(ref).data[Level].childClosure.template get<HierarchyKind>() = region;
			}


			template<typename Body>
			void visitPreOrder(const Body& body) {
				SubTreeRef::root().enumerate<depth,true>(body);
			}

			template<typename Body>
			void visitPostOrder(const Body& body) {
				SubTreeRef::root().enumerate<depth,false>(body);
			}

			// -- serialization support for network transferes --

			void store(utils::ArchiveWriter&) const {
				assert_not_implemented();
			}

			static PartitionTree load(utils::ArchiveReader&) {
				assert_not_implemented();
				return PartitionTree();
			}

			// -- load / store for files --

			void store(std::ostream& out) const {

				// start by writing out number of references
				out.write(reinterpret_cast<const char*>(&numReferences),sizeof(numReferences));

				// continue with node information
				out.write(reinterpret_cast<const char*>(data),sizeof(Node)*num_elements);

				// and end with references
				out.write(reinterpret_cast<const char*>(references),sizeof(SubMeshRef)*numReferences);

			}

			static PartitionTree load(std::istream& in) {

				// create the resulting tree (owning all its data)
				PartitionTree res;

				// read in number of references
				in.read(reinterpret_cast<char*>(&res.numReferences),sizeof(res.numReferences));

				// load nodes
				in.read(reinterpret_cast<char*>(res.data),sizeof(Node)*num_elements);

				// load references
				res.references = reinterpret_cast<SubMeshRef*>(malloc(sizeof(SubMeshRef)*res.numReferences));
				in.read(reinterpret_cast<char*>(res.references),sizeof(SubMeshRef)*res.numReferences);

				// done
				return res;
			}

			static PartitionTree interpret(utils::RawBuffer& raw) {

				// get size
				std::size_t numReferences = raw.consume<std::size_t>();

				// get nodes
				Node* nodes = raw.consumeArray<Node>(num_elements);

				// get references
				SubMeshRef* references = raw.consumeArray<SubMeshRef>(numReferences);

				// wrap up results
				return PartitionTree(nodes,numReferences,references);
			}


		private:

			const Node& getNode(const SubTreeRef& ref) const {
				assert_lt(ref.getIndex(),num_elements);
				return data[ref.getIndex()];
			}

			Node& getNode(const SubTreeRef& ref) {
				assert_lt(ref.getIndex(),num_elements);
				return data[ref.getIndex()];
			}

		};


		class NaiveMeshPartitioner {

		public:

			template<
				unsigned PartitionDepth,
				typename Nodes,
				typename Edges,
				typename Hierarchies,
				unsigned Levels
			>
			PartitionTree<Nodes,Edges,Hierarchies,Levels,PartitionDepth> partition(const MeshTopologyData<Nodes,Edges,Hierarchies,Levels>& data) const {

				// create empty partition tree
				PartitionTree<Nodes,Edges,Hierarchies,Levels,PartitionDepth> res;

				// set up node ranges for partitions
				data.forAllNodeKinds([&](const auto& nodeKind, const auto& level) {

						// get node kind and level
						using NodeKind = plain_type<decltype(nodeKind)>;
						// not directly accessing lvl::value here, as MSVC 15 refuses to acknowledge its constexpr-ness
						using lvl = get_level<decltype(level)>;

						// set root node to cover the full range
						auto num_nodes = data.template getNumNodes<NodeKind, lvl::value>();
						res.template setNodeRange<NodeKind, lvl::value>(
								SubTreeRef::root(),
								NodeRange<NodeKind, lvl::value>(
									NodeRef<NodeKind, lvl::value>{ 0 },
									NodeRef<NodeKind, lvl::value>{ NodeID((node_index_t)num_nodes) }
								)
						);

						// recursively sub-divide ranges
						res.visitPreOrder([&](const SubTreeRef& ref) {

							if (ref.isRoot()) return;

							// get the range of the parent
							auto range = res.template getNodeRange<NodeKind, lvl::value>(ref.getParent());

							// extract begin / end
							auto begin = range.getBegin();
							auto end = range.getEnd();

							// compute mid
							auto mid = NodeRef<NodeKind, lvl::value>(begin.id + (end.id - begin.id) / 2);

							// get range for this node
							if (ref.isLeftChild()) {
								range = NodeRange<NodeKind, lvl::value>(begin,mid);
							} else {
								range = NodeRange<NodeKind, lvl::value>(mid,end);
							}

							// update the range
							res.template setNodeRange<NodeKind, lvl::value>(ref,range);

						});

				});

				// set up closures for edges
				data.forAllEdgeKinds([&](const auto& edgeKind, const auto& level) {

					// get edge kind and level
					using EdgeKind = plain_type<decltype(edgeKind)>;
					// not directly accessing lvl::value here, as MSVC 15 refuses to acknowledge its constexpr-ness
					using lvl = get_level<decltype(level)>;

					// the closure is everything for now
					MeshRegion closure = SubMeshRef::root();

					// initialize all the closured with the full region
					res.visitPreOrder([&](const SubTreeRef& ref) {
						// fix forward closure
						res.template setForwardClosure<EdgeKind,lvl::value>(ref,closure);

						// fix backward closure
						res.template setBackwardClosure<EdgeKind,lvl::value>(ref,closure);
					});

				});


				// set up closures for hierarchies
				data.forAllHierarchyKinds([&](const auto& hierarchyKind, const auto& level) {

					// get hierarchy kind and level
					using HierarchyKind = plain_type<decltype(hierarchyKind)>;
					// not directly accessing lvl::value here, as MSVC 15 refuses to acknowledge its constexpr-ness
					using lvl = get_level<decltype(level)>;

					// make sure this is not called for level 0
					assert_gt(lvl::value,0) << "There should not be any hierarchies on level 0.";

					// the closure is everything for now
					MeshRegion closure = SubMeshRef::root();

					// initialize all the closured with the full region
					res.visitPreOrder([&](const SubTreeRef& ref) {

						// fix parent closure
						res.template setParentClosure<HierarchyKind,lvl::value-1>(ref,closure);

						// fix child closure
						res.template setChildClosure<HierarchyKind,lvl::value>(ref,closure);
					});

				});

				// close the data representation
				res.close();

				// done
				return res;
			}

		};


		template<
			typename NodeKind,
			typename ElementType,
			unsigned Level,
			typename PartitionTree
		>
		class MeshDataFragment {
		public:

			using facade_type = MeshData<NodeKind,ElementType,Level,PartitionTree>;
			using region_type = MeshRegion;
			using shared_data_type = PartitionTree;

		private:

			using partition_tree_type = PartitionTree;

			const partition_tree_type& partitionTree;

			region_type coveredRegion;

			std::vector<ElementType> data;

			friend facade_type;

		public:

			MeshDataFragment() = delete;

			MeshDataFragment(const partition_tree_type& ptree, const region_type& region)
				: partitionTree(ptree), coveredRegion(region) {

				// get upper boundary of covered node ranges
				std::size_t max = 0;
				region.scan([&](const SubTreeRef& cur){
					max = std::max<std::size_t>(max,ptree.template getNodeRange<NodeKind,Level>(cur).getEnd().id);
				});

				// resize data storage
				data.resize(max);

			}

		private:

			MeshDataFragment(const partition_tree_type& ptree, std::vector<ElementType>&& data)
				: partitionTree(ptree), coveredRegion(SubMeshRef::root()), data(std::move(data)) {}

		public:

			MeshDataFragment(const MeshDataFragment&) = delete;
			MeshDataFragment(MeshDataFragment&&) = default;

			MeshDataFragment& operator=(const MeshDataFragment&) = delete;
			MeshDataFragment& operator=(MeshDataFragment&&) = default;


			facade_type mask() {
				return facade_type(*this);
			}

			const region_type& getCoveredRegion() const {
				return coveredRegion;
			}

			const ElementType& operator[](const NodeRef<NodeKind,Level>& id) const {
				return data[id.getOrdinal()];
			}

			ElementType& operator[](const NodeRef<NodeKind,Level>& id) {
				return data[id.getOrdinal()];
			}

			std::size_t size() const {
				return data.size();
			}

			void resize(const region_type&) {

			}

			void insert(const MeshDataFragment& other, const region_type& area) {
				assert_true(core::isSubRegion(area,other.coveredRegion)) << "New data " << area << " not covered by source of size " << coveredRegion << "\n";
				assert_true(core::isSubRegion(area,coveredRegion))       << "New data " << area << " not covered by target of size " << coveredRegion << "\n";

				assert_not_implemented();
				std::cout << core::isSubRegion(area,other.coveredRegion);

//				// copy data line by line using memcpy
//				area.scanByLines([&](const point& a, const point& b){
//					auto start = flatten(a);
//					auto length = (flatten(b) - start) * sizeof(T);
//					std::memcpy(&data[start],&other.data[start],length);
//				});
			}

			void extract(utils::ArchiveWriter&, const region_type&) const {
				assert_not_implemented();
			}

			void insert(utils::ArchiveReader&) {
				assert_not_implemented();
			}


			// -- load / store for files --

			void store(std::ostream& out) const {

				// check that the element type is a trivial type
				assert_true(std::is_trivial<ElementType>::value)
						<< "Sorry, only trivial types may be stored through this infrastructure.";

				// this fragment is required to cover the entire mesh
				assert_eq(coveredRegion, SubMeshRef::root());

				// write covered data to output stream
				utils::write<std::size_t>(out,data.size());
				utils::write(out,data.begin(),data.end());
			}

			static MeshDataFragment load(const partition_tree_type& ptree, std::istream& in) {
				// restore the data buffer
				std::size_t size = utils::read<std::size_t>(in);
				std::vector<ElementType> data(size);
				utils::read(in,data.begin(),data.end());

				// create the data fragment
				return MeshDataFragment(ptree,std::move(data));
			}

			static MeshDataFragment interpret(const partition_tree_type& ptree, utils::RawBuffer& raw) {

				// TODO: when exchanging the vector by some manageable structure, replace this
				// For now: we copy the data

				// copy the data buffer
				std::size_t size = raw.consume<std::size_t>();
				auto start = raw.consumeArray<ElementType>(size);
				std::vector<ElementType> data(start, start + size);

				// create the data fragment
				return MeshDataFragment(ptree,std::move(data));
			}

		};


		/**
		 * An entity to reference the full range of a scan. This token
		 * can not be copied and will wait for the completion of the scan upon destruction.
		 */
		class scan_reference {

			core::treeture<void> handle;

		public:

			scan_reference(core::treeture<void>&& handle)
				: handle(std::move(handle)) {}

			scan_reference() {};
			scan_reference(const scan_reference&) = delete;
			scan_reference(scan_reference&&) = default;

			scan_reference& operator=(const scan_reference&) = delete;
			scan_reference& operator=(scan_reference&&) = default;

			~scan_reference() { handle.wait(); }

			void wait() const { handle.wait(); }

		};

	} // end namespace detail

	template<
		typename NodeKind,
		typename ElementType,
		unsigned Level,
		typename PartitionTree
	>
	class MeshData : public core::data_item<detail::MeshDataFragment<NodeKind,ElementType,Level,PartitionTree>> {

		template<typename NodeKinds,typename EdgeKinds,typename Hierarchies,unsigned Levels,unsigned PartitionDepth>
		friend class Mesh;

	public:

		using node_kind = NodeKind;

		using element_type = ElementType;

		using fragment_type = detail::MeshDataFragment<NodeKind,ElementType,Level,PartitionTree>;

	private:

		std::unique_ptr<fragment_type> owned;

		fragment_type* data;


		friend fragment_type;

		MeshData(fragment_type& data) : data(&data) {}

		MeshData(std::unique_ptr<fragment_type>&& data) : owned(std::move(data)), data(owned.get()) {}

		MeshData(const PartitionTree& ptree, const detail::MeshRegion& region)
			: owned(std::make_unique<fragment_type>(ptree,region)), data(owned.get()) {}

	public:

		const ElementType& operator[](const NodeRef<NodeKind,Level>& id) const {
			return (*data)[id];
		}

		ElementType& operator[](const NodeRef<NodeKind,Level>& id) {
			return (*data)[id];
		}

		std::size_t size() const {
			return (*data).size();
		}


		void store(std::ostream& out) const {
			// ensure that the data is owned
			assert_true(owned) << "Only supported when data is owned (not managed by some Data Item Manager)";
			owned->store(out);
		}

		static MeshData load(const PartitionTree& ptree, std::istream& in) {
			return std::make_unique<fragment_type>(fragment_type::load(ptree,in));
		}

		static MeshData interpret(const PartitionTree& ptree, utils::RawBuffer& raw) {
			return std::make_unique<fragment_type>(fragment_type::interpret(ptree,raw));
		}
	};


	/**
	 * The default implementation of a mesh is capturing all ill-formed parameterizations
	 * of the mesh type to provide cleaner compiler errors.
	 */
	template<
		typename Nodes,
		typename Edges,
		typename Hierarchies,
		unsigned Levels,
		unsigned PartitionDepth
	>
	class Mesh {

		static_assert(detail::is_nodes<Nodes>::value,
				"First template argument of Mesh must be of type nodes<...>");

		static_assert(detail::is_edges<Edges>::value,
				"Second template argument of Mesh must be of type edges<...>");

		static_assert(detail::is_hierarchies<Hierarchies>::value,
				"Third template argument of Mesh must be of type hierarchies<...>");

	};


	/**
	 * The type for representing the topological information of a hierarchical mesh.
	 */
	template<
		typename ... NodeKinds,
		typename ... EdgeKinds,
		typename ... Hierarchies,
		unsigned Levels,
		unsigned PartitionDepth
	>
	class Mesh<nodes<NodeKinds...>,edges<EdgeKinds...>,hierarchies<Hierarchies...>,Levels,PartitionDepth> {

		static_assert(Levels > 0, "There must be at least one level!");

	public:

		using topology_type = detail::MeshTopologyData<nodes<NodeKinds...>,edges<EdgeKinds...>,hierarchies<Hierarchies...>,Levels>;

		using partition_tree_type = detail::PartitionTree<nodes<NodeKinds...>,edges<EdgeKinds...>,hierarchies<Hierarchies...>,Levels,PartitionDepth>;

		template<typename NodeKind,typename ValueType,unsigned Level = 0>
		using mesh_data_type = MeshData<NodeKind,ValueType,Level,partition_tree_type>;

		using builder_type = MeshBuilder<nodes<NodeKinds...>,edges<EdgeKinds...>,hierarchies<Hierarchies...>,Levels>;

		friend builder_type;

		enum { levels = Levels };

	private:

		partition_tree_type partitionTree;

		topology_type data;

		Mesh(topology_type&& data, partition_tree_type&& partitionTree)
			: partitionTree(std::move(partitionTree)), data(std::move(data)) {
			assert_true(data.isClosed());
		}

	public:

		// -- ctors / dtors / assignments --

		Mesh(const Mesh&) = delete;
		Mesh(Mesh&&) = default;

		Mesh& operator=(const Mesh&) = delete;
		Mesh& operator=(Mesh&&) = default;


		// -- provide access to components --

		const topology_type& getTopologyData() const {
			return data;
		}

		const partition_tree_type& getPartitionTree() const {
			return partitionTree;
		}

		// -- mesh querying --

		template<typename Kind,unsigned Level = 0>
		std::size_t getNumNodes() const {
			return data.template getNumNodes<Kind,Level>();
		}

		// -- mesh interactions --

		template<
			typename EdgeKind,
			typename A,
			unsigned Level,
			typename B = typename EdgeKind::trg_node_kind
		>
		NodeList<B,Level> getSinks(const NodeRef<A,Level>& a) const {
			return data.edgeSets.template getSinks<EdgeKind>(a);
		}

		template<
			typename EdgeKind,
			typename A,
			unsigned Level,
			typename B = typename EdgeKind::trg_node_kind
		>
		NodeRef<B,Level> getSink(const NodeRef<A,Level>& a) const {
			const auto& list = getSinks<EdgeKind>(a);
			assert_eq(list.size(),1);
			return list.front();
		}

		template<
			typename EdgeKind,
			typename B,
			unsigned Level,
			typename A = typename EdgeKind::src_node_kind
		>
		NodeList<A,Level> getSources(const NodeRef<B,Level>& b) const {
			return data.edgeSets.template getSources<EdgeKind>(b);
		}

		template<
			typename EdgeKind,
			typename B,
			unsigned Level,
			typename A = typename EdgeKind::src_node_kind
		>
		NodeRef<A,Level> getSource(const NodeRef<B,Level>& b) const {
			const auto& list = getSources<EdgeKind>(b);
			assert_eq(list.size(),1);
			return list.front();
		}

		// -- overloading of getNeighbor convenience functions (aliases of getSink / getSource ) --

		template<
			typename EdgeKind,
			typename A,
			unsigned Level,
			typename B = typename EdgeKind::trg_node_kind
		>
		std::enable_if_t<std::is_same<A,typename EdgeKind::src_node_kind>::value,NodeRef<B,Level>>
		getNeighbor(const NodeRef<A,Level>& a) const {
			return getSink<EdgeKind>(a);
		}

		template<
			typename EdgeKind,
			typename A,
			unsigned Level,
			typename B = typename EdgeKind::trg_node_kind
		>
		std::enable_if_t<std::is_same<A,typename EdgeKind::src_node_kind>::value,NodeList<B,Level>>
		getNeighbors(const NodeRef<A,Level>& a) const {
			return getSinks<EdgeKind>(a);
		}

		template<
			typename EdgeKind,
			typename A,
			unsigned Level,
			typename B = typename EdgeKind::src_node_kind
		>
		std::enable_if_t<std::is_same<A,typename EdgeKind::trg_node_kind>::value,NodeRef<B,Level>>
		getNeighbor(const NodeRef<A,Level>& a) const {
			return getSource<EdgeKind>(a);
		}

		template<
			typename EdgeKind,
			typename A,
			unsigned Level,
			typename B = typename EdgeKind::src_node_kind
		>
		std::enable_if_t<std::is_same<A,typename EdgeKind::trg_node_kind>::value,NodeList<B,Level>>
		getNeighbors(const NodeRef<A,Level>& a) const {
			return getSources<EdgeKind>(a);
		}

		// -- parent / children relation --

		template<
			typename Hierarchy,
			typename A, unsigned Level,
			typename B = typename Hierarchy::parent_node_kind
		>
		NodeRef<B,Level+1> getParent(const NodeRef<A,Level>& a) const {
			return data.hierarchySets.template getParent<Hierarchy,Level>(a);
		}

		template<
			typename Hierarchy,
			typename A, unsigned Level,
			typename B = typename Hierarchy::child_node_kind
		>
		NodeList<B,Level-1> getChildren(const NodeRef<A,Level>& a) const {
			return data.hierarchySets.template getChildren<Hierarchy,Level>(a);
		}

		/**
		 * A sequential operation calling the given body for each node of the given kind
		 * on the given level in parallel.
		 *
		 * NOTE: this operation is processed sequentially, and can thus not be distributed
		 * among multiple nodes. Use pforAll instead
		 *
		 * @tparam Kind the kind of node to be visited
		 * @tparam Level the level of the mesh to be addressed
		 * @tparam Body the type of operation to be applied on each node
		 *
		 * @param body the operation to be applied on each node of the selected kind and level
		 * @return a scan reference for synchronizing upon the asynchronously processed operation
		 */
		template<typename Kind, unsigned Level = 0, typename Body>
		void forAll(const Body& body) const {
			// iterate over all selected elements
			for(const auto& cur : partitionTree.template getNodeRange<Kind,Level>(detail::SubTreeRef::root())) {
				body(cur);
			}
		}

		/**
		 * A parallel operation calling the given body for each node of the given kind
		 * on the given level in parallel.
		 *
		 * This is the main operator for iterating over nodes within a mesh. All visits
		 * will always be conducted in parallel.
		 *
		 * @tparam Kind the kind of node to be visited
		 * @tparam Level the level of the mesh to be addressed
		 * @tparam Body the type of operation to be applied on each node
		 *
		 * @param body the operation to be applied on each node of the selected kind and level
		 * @return a scan reference for synchronizing upon the asynchronously processed operation
		 */
		template<typename Kind, unsigned Level = 0, typename Body>
		detail::scan_reference pforAll(const Body& body) const {

			using range = detail::SubTreeRef;

			return core::prec(
				// -- base case test --
				[](const range& a){
					// when we reached a leaf, we are at the bottom
					return a.getDepth() == PartitionDepth;
				},
				// -- base case --
				[&](const range& a){
					// apply the body to the elements of the current range
					for(const auto& cur : partitionTree.template getNodeRange<Kind,Level>(a)) {
						body(cur);
					}
				},
				// -- step case --
				core::pick(
					// -- split --
					[](const range& a, const auto& rec){
						return core::parallel(
							rec(a.getLeftChild()),
							rec(a.getRightChild())
						);
					},
					// -- serialized step case (optimization) --
					[&](const range& a, const auto&){
						// apply the body to the elements of the current range
						for(const auto& cur : partitionTree.template getNodeRange<Kind,Level>(a)) {
							body(cur);
						}
					}
				)
			)(detail::SubTreeRef::root());
		}

		template<typename Kind,	unsigned Level = 0,
				typename MapOp,
				typename ReduceOp,
				typename InitLocalState,
				typename ReduceLocalState>
		typename utils::lambda_traits<ReduceLocalState>::result_type preduce(
				const MapOp& map,
				const ReduceOp& reduce,
				const InitLocalState& init,
				const ReduceLocalState& exit) const {
			typedef typename utils::lambda_traits<ReduceLocalState>::result_type res_type;

			using range = detail::SubTreeRef;

			auto handle = [](const InitLocalState& init, const MapOp& map, const ReduceLocalState& exit, const range& a,
					const partition_tree_type& partitionTree)->res_type {
				auto res = init();
				auto mapB = [map,&res](const auto& cur) {
					return map(cur,res);
				};
				for(const auto& cur : partitionTree.template getNodeRange<Kind,Level>(a)) {
					mapB(cur);
				}
				return exit(res);
			};


			// implements a binary splitting policy for iterating over the given iterator range
			return  core::prec(
				[](const range& a) {
					return a.getDepth() == PartitionDepth;
				},
				[&](const range& a)->res_type {
					return handle(init, map, exit, a, partitionTree);
				},
				core::pick(
					[reduce](const range& a, const auto& nested) {
						// here we have the splitting
						auto left = a.getLeftChild();
						auto right = a.getRightChild();

//						return user::add(nested(left), nested(right));
						return core::combine(std::move(nested(left)),std::move(nested(right)),reduce);
					},
					[&](const range& a, const auto&)->res_type {
						return handle(init, map, exit, a, partitionTree);
					}
				)
			)(detail::SubTreeRef::root()).get();
		}

		template<typename Kind,	unsigned Level = 0,
				typename MapOp,
				typename ReduceOp,
				typename InitLocalState>
		typename utils::lambda_traits<ReduceOp>::result_type preduce(
				const MapOp& map,
				const ReduceOp& reduce,
				const InitLocalState& init) const {
			return preduce<Kind, Level>(map, reduce, init, [](typename utils::lambda_traits<ReduceOp>::result_type a) { return a; });
		}

		template<typename Kind,	unsigned Level = 0,
				typename MapOp,
				typename ReduceOp>
		typename utils::lambda_traits<ReduceOp>::result_type preduce(
				const MapOp& map,
				const ReduceOp& reduce) const {
			typedef typename utils::lambda_traits<ReduceOp>::result_type res_type;

			return preduce<Kind, Level>(map, reduce, [](){ return res_type(); }, [](res_type a) { return a; });
		}

		// -- mesh data --

		template<typename NodeKind, typename T, unsigned Level = 0>
		MeshData<NodeKind,T,Level,partition_tree_type> createNodeData() const {
			return MeshData<NodeKind,T,Level,partition_tree_type>(partitionTree,detail::SubMeshRef::root());
		}

		template<typename NodeKind, typename T, unsigned N, unsigned Level = 0>
		std::array<MeshData<NodeKind,T,Level,partition_tree_type>, N> createNodeDataArray() const {
			return utils::build_array<N>([&] { return MeshData<NodeKind,T,Level,partition_tree_type>(partitionTree,detail::SubMeshRef::root()); } );
		}

		template<typename NodeKind, typename T, unsigned Level = 0>
		MeshData<NodeKind,T,Level,partition_tree_type> loadNodeData(std::istream& in) const {
			return MeshData<NodeKind,T,Level,partition_tree_type>::load(partitionTree,in);
		}

		template<typename NodeKind, typename T, unsigned Level = 0>
		MeshData<NodeKind,T,Level,partition_tree_type> interpretNodeData(utils::RawBuffer& raw) const {
			return MeshData<NodeKind,T,Level,partition_tree_type>::interpret(partitionTree,raw);
		}


		// -- mesh property handling --

		template<typename ... Properties>
		MeshProperties<Levels,partition_tree_type,Properties...> createProperties() const {
			return MeshProperties<Levels,partition_tree_type,Properties...>(*this);
		}

		template<typename ... Properties>
		MeshProperties<Levels,partition_tree_type,Properties...> loadProperties(std::istream& in) const {
			return MeshProperties<Levels,partition_tree_type,Properties...>::load(*this,in);
		}

		template<typename ... Properties>
		MeshProperties<Levels,partition_tree_type,Properties...> interpretProperties(utils::RawBuffer& raw) const {
			return MeshProperties<Levels,partition_tree_type,Properties...>::interpret(*this,raw);
		}

		// -- load / store for files --

		void store(std::ostream& out) const {

			// write partition tree
			partitionTree.store(out);

			// write topological data
			data.store(out);

		}

		static Mesh load(std::istream& in) {

			// interpret the partition tree
			auto partitionTree = partition_tree_type::load(in);

			// load topological data
			auto topologyData = topology_type::load(in);

			// create result
			return Mesh(
				std::move(topologyData),
				std::move(partitionTree)
			);

		}

		static Mesh interpret(utils::RawBuffer& raw) {

			// interpret the partition tree
			auto partitionTree = partition_tree_type::interpret(raw);

			// load topological data
			auto topologyData = topology_type::interpret(raw);

			// create result
			return Mesh(
				std::move(topologyData),
				std::move(partitionTree)
			);

		}

	};



	/**
	 * The default implementation of a mesh build is capturing all ill-formed parameterizations
	 * of the mesh builder type to provide cleaner compiler errors.
	 */
	template<
		typename Nodes,
		typename Edges,
		typename Hierarchies,
		unsigned layers
	>
	class MeshBuilder {

		static_assert(detail::is_nodes<Nodes>::value,
				"First template argument of MeshBuilder must be of type nodes<...>");

		static_assert(detail::is_edges<Edges>::value,
				"Second template argument of MeshBuilder must be of type edges<...>");

		static_assert(detail::is_hierarchies<Hierarchies>::value,
				"Third template argument of MeshBuilder must be of type hierarchies<...>");

	};

	/**
	 * A utility to construct meshes.
	 */
	template<
		typename ... NodeKinds,
		typename ... EdgeKinds,
		typename ... Hierarchies,
		unsigned Levels
	>
	class MeshBuilder<nodes<NodeKinds...>,edges<EdgeKinds...>,hierarchies<Hierarchies...>,Levels> {

		static_assert(Levels > 0, "There must be at least one level!");

	public:

		template<unsigned PartitionDepth>
		using mesh_type = Mesh<nodes<NodeKinds...>,edges<EdgeKinds...>,hierarchies<Hierarchies...>,Levels,PartitionDepth>;

		using topology_type = detail::MeshTopologyData<nodes<NodeKinds...>,edges<EdgeKinds...>,hierarchies<Hierarchies...>,Levels>;

	private:

		topology_type data;

	public:

		// -- mesh modeling --

		template<typename Kind,unsigned Level = 0>
		NodeRef<Kind,Level> create() {
			// TODO: check that Kind is a valid node kind
			static_assert(Level < Levels, "Trying to create a node on invalid level.");
			return data.nodeSets.template create<Kind,Level>();
		}

		template<typename Kind,unsigned Level = 0>
		NodeRange<Kind,Level> create(unsigned num) {
			// TODO: check that Kind is a valid node kind
			static_assert(Level < Levels, "Trying to create a node on invalid level.");
			return data.nodeSets.template create<Kind,Level>(num);
		}

		template<typename EdgeKind, typename NodeKindA, typename NodeKindB, unsigned Level>
		void link(const NodeRef<NodeKindA,Level>& a, const NodeRef<NodeKindB,Level>& b) {
			// TODO: check that EdgeKind is a valid edge kind
			static_assert(Level < Levels, "Trying to create an edge on invalid level.");
			static_assert(std::is_same<NodeKindA,typename EdgeKind::src_node_kind>::value, "Invalid source node type");
			static_assert(std::is_same<NodeKindB,typename EdgeKind::trg_node_kind>::value, "Invalid target node type");
			return data.edgeSets.template addEdge<EdgeKind,Level>(a,b);
		}

		template<typename HierarchyKind, typename NodeKindA, typename NodeKindB, unsigned LevelA, unsigned LevelB>
		void link(const NodeRef<NodeKindA,LevelA>& parent, const NodeRef<NodeKindB,LevelB>& child) {
			// TODO: check that HierarchyKind is a valid hierarchy kind
			static_assert(LevelA == LevelB+1, "Can not connect nodes of non-adjacent levels in hierarchies");
			static_assert(LevelA < Levels, "Trying to create a hierarchical edge to an invalid level.");
			static_assert(std::is_same<NodeKindA,typename HierarchyKind::parent_node_kind>::value, "Invalid source node type");
			static_assert(std::is_same<NodeKindB,typename HierarchyKind::child_node_kind>::value, "Invalid target node type");
			return data.hierarchySets.template addChild<HierarchyKind,LevelA>(parent,child);
		}

		// -- build mesh --

		template<typename Partitioner, unsigned PartitionDepth = 0>
		mesh_type<PartitionDepth> build(const Partitioner& partitioner) const & {

			// close the topological data
			topology_type meshData = data;
			meshData.close();

			// partition the mesh
			auto partitionTree = partitioner.template partition<PartitionDepth>(meshData);

			return mesh_type<PartitionDepth>(std::move(meshData), std::move(partitionTree));
		}

		template<unsigned PartitionDepth = 0>
		mesh_type<PartitionDepth> build() const & {
			return build<detail::NaiveMeshPartitioner,PartitionDepth>(detail::NaiveMeshPartitioner());
		}


		template<typename Partitioner, unsigned PartitionDepth = 0>
		mesh_type<PartitionDepth> build(const Partitioner& partitioner) && {

			// partition the mesh
			auto partitionTree = partitioner.template partition<PartitionDepth>(data);

			return mesh_type<PartitionDepth>(std::move(data), std::move(partitionTree));
		}

		template<unsigned PartitionDepth = 0>
		mesh_type<PartitionDepth> build() const && {
			return std::move(*this).template build<detail::NaiveMeshPartitioner,PartitionDepth>(detail::NaiveMeshPartitioner());
		}

	};


	// -- Mesh Property Collections --------------------------------------


	// TODO: reduce the template instantiations complexity of this code.

	namespace detail {

		template<typename PartitionTree, unsigned Level, typename ... Properties>
		class MeshPropertiesData {

			using property_list = utils::type_list<Properties...>;

			template<typename Property>
			using mesh_data_type = MeshData<typename Property::node_kind,typename Property::value_type,Level,PartitionTree>;

			using data_t = std::tuple<mesh_data_type<Properties>...>;

			data_t data;

			MeshPropertiesData(data_t&& data) : data(std::move(data)) {}

		public:

			template<typename Mesh>
			MeshPropertiesData(const Mesh& mesh)
				: data(mesh.template createNodeData<typename Properties::node_kind,typename Properties::value_type,Level>()...) {}

			template<typename Property>
			mesh_data_type<Property>& get() {
				return std::get<utils::type_index<Property,property_list>::value>(data);
			}

			template<typename Property>
			const mesh_data_type<Property>& get() const {
				return std::get<utils::type_index<Property,property_list>::value>(data);
			}

			void store(std::ostream& out) const {
				// write property data
				utils::forEach(data,[&](const auto& entry){
					entry.store(out);
				});
			}

			template<typename Mesh>
			static MeshPropertiesData load(const Mesh& mesh, std::istream& in) {
				// a temporary tuple type to be filled with temporary results
				using tmp_data_type = std::tuple<std::unique_ptr<mesh_data_type<Properties>>...>;

				// load property data
				tmp_data_type data;
				utils::forEach(data,[&](auto& entry){
					// load data
					using data_type = typename std::remove_reference_t<decltype(entry)>::element_type;
					using node_kind = typename data_type::node_kind;
					using value_type = typename data_type::element_type;
					entry = std::make_unique<data_type>(mesh.template loadNodeData<node_kind,value_type,Level>(in));
				});

				// move data to tuple
				return MeshPropertiesData(utils::map(data,[&](auto& entry){
					return std::move(*entry.get());
				}));
			}

			template<typename Mesh>
			static MeshPropertiesData interpret(const Mesh& mesh, utils::RawBuffer& raw) {
				// a temporary tuple type to be filled with temporary results
				using tmp_data_type = std::tuple<std::unique_ptr<mesh_data_type<Properties>>...>;

				// load property data
				tmp_data_type data;
				utils::forEach(data,[&](auto& entry){
					// load data
					using data_type = typename std::remove_reference_t<decltype(entry)>::element_type;
					using node_kind = typename data_type::node_kind;
					using value_type = typename data_type::element_type;
					entry = std::make_unique<data_type>(mesh.template interpretNodeData<node_kind,value_type,Level>(raw));
				});

				// move data to tuple
				return MeshPropertiesData(utils::map(data,[&](auto& entry){
					return std::move(*entry.get());
				}));
			}

		};

		template<typename PartitionTree, unsigned Level, typename ... Properties>
		class MeshPropertiesLevels {

			template<unsigned Lvl>
			using level_data = MeshPropertiesData<PartitionTree,Lvl,Properties...>;

			using nested_level_type = MeshPropertiesLevels<PartitionTree,Level-1,Properties...>;

			level_data<Level> data;

			nested_level_type nested;

			MeshPropertiesLevels(level_data<Level>&& data, nested_level_type&& nested)
				: data(std::move(data)), nested(std::move(nested)) {}

		public:

			template<typename Mesh>
			MeshPropertiesLevels(const Mesh& mesh)
				: data(mesh), nested(mesh) {}

			template<unsigned Lvl>
			std::enable_if_t<Lvl==Level,level_data<Level>>&
			get() {
				return data;
			}

			template<unsigned Lvl>
			const std::enable_if_t<Lvl==Level,level_data<Level>>&
			get() const {
				return data;
			}

			template<unsigned Lvl>
			std::enable_if_t<Lvl!=Level,level_data<Lvl>>&
			get() {
				return nested.template get<Lvl>();
			}

			template<unsigned Lvl>
			const std::enable_if_t<Lvl!=Level,level_data<Lvl>>&
			get() const {
				return nested.template get<Lvl>();
			}

			void store(std::ostream& out) const {
				// write property data
				data.store(out);
				// write nested data
				nested.store(out);
			}


			template<typename Mesh>
			static MeshPropertiesLevels load(const Mesh& mesh, std::istream& in) {
				// load property data
				auto data = level_data<Level>::load(mesh,in);
				// load nested data
				auto nested = nested_level_type::load(mesh,in);
				// build level data
				return MeshPropertiesLevels(std::move(data),std::move(nested));
			}

			template<typename Mesh>
			static MeshPropertiesLevels interpret(const Mesh& mesh, utils::RawBuffer& raw) {
				// interpret property data
				auto data = level_data<Level>::interpret(mesh,raw);
				// interpret nested data
				auto nested = nested_level_type::interpret(mesh,raw);
				// build level data
				return MeshPropertiesLevels(std::move(data),std::move(nested));
			}

		};


		template<typename PartitionTree, typename ... Properties>
		class MeshPropertiesLevels<PartitionTree,0,Properties...> {

			using level_data = MeshPropertiesData<PartitionTree,0,Properties...>;

			level_data data;

			MeshPropertiesLevels(level_data&& data) : data(std::move(data)) {}

		public:

			template<typename Mesh>
			MeshPropertiesLevels(const Mesh& mesh)
				: data(mesh) {}

			template<unsigned Lvl>
			std::enable_if_t<Lvl==0,level_data>&
			get() {
				return data;
			}

			template<unsigned Lvl>
			const std::enable_if_t<Lvl==0,level_data>&
			get() const {
				return data;
			}

			void store(std::ostream& out) const {
				// write property data
				data.store(out);
			}

			template<typename Mesh>
			static MeshPropertiesLevels load(const Mesh& mesh, std::istream& in) {
				// load property data
				return level_data::load(mesh,in);
			}

			template<typename Mesh>
			static MeshPropertiesLevels interpret(const Mesh& mesh, utils::RawBuffer& raw) {
				// interpret property data
				return level_data::interpret(mesh,raw);
			}

		};

	}

	template<unsigned Levels, typename PartitionTree, typename ... Properties>
	class MeshProperties {

		template<typename N,typename E,typename H,unsigned L,unsigned P>
		friend class Mesh;

		using DataStore = detail::MeshPropertiesLevels<PartitionTree,Levels-1,Properties...>;

		DataStore data;

		template<typename Mesh>
		MeshProperties(const Mesh& mesh) : data(mesh) {}

		MeshProperties(DataStore&& data) : data(std::move(data)) {}

	public:

		template<typename Property, unsigned Level = 0>
		MeshData<typename Property::node_kind,typename Property::value_type,Level,PartitionTree>&
		get() {
			return data.template get<Level>().template get<Property>();
		}

		template<typename Property, unsigned Level = 0>
		const MeshData<typename Property::node_kind,typename Property::value_type,Level,PartitionTree>&
		get() const {
			return data.template get<Level>().template get<Property>();
		}

		template<typename Property, unsigned Level = 0>
		typename Property::value_type& get(const NodeRef<typename Property::node_kind,Level>& node) {
			return get<Property,Level>()[node];
		}

		template<typename Property, unsigned Level = 0>
		const typename Property::value_type& get(const NodeRef<typename Property::node_kind,Level>& node) const {
			return get<Property,Level>()[node];
		}

		// -- load / store for files --

		void store(std::ostream& out) const {
			// write property data
			data.store(out);
		}

		template<typename Mesh>
		static MeshProperties load(const Mesh& mesh, std::istream& in) {
			// forward call to data store
			return MeshProperties(DataStore::load(mesh,in));
		}

		template<typename Mesh>
		static MeshProperties interpret(const Mesh& mesh, utils::RawBuffer& raw) {
			// forward call to data store
			return MeshProperties(DataStore::interpret(mesh,raw));
		}
	};

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
