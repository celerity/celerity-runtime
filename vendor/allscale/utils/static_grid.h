#pragma once

#include <cstring>
#include <type_traits>

#include "allscale/utils/functional_utils.h"
#include "allscale/utils/serializer.h"
#include "allscale/utils/vector.h"

namespace allscale {
namespace utils {

	template<typename Cell, size_t ... size>
	struct StaticGrid;

	template<typename Cell, size_t a, size_t ... rest>
	struct StaticGrid<Cell,a,rest...> {
		using data_type = std::array<StaticGrid<Cell,rest...>,a>;
		using addr_type = utils::Vector<int64_t,sizeof...(rest)+1>;

	private:

		data_type data;

		template<typename T>
		typename std::enable_if<std::is_trivially_copyable<T>::value,void>::type
		assignInternal(const StaticGrid& other) {
			std::memcpy(&data,&other.data,sizeof(data_type));
		}

		template<typename T>
		typename std::enable_if<!std::is_trivially_copyable<T>::value,void>::type
		assignInternal(const StaticGrid& other) {
			data = other.data;
		}

	public:

		StaticGrid& operator=(const StaticGrid& other) {
			if (this == &other) return *this;
			assignInternal<Cell>(other);
			return *this;
		}

		Cell& operator[](const addr_type& addr) {
			return this->template operator[]<sizeof...(rest)+1>(addr);
		}

		const Cell& operator[](const addr_type& addr) const {
			return this->template operator[]<sizeof...(rest)+1>(addr);
		}

		template<std::size_t D>
		Cell& operator[](const utils::Vector<int64_t,D>& addr) {
			allscale_check_bounds((size_t)addr[D - sizeof...(rest)-1], data);
			return data[addr[D-sizeof...(rest)-1]][addr];
		}

		template<std::size_t D>
		const Cell& operator[](const utils::Vector<int64_t,D>& addr) const {
			allscale_check_bounds((size_t)addr[D - sizeof...(rest)-1], data);
			return data[addr[D-sizeof...(rest)-1]][addr];
		}

		utils::Vector<std::size_t, sizeof...(rest)+1> size() const {
			return { a, rest... };
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 1, void>
		forEach(const Lambda& lambda) const {
			for(const auto& cur : data) {
				cur.forEach(lambda);
			}
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 1, void>
		forEach(const Lambda& lambda) {
			for(auto& cur : data) {
				cur.forEach(lambda);
			}
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		forEach(const Lambda& lambda) const {
			addr_type pos;
			_forEachInternal(pos,lambda);
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		forEach(const Lambda& lambda) {
			addr_type pos;
			_forEachInternal(pos,lambda);
		}

		void store(utils::ArchiveWriter& writer) const {
			for(const auto& e : data) {
				writer.write(e);
			}
		}

		static StaticGrid load(utils::ArchiveReader& reader) {
			StaticGrid grid;
			for(auto& e : grid.data) {
				e = reader.read<typename data_type::value_type>();
			}
			return grid;
		}

	private:

		template<typename T, size_t ... s>
		friend struct StaticGrid;

		template<typename Lambda, std::size_t D>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		_forEachInternal(utils::Vector<int64_t,D>& pos, const Lambda& lambda) const {
			auto& i = pos[D-sizeof...(rest)-1];
			i = 0;
			for(const auto& cur : data) {
				cur._forEachInternal(pos,lambda);
				i++;
			}
		}

		template<typename Lambda, std::size_t D>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		_forEachInternal(utils::Vector<int64_t,D>& pos, const Lambda& lambda) {
			auto& i = pos[D-sizeof...(rest)-1];
			i = 0;
			for(auto& cur : data) {
				cur._forEachInternal(pos,lambda);
				i++;
			}
		}

	};

	template<typename Cell>
	struct StaticGrid<Cell> {
		using data_type = Cell;
		using addr_type = utils::Vector<int64_t, 0>;

	private:

		data_type data;

		template<typename T>
		typename std::enable_if<std::is_trivially_copyable<T>::value,void>::type
		assignInternal(const StaticGrid& other) {
			std::memcpy(&data,&other.data,sizeof(data_type));
		}

		template<typename T>
		typename std::enable_if<!std::is_trivially_copyable<T>::value,void>::type
		assignInternal(const StaticGrid& other) {
			data = other.data;
		}

	public:

		StaticGrid& operator=(const StaticGrid& other) {
			if (this == &other) return *this;
			assignInternal<Cell>(other);
			return *this;
		}

		Cell& operator[](const addr_type& addr) {
			return this->template operator[]<0>(addr);
		}

		const Cell& operator[](const addr_type& addr) const {
			return this->template operator[]<0>(addr);
		}

		template<std::size_t D>
		Cell& operator[](const utils::Vector<int64_t,D>&) {
			return data;
		}

		template<std::size_t D>
		const Cell& operator[](const utils::Vector<int64_t,D>&) const {
			return data;
		}

		std::size_t size() const {
			return 1;
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 1, void>
		forEach(const Lambda& lambda) const {
			lambda(data);
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 1, void>
		forEach(const Lambda& lambda) {
			lambda(data);
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		forEach(const Lambda& lambda) const {
			lambda(addr_type(),data);
		}

		template<typename Lambda>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		forEach(const Lambda& lambda) {
			lambda(addr_type(),data);
		}

		void store(utils::ArchiveWriter& writer) const {
			writer.write(data);
		}

		static StaticGrid load(utils::ArchiveReader& reader) {
			StaticGrid grid;
			grid.data = std::move(reader.read<data_type>());
			return grid;
		}

	private:

		template<typename T, size_t ... s>
		friend struct StaticGrid;

		template<typename Lambda, std::size_t D>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		_forEachInternal(utils::Vector<int64_t,D>& pos, const Lambda& lambda) const {
			lambda(const_cast<const utils::Vector<int64_t,D>&>(pos),data);
		}

		template<typename Lambda, std::size_t D>
		std::enable_if_t<lambda_traits<Lambda>::arity == 2, void>
		_forEachInternal(utils::Vector<int64_t,D>& pos, const Lambda& lambda) {
			lambda(const_cast<const utils::Vector<int64_t,D>&>(pos),data);
		}

	};

} // end utils
} // end namespace allscale
