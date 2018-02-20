#pragma once

#include <array>
#include <algorithm>
#include <tuple>

#include "allscale/utils/printer/arrays.h"
#include "allscale/utils/assert.h"
#include "allscale/utils/unused.h"
#include "allscale/utils/serializer/arrays.h"

namespace allscale {
namespace utils {

	// generic vector implementation
	template<typename T, std::size_t Dims>
	class Vector {

		std::array<T, Dims> data;

	public:

		using element_type = T;

		Vector() = default;

		Vector(const T& e) {
			data.fill(e);
		}

		Vector(const Vector&) = default;
		Vector(Vector&&) = default;

		template<typename R>
		Vector(const Vector<R,Dims>& other)
			: data(other.data) {}

		template<typename R>
		Vector(const std::array<R,Dims>& other)
			: data(other) {}

		Vector(const std::initializer_list<T>& values) {
			assert_eq(Dims,values.size());
			init(values);
		}

		template<typename ... Rest>
		Vector(T a, T b, Rest ... rest) : data{ {a,b,rest...} } {
			static_assert(Dims == sizeof...(rest)+2, "Invalid number of components!");
		}


		Vector& operator=(const Vector& other) = default;
		Vector& operator=(Vector&& other) = default;

		T& operator[](const std::size_t index) {
			return data[index];
		}

		const T& operator[](const std::size_t index) const {
			return data[index];
		}

		// relational operators
		// defined in-class, since the private std::array data member has matching operators to forward to

		bool operator==(const Vector& other) const {
			return data == other.data;
		}

		bool operator!=(const Vector& other) const {
			return !(data == other.data);
		}

		bool operator<(const Vector& other) const {
			return data < other.data;
		}

		bool operator<=(const Vector& other) const {
			return data <= other.data;
		}

		bool operator>=(const Vector& other) const {
			return data >= other.data;
		}

		bool operator>(const Vector& other) const {
			return data > other.data;
		}

		// allow implicit casts to std::array
		operator const std::array<T, Dims>&() const { return data; }

		bool dominatedBy(const Vector<T,Dims>& other) const {
			for(std::size_t i=0; i<Dims; i++) {
				if (other[i] < data[i]) return false;
			}
			return true;
		}

		bool strictlyDominatedBy(const Vector<T,Dims>& other) const {
			for(std::size_t i=0; i<Dims; i++) {
				if (other[i] <= data[i]) return false;
			}
			return true;
		}

		// Adds printer support to this vector.
		friend std::ostream& operator<<(std::ostream& out, const Vector& vec) {
			return out << vec.data;
		}

	private:

		template<typename R, std::size_t ... Index>
		void init_internal(const std::initializer_list<R>& list, const std::integer_sequence<std::size_t,Index...>&) {
			__allscale_unused auto bla = { data[Index] = *(list.begin() + Index) ... };
		}

		template<typename R>
		void init(const std::initializer_list<R>& list) {
			init_internal(list,std::make_index_sequence<Dims>());
		}

	};

	template<typename T, std::size_t Dims, typename S>
	Vector<T,Dims>& operator+=(Vector<T,Dims>& a, const Vector<S,Dims>& b) {
		for(std::size_t i = 0; i<Dims; i++) {
			a[i] += b[i];
		}
		return a;
	}

	template<typename T, std::size_t Dims, typename S>
	Vector<T,Dims>& operator-=(Vector<T,Dims>& a, const Vector<S,Dims>& b) {
		for(size_t i = 0; i<Dims; i++) {
			a[i] -= b[i];
		}
		return a;
	}

	template<typename T, std::size_t Dims, typename S>
	Vector<T,Dims>& operator*=(Vector<T,Dims>& a, const S& fac) {
		for(size_t i =0; i<Dims; i++) {
			a[i] *= fac;
		}
		return a;
	}

	template<typename T, std::size_t Dims, typename S>
	Vector<T,Dims>& operator/=(Vector<T,Dims>& a, const S& fac) {
		for(size_t i =0; i<Dims; i++) {
			a[i] /= fac;
		}
		return a;
	}


	template<typename T, std::size_t Dims>
	Vector<T,Dims> operator+(const Vector<T,Dims>& a, const Vector<T,Dims>& b) {
		Vector<T,Dims> res(a);
		return res += b;
	}

	template<typename T, std::size_t Dims>
	Vector<T,Dims> operator-(const Vector<T, Dims>& a, const Vector<T, Dims>& b) {
		Vector<T,Dims> res(a);
		return res -= b;
	}

	template<typename T, std::size_t Dims, typename S>
	Vector<T,Dims> operator*(const Vector<T, Dims>& vec, const S& fac) {
		Vector<T,Dims> res(vec);
		return res *= fac;
	}

	template<typename T, std::size_t Dims, typename S>
	Vector<T, Dims> operator*(const S& fac, const Vector<T, Dims>& vec) {
		return vec * fac;
	}

	template<typename T, std::size_t Dims, typename S>
	Vector<T,Dims> operator/(const Vector<T, Dims>& vec, const S& fac) {
		Vector<T,Dims> res(vec);
		return res /= fac;
	}

	template<typename T, std::size_t Dims, typename Lambda>
	Vector<T,Dims> elementwise(const Vector<T,Dims>& a, const Vector<T,Dims>& b, const Lambda& op) {
		Vector<T,Dims> res;
		for(unsigned i=0; i<Dims; i++) {
			res[i] = op(a[i],b[i]);
		}
		return res;
	}

	template<typename T, std::size_t Dims>
	Vector<T,Dims> elementwiseMin(const Vector<T,Dims>& a, const Vector<T,Dims>& b) {
		return elementwise(a,b,[](const T& a, const T& b) { return std::min<T>(a,b); });
	}

	template<typename T, std::size_t Dims>
	Vector<T,Dims> elementwiseMax(const Vector<T,Dims>& a, const Vector<T,Dims>& b) {
		return elementwise(a,b,[](const T& a, const T& b) { return std::max<T>(a,b); });
	}

	template<typename T, std::size_t Dims>
	Vector<T,Dims> elementwiseProduct(const Vector<T,Dims>& a, const Vector<T,Dims>& b) {
		return elementwise(a,b,[](const T& a, const T& b) { return a*b; });
	}

	template<typename T, std::size_t Dims>
	Vector<T,Dims> elementwiseDivision(const Vector<T,Dims>& a, const Vector<T,Dims>& b) {
		return elementwise(a,b,[](const T& a, const T& b) { return a/b; });
	}

	template<typename T, std::size_t Dims>
	Vector<T,Dims> elementwiseRemainder(const Vector<T,Dims>& a, const Vector<T,Dims>& b) {
		return elementwise(a,b,[](const T& a, const T& b) { return a % b; });
	}

	template<typename T, std::size_t Dims>
	Vector<T,Dims> elementwiseModulo(const Vector<T,Dims>& a, const Vector<T,Dims>& b) {
		return elementwiseRemainder(a,b);
	}


	template<typename T, std::size_t Dims>
	T sumOfSquares(const Vector<T,Dims>& vec) {
		T sum = T();
		for(unsigned i = 0; i < Dims; i++) {
			sum += vec[i] * vec[i];
		}
		return sum;
	}

	// specialization for 3-dimensional vectors, providing access to named data members x, y, z
	template <typename T>
	class Vector<T, 3> {
	public:

		using element_type = T;

		T x, y, z;

		Vector() = default;

		Vector(const T& e) : x(e), y(e), z(e) { }

		Vector(T x, T y, T z) : x(x), y(y), z(z) { }

		Vector(const Vector&) = default;
		Vector(Vector&&) = default;

		template<typename R>
		Vector(const Vector<R,3>& other) : x(other.x), y(other.y), z(other.z) {}

		template<typename R>
		Vector(const std::array<R,3>& other) : x(other[0]), y(other[1]), z(other[2]) {}

		T& operator[](std::size_t i) {
			return (i==0) ? x : (i==1) ? y : z;
		}

		const T& operator[](std::size_t i) const {
			return (i==0) ? x : (i==1) ? y : z;
		}

		Vector& operator=(const Vector& other) = default;
		Vector& operator=(Vector&& other) = default;

		bool operator==(const Vector& other) const {
			return std::tie(x,y,z) == std::tie(other.x,other.y,other.z);
		}

		bool operator!=(const Vector& other) const {
			return !(*this == other);
		}

		bool operator<(const Vector& other) const {
			return asArray() < other.asArray();
		}

		bool operator<=(const Vector& other) const {
			return asArray() <= other.asArray();
		}

		bool operator>=(const Vector& other) const {
			return asArray() >= other.asArray();
		}

		bool operator>(const Vector& other) const {
			return asArray() > other.asArray();
		}

		operator const std::array<T, 3>&() const { return asArray(); }

		const std::array<T,3>& asArray() const {
			return reinterpret_cast<const std::array<T,3>&>(*this);
		}

		bool dominatedBy(const Vector& other) const {
			return other.x >= x && other.y >= y && other.z >= z;
		}

		bool strictlyDominatedBy(const Vector& other) const {
			return other.x > x && other.y > y && other.z > z;
		}

		// Adds printer support to this vector.
		friend std::ostream& operator<<(std::ostream& out, const Vector& vec) {
			return out << "[" << vec.x << "," << vec.y << "," << vec.z << "]";
		}

	};

	template<typename T>
	Vector<T, 3> crossProduct(const Vector<T, 3>& a, const Vector<T, 3>& b) {
		return Vector<T, 3> {
			a[1] * b[2] - a[2] * b[1],
			a[2] * b[0] - a[0] * b[2],
			a[0] * b[1] - a[1] * b[0]
		};
	}

	// specialization for 2-dimensional vectors, providing access to named data members x, y
	template <typename T>
	class Vector<T, 2> {
	public:

		using element_type = T;

		T x, y;

		Vector() = default;

		Vector(const T& e) : x(e), y(e) { }

		Vector(T x, T y) : x(x), y(y) { }

		Vector(const Vector&) = default;
		Vector(Vector&&) = default;

		template<typename R>
		Vector(const Vector<R,2>& other) : x(other.x), y(other.y) {}

		template<typename R>
		Vector(const std::array<R,2>& other) : x(other[0]), y(other[1]) {}

		T& operator[](std::size_t i) {
			return (i == 0) ? x : y;
		}

		const T& operator[](std::size_t i) const {
			return (i == 0) ? x : y;
		}

		Vector& operator=(const Vector& other) = default;
		Vector& operator=(Vector&& other) = default;

		bool operator==(const Vector& other) const {
			return asArray() == other.asArray();
		}

		bool operator!=(const Vector& other) const {
			return !(*this == other);
		}

		bool operator<(const Vector& other) const {
			return asArray() < other.asArray();
		}

		bool operator<=(const Vector& other) const {
			return asArray() <= other.asArray();
		}

		bool operator>=(const Vector& other) const {
			return asArray() >= other.asArray();
		}

		bool operator>(const Vector& other) const {
			return asArray() > other.asArray();
		}

		operator const std::array<T, 2>&() const { return asArray(); }

		const std::array<T,2>& asArray() const {
			return reinterpret_cast<const std::array<T,2>&>(*this);
		}

		bool dominatedBy(const Vector& other) const {
			return other.x >= x && other.y >= y;
		}

		bool strictlyDominatedBy(const Vector& other) const {
			return other.x > x && other.y > y;
		}

		// Adds printer support to this vector.
		friend std::ostream& operator<<(std::ostream& out, const Vector& vec) {
			return out << "[" << vec.x << "," << vec.y << "]";
		}

	};

	/**
	 * Add support for serializing / de-serializing Vector instances.
	 * The implementation is simply re-using the serializing capabilities of arrays.
	 */
	template<typename T, std::size_t Dims>
	struct serializer<Vector<T,Dims>,typename std::enable_if<is_serializable<T>::value,void>::type> : public serializer<std::array<T,Dims>> {};

} // end namespace utils
} // end namespace allscale
