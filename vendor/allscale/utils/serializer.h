#pragma once

#include <cstring>
#include <sstream>
#include <type_traits>
#include <vector>

#include "allscale/utils/assert.h"

#if defined(ALLSCALE_WITH_HPX)
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/array.hpp>
#endif

namespace allscale {
namespace utils {

	// ---------------------------------------------------------------------------------
	//									Declarations
	// ---------------------------------------------------------------------------------

	/**
	 * An archive contains the serialized version of some data structure (fragment).
	 * It enables the exchange of data between e.g. address spaces.
	 */
	class Archive;

	/**
	 * An archive writer is a builder for archives. It is utilized for serializing objects.
	 */
	class ArchiveWriter;

	/**
	 * An archive reader is a utility to reconstruct data structures from archives.
	 */
	class ArchiveReader;

	/**
	 * A serializer describes the way types are converted to and restored from archives.
	 */
	template<typename T, typename _ = void>
	struct serializer;

	/**
	 * This type trait can be utilized to test whether a given type is serializable,
	 * thus packable into an archive, or not.
	 */
	template <typename T, typename _ = void>
	struct is_serializable;

	/**
	 * A facade function for packing an object into an archive.
	 */
	template<typename T>
	typename std::enable_if<is_serializable<T>::value,Archive>::type
	serialize(const T&);

	/**
	 * A facade function for unpacking an object from an archive.
	 */
	template<typename T>
	typename std::enable_if<is_serializable<T>::value,T>::type
	deserialize(Archive&);


	// ---------------------------------------------------------------------------------
	//									Definitions
	// ---------------------------------------------------------------------------------


	namespace detail {

		/**
		 * A simple, initial, functionally complete implementation of a data buffer
		 * for storing data within an archive.
		 */
		class DataBuffer {

			// check some underlying assumption
			static_assert(sizeof(char)==1, "If a char is more than a byte, this implementation needs to be checked.");

			// the actual data store (std::vector handles the dynamic growing for us)
			std::vector<char> data;

		public:

			DataBuffer() {}

			DataBuffer(const DataBuffer&) = default;
			DataBuffer(DataBuffer&&) = default;

			DataBuffer(const std::vector<char>& data) : data(data) {}
			DataBuffer(std::vector<char>&& data) : data(std::move(data)) {}

			DataBuffer& operator=(const DataBuffer&) = default;
			DataBuffer& operator=(DataBuffer&&) = default;

			/**
			 * The main function for appending data to this buffer.
			 */
			void append(const char* start, std::size_t count) {
				// create space
				auto pos = data.size();
				data.resize(pos + count / sizeof(char));

				// append at end
				std::memcpy(&data[pos],start,count);

			}

			/**
			 * Obtains the number of bytes this buffer is occupying.
			 */
			std::size_t size() const {
				return data.size() * sizeof(char);
			}

			/**
			 * Obtains a pointer to the begin of the internally maintained buffer (inclusive).
			 */
			const char* begin() const {
				return &data.front();
			}

			/**
			 * Obtains a pointer to the end of the internally maintained buffer (exclusive).
			 */
			const char* end() const {
				return &data.back() + 1;
			}

			/**
			 * Support implicit conversion of this buffer to a vector of characters.
			 */
			operator const std::vector<char>&() const {
				return data;
			}

			/**
			 * Also enable the implicit hand-off of the ownership of the underlying char store.
			 */
			operator std::vector<char>() && {
				return std::move(data);
			}


		};

	} // end namespace detail


	class Archive {

		friend class ArchiveWriter;
		friend class ArchiveReader;

		// the data represented by this archive
		detail::DataBuffer data;

		Archive(detail::DataBuffer&& data)
			: data(std::move(data)) {}

	public:



		Archive(const Archive&) = default;
		Archive(Archive&&) = default;

		Archive(const std::vector<char>& buffer) : data(buffer) {}
		Archive(std::vector<char>&& buffer) : data(std::move(buffer)) {}

		Archive& operator=(const Archive&) = default;
		Archive& operator=(Archive&&) = default;

		/**
		 * Support implicit conversion of this archive to a vector of characters.
		 */
		operator const std::vector<char>&() const {
			return data;
		}

		/**
		 * Also enable the implicit hand-off of the ownership of the underlying buffer.
		 */
		operator std::vector<char>() && {
			return std::move(data);
		}

		/**
		 * Provide explicit access to the underlying char buffer.
		 */
		const std::vector<char>& getBuffer() const {
			return data;
		}
	};

#if !defined(ALLSCALE_WITH_HPX)
	class ArchiveWriter {

		// the buffer targeted by this archive writer
		detail::DataBuffer data;

	public:

		ArchiveWriter() {}

		ArchiveWriter(const ArchiveWriter&) = delete;
		ArchiveWriter(ArchiveWriter&&) = default;

		ArchiveWriter& operator=(const ArchiveWriter&) = delete;
		ArchiveWriter& operator=(ArchiveWriter&&) = default;

		/**
		 * Appends a given number of bytes to the end of the underlying data buffer.
		 */
		void write(const char* src, std::size_t count) {
			data.append(src,count);
		}

		/**
		 * A utility function wrapping the invocation of the serialization mechanism.
		 */
		template<typename T>
		std::enable_if_t<is_serializable<T>::value,void>
		write(const T& value) {
			// use serializer to store object of this type
			serializer<T>::store(*this,value);
		}

		/**
		 * Obtains the archive produces by this writer. After the call,
		 * this writer must not be used any more.
		 */
		Archive toArchive() && {
			return std::move(data);
		}

	};
#else
    class ArchiveWriter {
        hpx::serialization::output_archive &ar_;

    public:
        ArchiveWriter(hpx::serialization::output_archive &ar) : ar_(ar) {}

        /**
		 * Appends a given number of bytes to the end of the underlying data buffer.
		 */
		void write(const char* src, std::size_t count) {
            ar_ & hpx::serialization::make_array(src, count);
		}

		/**
		 * A utility function wrapping the invocation of the serialization mechanism.
		 */
		template<typename T>
		std::enable_if_t<is_serializable<T>::value,void>
		write(const T& value) {
// 			// use serializer to store object of this type
			serializer<T>::store(*this,value);
		}

        template<typename T>
		std::enable_if_t<!is_serializable<T>::value,void>
		write(const T& value) {
            ar_ & value;
		}
    };
#endif

#if !defined(ALLSCALE_WITH_HPX)
	class ArchiveReader {

		// the current point of the reader
		const char* cur;

		// the end of the reader (only checked for debugging)
		const char* end;

	public:

		/**
		 * A archive reader can only be obtained from an existing archive.
		 */
		ArchiveReader(const Archive& archive)
			: cur(archive.data.begin()), end(archive.data.end()) {}

		ArchiveReader(const ArchiveReader&) = delete;
		ArchiveReader(ArchiveReader&&) = default;

		ArchiveReader& operator=(const ArchiveReader&) = delete;
		ArchiveReader& operator=(ArchiveReader&&) = default;

		/**
		 * Reads a number of bytes from the underlying buffer.
		 */
		void read(char* dst, std::size_t count) {
			// copy the data
			std::memcpy(dst,cur,count);
			// move pointer forward
			cur += count;

			// make sure that we did not cross the end of the buffer
			assert_le(cur,end);
		}

		/**
		 * A utility function wrapping up the de-serialization of an object
		 * of type T from the underlying buffer.
		 */
		template<typename T>
		std::enable_if_t<is_serializable<T>::value,T>
		read() {
			// use serializer to restore object of this type
			return serializer<T>::load(*this);
		}

	};
#else
	class ArchiveReader {
        hpx::serialization::input_archive &ar_;

    public:
        ArchiveReader(hpx::serialization::input_archive &ar) : ar_(ar) {}

		/**
		 * Reads a number of bytes from the underlying buffer.
		 */
		void read(char* dst, std::size_t count) {
            ar_ & hpx::serialization::make_array(dst, count);
		}

		/**
		 * A utility function wrapping up the de-serialization of an object
		 * of type T from the underlying buffer.
		 */
		template<typename T>
		std::enable_if_t<is_serializable<T>::value,T>
		read() {
			// use serializer to restore object of this type
			return serializer<T>::load(*this);
		}

		template<typename T>
		std::enable_if_t<!is_serializable<T>::value,T>
		read() {
			// use serializer to restore object of this type
            T t;
            ar_ & t;
            return t;
		}
	};
#endif


	/**
	 * Adds support for the serialization to every type T supporting
	 *
	 * 	- a static member function  T load(ArchiveReader&)
	 * 	- a member function         void store(ArchiveWriter&)
	 *
	 * Thus, serialization / deserialization can be integrated through member functions.
	 */
	template<typename T>
	struct serializer<T, typename std::enable_if<
			// targeted types have to offer a static load member ...
			std::is_same<decltype(T::load(std::declval<ArchiveReader&>())),T>::value &&
			// ... and a store member function
			std::is_same<decltype(std::declval<const T&>().store(std::declval<ArchiveWriter&>())),void>::value,
		void>::type> {

		static T load(ArchiveReader& a) {
			return T::load(a);
		}
		static void store(ArchiveWriter& a, const T& value) {
			value.store(a);
		}
	};


	/**
	 * Enables the skipping of const qualifiers for types.
	 * Also const values can be serialized and deserialized if requested.
	 */
	template<typename T>
	struct serializer<const T,typename std::enable_if<
			is_serializable<T>::value,
		void>::type> : public serializer<T> {};



	// -- primitive type serialization --

	namespace detail {

		/**
		 * A helper functor for serializing primitive types.
		 */
		template<typename T>
		struct primitive_serializer {
			static T load(ArchiveReader& reader) {
				T res = 0;
				reader.read(reinterpret_cast<char*>(&res),sizeof(T));
				return res;
			}
			static void store(ArchiveWriter& writer, const T& value) {
				writer.write(reinterpret_cast<const char*>(&value),sizeof(T));
			}
		};

	} // end namespace detail

	template<> struct serializer<bool>          :  public detail::primitive_serializer<bool> {};

	template<> struct serializer<char>          :  public detail::primitive_serializer<char> {};
	template<> struct serializer<signed char>   :  public detail::primitive_serializer<signed char> {};
	template<> struct serializer<unsigned char> :  public detail::primitive_serializer<unsigned char> {};
	template<> struct serializer<char16_t>      :  public detail::primitive_serializer<char16_t> {};
	template<> struct serializer<char32_t>      :  public detail::primitive_serializer<char32_t> {};
	template<> struct serializer<wchar_t>       :  public detail::primitive_serializer<wchar_t> {};

	template<> struct serializer<short int>     :  public detail::primitive_serializer<short int> {};
	template<> struct serializer<int>           :  public detail::primitive_serializer<int> {};
	template<> struct serializer<long int>      :  public detail::primitive_serializer<long int> {};
	template<> struct serializer<long long int> :  public detail::primitive_serializer<long long int> {};

	template<> struct serializer<unsigned short int>     :  public detail::primitive_serializer<unsigned short int> {};
	template<> struct serializer<unsigned int>           :  public detail::primitive_serializer<unsigned int> {};
	template<> struct serializer<unsigned long int>      :  public detail::primitive_serializer<unsigned long int> {};
	template<> struct serializer<unsigned long long int> :  public detail::primitive_serializer<unsigned long long int> {};

	template<> struct serializer<float>       :  public detail::primitive_serializer<float> {};
	template<> struct serializer<double>      :  public detail::primitive_serializer<double> {};
	template<> struct serializer<long double> :  public detail::primitive_serializer<long double> {};


	template <typename T, typename _>
	struct is_serializable : public std::false_type {};

	template <typename T>
	struct is_serializable<T, typename std::enable_if<
			std::is_same<decltype((T(*)(Archive&))(&serializer<T>::load)), T(*)(Archive&)>::value &&
			std::is_same<decltype((void(*)(Archive&, const T&))(&serializer<T>::store)), void(*)(Archive&, const T&)>::value,
		void>::type> : public std::true_type {};



	// -- facade functions --
#if !defined(ALLSCALE_WITH_HPX)
	template<typename T>
	typename std::enable_if<is_serializable<T>::value,Archive>::type
	serialize(const T& value) {
		ArchiveWriter writer;
		writer.write(value);
		return std::move(writer).toArchive();
	}

	template<typename T>
	typename std::enable_if<is_serializable<T>::value,T>::type
	deserialize(Archive& a) {
		return ArchiveReader(a).read<T>();
	}
#endif

} // end namespace utils
} // end namespace allscale

#if defined(ALLSCALE_WITH_HPX)
namespace hpx {
namespace serialization {
    template <typename T>
    typename std::enable_if<
        ::allscale::utils::is_serializable<T>::value &&
        !(std::is_integral<T>::value || std::is_floating_point<T>::value),
        output_archive&
    >::type
    serialize(output_archive & ar, T const & t, int) {
        allscale::utils::ArchiveWriter writer(ar);
        writer.write(t);
        return ar;
    }

    template <typename T>
    typename std::enable_if<
        ::allscale::utils::is_serializable<T>::value &&
        !(std::is_integral<T>::value || std::is_floating_point<T>::value),
        input_archive&
    >::type
    serialize(input_archive & ar, T & t, int) {

        allscale::utils::ArchiveReader reader(ar);
        t = reader.read<T>();
        return ar;
    }
} // end namespace serialization
} // end namespace allscale
#endif
