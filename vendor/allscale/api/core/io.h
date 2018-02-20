#pragma once

#include <string>

#include "allscale/api/core/impl/reference/io.h"
#include "allscale/utils/serializer.h"

namespace allscale {
namespace api {
namespace core {


	// ----------------------------------------------------------------------
	//							   Declarations
	// ----------------------------------------------------------------------


	/**
	 * Supported IO modes for stream based operations.
	 * @see http://en.cppreference.com/w/cpp/io/c#Binary_and_text_modes
	 */
	enum class Mode {
		Text, Binary
	};

	/**
	 * An abstraction for a file or buffer to read/write from.
	 */
	class Entry;

	/**
	 * An out-of-order stream for reading information from a file/buffer previously
	 * written using an output stream.
	 */
	class InputStream;

	/**
	 * An out-of-order stream for writing information to some file/buffer.
	 */
	class OutputStream;

	/**
	 * A utility for reading the content of a storage entity (e.g. a file) through
	 * memory mapped IO.
	 */
	class MemoryMappedInput;

	/**
	 * A utility for reading and writing the content of a storage entity (e.g. a file) through
	 * memory mapped IO.
	 */
	class MemoryMappedOutput;

	/**
	 * An IO manager for in-memory data buffer manipulations.
	 */
	class BufferIOManager;

	/**
	 * An IO manager providing access to the file system.
	 */
	class FileIOManager;



	// ----------------------------------------------------------------------
	//							   Definitions
	// ----------------------------------------------------------------------


	// -- Stream Based IO ---------------------------------------------------

	/**
	 * A converter between this interface and the reference implementation
	 */
	inline impl::reference::Mode toRefMode(Mode mode) {
		switch(mode) {
		case Mode::Text: 	return impl::reference::Mode::Text;
		case Mode::Binary: 	return impl::reference::Mode::Binary;
		}
		assert_fail() << "Invalid mode encountered!";
		return {};
	}

	class Entry {

		friend InputStream;

		friend OutputStream;

		friend MemoryMappedInput;

		friend MemoryMappedOutput;

		template<typename StorageManager>
		friend class IOManager;

		using RefEntry = impl::reference::Entry;

		// the wrapped up reference implementation
		RefEntry entry;

		// the constructor is private to restrict creation to the corresponding factories
		Entry(const RefEntry& entry) : entry(entry) {}

	};


	/**
	 * A stream to read data from some entry of an IO manager.
	 */
	class InputStream {

		template<typename StreamFactory>
		friend class IOManager;

		using RefInStream = impl::reference::InputStream;

		// the wrapped up reference implementation
		RefInStream& istream;

		InputStream(RefInStream& istream)
			: istream(istream) {}

	public:

		InputStream(const InputStream&) = delete;
		InputStream(InputStream&&) = default;

		/**
		 * Obtains the entry this stream is associated to.
		 */
		Entry getEntry() const {
			return istream.getEntry();
		}

		/**
		 * Provides atomic access to this stream, allowing the given body to
		 * to perform a sequence of read operations without potential interference
		 * of other threads.
		 */
		template<typename Body>
		InputStream& atomic(const Body& body) {
			istream.atomic(body);
			return *this;
		}

		/**
		 * Reads a single instance of the given type (atomic).
		 */
		template<typename T>
		T read() {
			return istream.read<T>();
		}

		/**
		 * An idiomatic overload of the read operation.
		 */
		template<typename T>
		InputStream& operator>>(T& trg) {
			istream >> trg;
			return *this;
		}

		/**
		 * Allows to test whether this stream is in a valid state. It can, for instance,
		 * be utilized to determine whether there has been an error during the last
		 * performed operation or whether in text mode the end of a file has been reached.
		 */
		operator bool() const {
			return istream;
		}

		// -- make it serializable --

		static InputStream load(utils::ArchiveReader& a) {
			return { RefInStream::load(a) };
		}

		void store(utils::ArchiveWriter& a) const {
			istream.store(a);
		}
	};



	/**
	 * A stream to write data to some entry of an IO manager.
	 */
	class OutputStream {

		template<typename StorageManager>
		friend class IOManager;

		using RefOutStream = impl::reference::OutputStream;

		RefOutStream& ostream;

		OutputStream(RefOutStream& ostream)
			: ostream(ostream) {}

	public:

		OutputStream(const OutputStream&) = delete;
		OutputStream(OutputStream&&) = default;

		/**
		 * Obtains the entry this stream is associated to.
		 */
		Entry getEntry() const {
			return ostream.getEntry();
		}

		/**
		 * Provides atomic access to this stream, allowing the given body to
		 * to perform a sequence of write operations without potential interference
		 * of other threads.
		 */
		template<typename Body>
		OutputStream& atomic(const Body& body) {
			ostream.atomic(body);
			return *this;
		}

		/**
		 * Writes a single instance of the given type (atomic).
		 */
		template<typename T>
		OutputStream& write(const T& value) {
			ostream.write(value);
			return *this;
		}

		/**
		 * An idiomatic overload of the write operation.
		 */
		template<typename T>
		OutputStream& operator<<(const T& value) {
			ostream << value;
			return *this;
		}
		OutputStream& operator<<(const char* value) {
			ostream << value;
			return *this;
		}

		/**
		 * Allows to test whether this stream is in a valid state. It can, for instance,
		 * be utilized to determine whether there has been an error during the last
		 * performed operation.
		 */
		operator bool() const {
			return ostream;
		}

		// -- make it serializable --

		static OutputStream load(utils::ArchiveReader& a) {
			return { RefOutStream::load(a) };
		}

		void store(utils::ArchiveWriter& a) const {
			ostream.store(a);
		}
	};





	// -- Memory Mapped IO --------------------------------------------------


	/**
	 * A utility for reading the content of a storage entity (e.g. a file) through
	 * memory mapped IO.
	 */
	class MemoryMappedInput {

		template<typename StorageManager>
		friend class IOManager;

		using Impl = impl::reference::MemoryMappedInput;

		Impl impl;

		MemoryMappedInput(Impl&& impl) : impl(impl) {}

	public:

		/**
		 * The identifier for the underlying storage entity.
		 */
		Entry getEntry() const {
			return impl.getEntry();
		}

		/**
		 * Provides access to the underlying data by interpreting it
		 * as an instance of type T.
		 */
		template<typename T>
		const T& access() const {
			return impl.access<T>();
		}

		/**
		 * Provides access to the underlying data by interpreting it
		 * as an array of instances of type T.
		 */
		template<typename T>
		const T* accessArray() const {
			return &access<T>();
		}

		// -- make it serializable --

		static MemoryMappedInput load(utils::ArchiveReader& a) {
			return { Impl::load(a) };
		}

		void store(utils::ArchiveWriter& a) const {
			impl.store(a);
		}
	};

	/**
	 * A utility for reading and writing the content of a storage entity (e.g. a file) through
	 * memory mapped IO.
	 */
	class MemoryMappedOutput {

		template<typename StorageManager>
		friend class IOManager;

		using Impl = impl::reference::MemoryMappedOutput;

		Impl impl;

		MemoryMappedOutput(Impl&& impl) : impl(impl) {}

	public:

		/**
		 * The identifier for the underlying storage entity.
		 */
		Entry getEntry() const {
			return impl.getEntry();
		}

		/**
		 * Provides access to the underlying data by interpreting it
		 * as an instance of type T.
		 */
		template<typename T>
		T& access() const {
			return impl.access<T>();
		}

		/**
		 * Provides access to the underlying data by interpreting it
		 * as an array of instances of type T.
		 */
		template<typename T>
		T* accessArray() const {
			return &access<T>();
		}

		// -- make it serializable --

		static MemoryMappedOutput load(utils::ArchiveReader& a) {
			return { Impl::load(a) };
		}

		void store(utils::ArchiveWriter& a) const {
			impl.store(a);
		}
	};



	// -- IO Manager --------------------------------------------------------

	/**
	 * An IO manager, as the central dispatcher for IO operations.
	 */
	template<typename StorageManager>
	class IOManager {

		using Impl = impl::reference::IOManager<StorageManager>;

		Impl impl;

	public:

		/**
		 * Creates a new entry with the given name in the underlying storage system.
		 *
		 * @param name the name of the entry (e.g. file)
		 * @param mode whether it is a binary or text file
		 * @return a entry ID referencing the newly created resource
		 */
		Entry createEntry(const std::string& name, Mode mode = Mode::Text) {
			return impl.createEntry(name, toRefMode(mode));
		}

		/**
		 * Register a new output stream with the given name within the system.
		 * The call will create the underlying file and prepare output operations.
		 *
		 *  NOTE: this method is not thread safe!
		 *
		 * @param entry the name of the stream to be opened -- nothing happens if already opened
		 */
		InputStream openInputStream(Entry entry) {
			return InputStream(impl.openInputStream(entry.entry));
		}

		/**
		 * Register a new output stream with the given name within the system.
		 * The call will create the underlying file and prepare output operations.
		 *
		 *  NOTE: this method is not thread safe!
		 *
		 * @param entry the name of the stream to be opened -- nothing happens if already opened
		 */
		OutputStream openOutputStream(Entry entry) {
			return OutputStream(impl.openOutputStream(entry.entry));
		}

		/**
		 * Register a new memory mapped input with the given name within the system.
		 * The call will load the underlying storage and prepare input operations.
		 *
		 *  NOTE: this method is not thread safe!
		 *
		 * @param entry the storage entry to be opened -- nothing happens if already opened
		 */
		MemoryMappedInput openMemoryMappedInput(Entry entry) {
			return MemoryMappedInput(impl.openMemoryMappedInput(entry.entry));
		}

		/**
		 * Register a new memory mapped output with the given name within the system.
		 * The call will create the underlying storage and prepare output operations.
		 *
		 *  NOTE: this method is not thread safe!
		 *
		 * @param entry the storage entry to be opened -- nothing happens if already opened
		 */
		MemoryMappedOutput openMemoryMappedOutput(Entry entry, std::size_t size) {
			return MemoryMappedOutput(impl.openMemoryMappedOutput(entry.entry,size));
		}

		/**
		 * Obtains an input stream to read data from a storage entry.
		 * The storage entry is maintained by the manager and the provided output stream
		 * is only valid within the current thread.
		 *
		 * @param entry the name of the storage entry to be targeted -- must be open
		 * @return a stream to append data to
		 */
		InputStream getInputStream(Entry entry) {
			return InputStream(impl.getInputStream(entry.entry));
		}

		/**
		 * Obtains an output stream to write data to a storage entry.
		 * The storage entry is maintained by the manager and the provided output stream
		 * is only valid within the current thread.
		 *
		 * @param entry the name of the storage entry to be targeted -- must be open
		 * @return a stream to append data to
		 */
		OutputStream getOutputStream(Entry entry) {
			return OutputStream(impl.getOutputStream(entry.entry));
		}

		/**
		 * Obtains a memory mapped input to read data from a storage entry.
		 * The storage entry is maintained by the manager and the provided memory mapped
		 * input is only valid within the current thread.
		 *
		 * @param entry the name of the storage entry to be targeted -- must be open
		 * @return a requested memory mapped input
		 */
		MemoryMappedInput getMemoryMappedInput(Entry entry) {
			return MemoryMappedInput(impl.getMemoryMappedInput(entry));
		}

		/**
		 * Obtains a memory mapped output to write data to a storage entry.
		 * The storage entry is maintained by the manager and the provided memory mapped
		 * output is only valid within the current thread.
		 *
		 * @param entry the name of the storage entry to be targeted -- must be open
		 * @return a requested memory mapped output
		 */
		MemoryMappedOutput getMemoryMappedOutput(Entry entry, std::size_t size) {
			return MemoryMappedOutput(impl.getMemoryMappedOutput(entry,size));
		}

		/**
		 * Closes the given stream.
		 */
		void close(const InputStream& in) {
			impl.close(in.istream);
		}

		/**
		 * Closes the given stream.
		 */
		void close(const OutputStream& out) {
			impl.close(out.ostream);
		}

		/**
		 * Closes the given memory mapped entry.
		 */
		void close(const MemoryMappedInput& in) {
			impl.close(in.impl);
		}

		/**
		 * Closes the given memory mapped entry.
		 */
		void close(const MemoryMappedOutput& out) {
			impl.close(out.impl);
		}

		/**
		 * Determines whether the given entry exists.
		 */
		bool exists(Entry entry) const {
			return impl.exists(entry.entry);
		}

		/**
		 * Deletes the entry with the given name.
		 */
		void remove(Entry entry) {
			impl.remove(entry.entry);
		}

	};

	// Definition of the BufferIOManager
	class BufferIOManager : public IOManager<core::impl::reference::BufferStorageFactory> {

	};

	// Definition of the FileIOManager
	class FileIOManager : public IOManager<core::impl::reference::FileStorageFactory> {

		/**
		 * Make constructor private to avoid instances.
		 */
		FileIOManager() {}

	public:

		/**
		 * Provide access to the singleton instance.
		 */
		static FileIOManager& getInstance() {
			static FileIOManager mgr;
			return mgr;
		}

	};

} // end namespace core
} // end namespace api
} // end namespace allscale

