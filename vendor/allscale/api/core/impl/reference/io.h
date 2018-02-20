#pragma once

#include <map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/types.h>

#ifdef _MSC_VER
	// includes
	#include <io.h>
	// marcos for function identifiers
	#define CLOSE_WRAPPER _close
	#define LSEEK_WRAPPER _lseek
	#define OPEN_WRAPPER _open
	#define READ_WRAPPER _read
	#define WRITE_WRAPPER _write
	// macros for flags
	#define S_IRUSR _S_IREAD
	#define S_IWUSR _S_IWRITE
#else
	// includes
	#include <sys/mman.h>
	#include <unistd.h>
	// marcos for function identifiers
	#define CLOSE_WRAPPER close
	#define LSEEK_WRAPPER lseek
	#define OPEN_WRAPPER open
	#define READ_WRAPPER read
	#define WRITE_WRAPPER write
#endif

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include "allscale/utils/assert.h"
#include "allscale/utils/serializer.h"


namespace allscale {
namespace api {
namespace core {
namespace impl {
namespace reference {

	/**
	 * Supported IO modes.
	 */
	enum class Mode {
		Text, Binary
	};

	/**
	 * The kind of handle to reference entities within an IO manager.
	 */
	struct Entry {
		std::size_t id;
		bool operator<(const Entry& other) const { return id < other.id; }
	};

	/**
	 * A common base class for Input and Output Streams.
	 */
	class IOStream {
	protected:

		Entry entry;

		std::mutex operation_lock;

		IOStream(const Entry& entry) : entry(entry) {}

		IOStream(IOStream&& other)
			: entry(other.entry) {}

	public:

		Entry getEntry() const {
			return entry;
		}

	};

	/**
	 * A stream to load data in the form of a stream of entries.
	 */
	class InputStream : public IOStream {

		template<typename Factory>
		friend class IOManager;

	public:
		struct IStreamWrapper {
			std::istream& in;
			IStreamWrapper(std::istream& in) : in(in) {}
			template<typename T>
			IStreamWrapper& operator>>(T& value) {
				in >> value;
				return *this;
			}
			template<typename T>
			T read() {
				T value;
				in.read((char*)&value, sizeof(T));
				return value;
			}
			template<typename T>
			IStreamWrapper& read(T& res) {
				in.read((char*)&res, sizeof(T));
				return *this;
			}
		};

	private:
		IStreamWrapper in;

		InputStream(const Entry& entry, std::istream& in)
			: IOStream(entry), in(in) {}

	public:

		InputStream(InputStream&& other)
			: IOStream(std::move(other)), in(other.in) {}

		template<typename Body>
		void atomic(const Body& body) {
			// protect output by locking it
			std::lock_guard<std::mutex> lease(operation_lock);

			// let the body read it's information
			body(in);

			// free the lock - automatically
		}

		template<typename T>
		void operator>>(T& value) {
			atomic([&](IStreamWrapper& in) { in >> value; });
		}

		template<typename T>
		T read() {
			T res;
			atomic([&](IStreamWrapper& in) {
				res = in.read<T>();
			});
			return res;
		}

		operator bool() const {
			return (bool)in.in;
		}

		static InputStream& load(utils::ArchiveReader&) {
			assert_not_implemented();
			exit(1); // prevent return warning
		}

		void store(utils::ArchiveWriter&) const {
			assert_not_implemented();
		}
	};

	/**
	 * A stream to store data in the form of a stream of entries.
	 */
	class OutputStream : public IOStream {

		template<typename Factory>
		friend class IOManager;

	public:
		struct OStreamWrapper {
			std::ostream& out;
			OStreamWrapper(std::ostream& out) : out(out) {}
			template<typename T>
			OStreamWrapper& operator<<(const T& value) {
				out << value;
				return *this;
			}
			OStreamWrapper& operator<<(const char* value) {
				out << value;
				return *this;
			}
			template<typename T>
			OStreamWrapper& write(const T& value) {
				out.write((char*)&value, sizeof(T));
				return *this;
			}
		};

	private:
		OStreamWrapper out;

		OutputStream(const Entry& entry, std::ostream& out)
			: IOStream(entry), out(out) {}

	public:

		OutputStream(OutputStream&& other)
			: IOStream(std::move(other)), out(other.out) {}

		template<typename Body>
		void atomic(const Body& body) {
			// protect output by locking it
			std::lock_guard<std::mutex> lease(operation_lock);

			// let the body write it's information
			body(out);

			// free the lock - automatically
		}

		template<typename T>
		void operator<<(const T& value) {
			atomic([&](OStreamWrapper& out) {
				out << value;
			});
		}
		void operator<<(const char* value) {
			atomic([&](OStreamWrapper& out) {
				out << value;
			});
		}

		template<typename T>
		void write(const T& value) {
			atomic([&](OStreamWrapper& out) {
				out.write(value);
			});
		}

		operator bool() const {
			return (bool)out.out;
		}

		static OutputStream& load(utils::ArchiveReader&) {
			assert_not_implemented();
			exit(1); // prevent return warning
		}

		void store(utils::ArchiveWriter&) const {
			assert_not_implemented();
		}
	};




	class MemoryMappedIO {

		Entry entry;

		void* base;

	public:

		MemoryMappedIO(const Entry& entry, void* base)
			: entry(entry), base(base) {}

		Entry getEntry() const {
			return entry;
		}

	protected:

		void* getBase() const {
			return base;
		}

	};

	class MemoryMappedInput : public MemoryMappedIO {

		template<typename Factory>
		friend class IOManager;

		MemoryMappedInput(const Entry& entry, void* base)
			: MemoryMappedIO(entry,base) {}

	public:

		template<typename T>
		const T& access() const {
			return *static_cast<const T*>(getBase());
		}

		// -- make it serializable --

		static MemoryMappedInput load(utils::ArchiveReader&) {
			assert_not_implemented();
			exit(1); // prevent return warning
		}

		void store(utils::ArchiveWriter&) const {
			assert_not_implemented();
		}
	};

	class MemoryMappedOutput : public MemoryMappedIO {

		template<typename Factory>
		friend class IOManager;

		MemoryMappedOutput(const Entry& entry, void* base)
			: MemoryMappedIO(entry,base) {}

	public:

		template<typename T>
		T& access() const {
			return *static_cast<T*>(getBase());
		}

		// -- make it serializable --

		static MemoryMappedOutput load(utils::ArchiveReader&) {
			assert_not_implemented();
			exit(1); // prevent return warning
		}

		void store(utils::ArchiveWriter&) const {
			assert_not_implemented();
		}
	};

	/**
	 * An IO manager, as the central dispatcher for IO operations.
	 */
	template<typename StorageManager>
	class IOManager {

		/**
		 * The underlying store.
		 */
		StorageManager store;

		/**
		 * The central register of all open output streams.
		 */
		std::map<Entry,InputStream> inputStreams;

		/**
		 * The central register of all open output streams.
		 */
		std::map<Entry,OutputStream> outputStreams;

		/**
		 * The central register of all open memory mapped inputs.
		 */
		std::map<Entry,MemoryMappedInput> memoryMappedInputs;

		/**
		 * The central register of all open memory mapped outputs.
		 */
		std::map<Entry,MemoryMappedOutput> memoryMappedOutputs;

	public:

		~IOManager() {
			// close and destroy all input streams
			for(auto& cur : inputStreams) {
				closeStream(cur.second);
			}
			// close and destroy all output streams
			for(auto& cur : outputStreams) {
				closeStream(cur.second);
			}
			// close and destroy all memory mapped inputs
			for(auto& cur : memoryMappedInputs) {
				closeMemoryMappedIO(cur.second);
			}
			// close and destroy all memory mapped outputs
			for(auto& cur : memoryMappedOutputs) {
				closeMemoryMappedIO(cur.second);
			}
		}

		/**
		 * Creates a new entry with the given name in the underlying storage system.
		 *
		 * @param name the name of the entry (e.g. file)
		 * @param mode whether it is a binary or text file
		 * @return a entry ID referencing the newly created resource
		 */
		Entry createEntry(const std::string& name, Mode mode = Mode::Text) {
			return store.createEntry(name, mode);
		}

		/**
		 * Register a new output stream with the given name within the system.
		 * The call will create the underlying file and prepare output operations.
		 *
		 *  NOTE: this method is not thread safe!
		 *
		 * @param entry the name of the stream to be opened -- nothing happens if already opened
		 */
		InputStream& openInputStream(Entry entry) {

			// check for present
			auto pos = inputStreams.find(entry);
			if (pos != inputStreams.end()) return pos->second;

			// create new input stream
			InputStream res(entry, *store.createInputStream(entry));

			// register stream
			inputStreams.emplace(entry, std::move(res));

			// return result
			return getInputStream(entry);
		}

		/**
		 * Register a new output stream with the given name within the system.
		 * The call will create the underlying file and prepare output operations.
		 *
		 *  NOTE: this method is not thread safe!
		 *
		 * @param entry the name of the stream to be opened -- nothing happens if already opened
		 */
		OutputStream& openOutputStream(Entry entry) {

			// check for present
			auto pos = outputStreams.find(entry);
			if (pos != outputStreams.end()) return pos->second;

			// create new input stream
			OutputStream res(entry, *store.createOutputStream(entry));

			// register stream
			outputStreams.emplace(entry, std::move(res));

			// return result
			return getOutputStream(entry);
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

			// check for present
			auto pos = memoryMappedInputs.find(entry);
			if (pos != memoryMappedInputs.end()) return pos->second;

			// create new input stream
			MemoryMappedInput res(entry, store.createMemoryMappedInput(entry));

			// register stream
			memoryMappedInputs.emplace(entry, std::move(res));

			// return result
			return getMemoryMappedInput(entry);
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

			// check for present
			auto pos = memoryMappedOutputs.find(entry);
			if (pos != memoryMappedOutputs.end()) return pos->second;

			// create new input stream
			MemoryMappedOutput res(entry, store.createMemoryMappedOutput(entry,size));

			// register stream
			memoryMappedOutputs.emplace(entry, std::move(res));

			// return result
			return getMemoryMappedOutput(entry);
		}


		/**
		 * Obtains an input stream to read data from a storage entry.
		 * The storage entry is maintained by the manager and the provided output stream
		 * is only valid within the current thread.
		 *
		 * @param entry the name of the storage entry to be targeted -- must be open
		 * @return a stream to append data to
		 */
		InputStream& getInputStream(Entry entry) {
			assert_true(inputStreams.find(entry) != inputStreams.end());
			return inputStreams.find(entry)->second;
		}

		/**
		 * Obtains an output stream to write data to a storage entry.
		 * The storage entry is maintained by the manager and the provided output stream
		 * is only valid within the current thread.
		 *
		 * @param entry the name of the storage entry to be targeted -- must be open
		 * @return a stream to append data to
		 */
		OutputStream& getOutputStream(Entry entry) {
			assert_true(outputStreams.find(entry) != outputStreams.end());
			return outputStreams.find(entry)->second;
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
			assert_true(memoryMappedInputs.find(entry) != memoryMappedInputs.end());
			return memoryMappedInputs.find(entry)->second;
		}

		/**
		 * Obtains a memory mapped output to write data to a storage entry.
		 * The storage entry is maintained by the manager and the provided memory mapped
		 * output is only valid within the current thread.
		 *
		 * @param entry the name of the storage entry to be targeted -- must be open
		 * @return a requested memory mapped output
		 */
		MemoryMappedOutput getMemoryMappedOutput(Entry entry) {
			assert_true(memoryMappedOutputs.find(entry) != memoryMappedOutputs.end());
			return memoryMappedOutputs.find(entry)->second;
		}

		/**
		 * Closes the stream with the given name.
		 */
		void closeInputStream(Entry entry) {
			// get the stream
			auto pos = inputStreams.find(entry);
			if (pos == inputStreams.end()) return;

			// close the stream
			closeStream(pos->second);

			// erase the entry
			inputStreams.erase(pos);
		}

		/**
		 * Closes the stream with the given name.
		 */
		void closeOutputStream(Entry entry) {
			// get the stream
			auto pos = outputStreams.find(entry);
			if (pos == outputStreams.end()) return;

			// close the stream
			closeStream(pos->second);

			// erase the entry
			outputStreams.erase(pos);
		}

		/**
		 * Closes the given stream.
		 */
		void close(const InputStream& in) {
			closeInputStream(in.getEntry());
		}

		/**
		 * Closes the given stream.
		 */
		void close(const OutputStream& out) {
			closeOutputStream(out.getEntry());
		}

		/**
		 * Closes the given memory mapped input.
		 */
		void close(const MemoryMappedInput& in) {
			auto pos = memoryMappedInputs.find(in.getEntry());
			if (pos == memoryMappedInputs.end()) return;

			// remove memory mapping
			closeMemoryMappedIO(in);

			// erase entry from register
			memoryMappedInputs.erase(pos);
		}

		/**
		 * Closes the given memory mapped output.
		 */
		void close(const MemoryMappedOutput& out) {
			auto pos = memoryMappedOutputs.find(out.getEntry());
			if (pos == memoryMappedOutputs.end()) return;

			// remove memory mapping
			closeMemoryMappedIO(out);

			// erase entry from register
			memoryMappedOutputs.erase(pos);
		}

		/**
		 * Determines whether the given entry exists.
		 */
		bool exists(Entry entry) const {
			return store.exists(entry);
		}

		/**
		 * Deletes the entry with the given name.
		 */
		void remove(Entry entry) {
			store.remove(entry);
		}

	private:

		/**
		 * Closes the given input stream.
		 */
		void closeStream(InputStream& in) {
			// closes the stream
			store.close(in.in.in);
		}

		/**
		 * Closes the given output stream.
		 */
		void closeStream(OutputStream& out) {
			// closes the stream
			store.close(out.out.out);
		}

		/**
		 * Close the given memory mapped IO connection.
		 */
		void closeMemoryMappedIO(const MemoryMappedInput& input) {
			// closes the memory mapped input
			store.close(input);
		}

		/**
		* Close the given memory mapped IO connection.
		*/
		void closeMemoryMappedIO(const MemoryMappedOutput& output) {
			// closes the memory mapped output
			store.close(output);
		}

	};



	// ----------------------------------------------------------------------
	//				    for in-memory buffer operations
	// ----------------------------------------------------------------------


	struct BufferStorageFactory {

		struct Buffer {
			std::string name;
			Mode mode;
			std::stringstream* stream;
		};

		struct MemoryMappedBuffer {
			std::size_t size;
			void* base;
		};

		std::size_t counter = 0;

		std::map<Entry, Buffer> buffers;

		std::map<Entry,MemoryMappedBuffer> memoryMappedBuffers;

		~BufferStorageFactory() {
			for(const auto& cur : buffers) delete cur.second.stream;
			for(const auto& cur : memoryMappedBuffers) free(cur.second.base);
		}

		Entry createEntry(const std::string& name, Mode mode) {
			// check for present entry
			for(const auto& cur : buffers) {
				if (cur.second.name == name) {
					return cur.first;
				}
			}

			// create a new entry
			Entry id{counter++};
			Buffer& entry = buffers[id];
			entry.name = name;
			entry.mode = mode;
			entry.stream = nullptr;
			return id;
		}

		std::istream* createInputStream(Entry entry) {

			// search for entry
			auto pos = buffers.find(entry);
			if (pos == buffers.end()) {
				assert_fail() << "Unable to create input stream to unknown entity!";
				return nullptr;
			}


			// reuse current stream content
			std::stringstream* old = pos->second.stream;
			std::stringstream* res = (pos->second.mode == Mode::Binary) ?
					new std::stringstream((old) ? old->str() : std::basic_string<char>(), std::ios_base::in | std::ios_base::binary ) :
					new std::stringstream((old) ? old->str() : std::basic_string<char>(), std::ios_base::in );
			delete old;
			pos->second.stream = res;
			return res;
		}

		std::ostream* createOutputStream(Entry entry) {

			// search for entry
			auto pos = buffers.find(entry);
			if (pos == buffers.end()) {
				assert_fail() << "Unable to create output stream to unknown entity!";
				return nullptr;
			}

			// reuse current stream content
			std::stringstream* old = pos->second.stream;
			std::stringstream* res = (pos->second.mode == Mode::Binary) ?
					new std::stringstream((old) ? old->str() : std::basic_string<char>(), std::ios_base::out | std::ios_base::binary ) :
					new std::stringstream((old) ? old->str() : std::basic_string<char>(), std::ios_base::out );
			delete old;
			pos->second.stream = res;
			return res;
		}

		void* createMemoryMappedInput(const Entry& entry) {
			// the target buffer needs to be present
			auto pos = memoryMappedBuffers.find(entry);
			if (pos == memoryMappedBuffers.end()) return nullptr;
			return pos->second.base;
		}

		void* createMemoryMappedOutput(const Entry& entry, std::size_t size) {
			// check whether there is already such a buffer
			auto pos = memoryMappedBuffers.find(entry);
			if (pos != memoryMappedBuffers.end()) {
				// use existing
				assert_eq(size,pos->second.size) << "Cannot change size of buffer during re-opening!";
				return pos->second.base;
			}

			// create a new buffer
			auto& buffer = memoryMappedBuffers[entry];
			buffer.size = size;
			buffer.base = std::malloc(size);
			return buffer.base;
		}

		void close(const MemoryMappedIO&) {
			// nothing to do
		}

		void close(std::istream&) {
			// nothing to do
		}

		void close(std::ostream&) {
			// nothing to do
		}

		bool exists(Entry entry) const {
			return buffers.find(entry) != buffers.end();
		}

		void remove(Entry entry) {
			auto pos = buffers.find(entry);
			if (pos == buffers.end()) return;
			delete pos->second.stream;
			buffers.erase(pos);
		}
	};

	class BufferIOManager : public IOManager<BufferStorageFactory> {

	};


	// ----------------------------------------------------------------------
	//					  	  for file IO
	// ----------------------------------------------------------------------

	struct FileStorageFactory {

		using file_descriptor = int;

		struct File {
			// general
			std::string name;
			Mode mode;

			// for memory-mapped files
			file_descriptor fd;
			std::size_t size;
			void* base;

			File(const std::string& name, Mode mode)
				: name(name), mode(mode), fd(0), size(0), base(nullptr) {}

		};

		std::vector<File> files;

		Entry createEntry(const std::string& name, Mode mode) {
			// check for present entry
			for(std::size_t i=0; i < files.size(); ++i) {
				if (files[i].name == name) return Entry{i};
			}

			// create a new entry
			Entry id{files.size()};
			files.push_back(File(name,mode));
			return id;
		}

		std::istream* createInputStream(Entry entry) {

			// check valid entry id
			if (entry.id >= files.size()) {
				assert_fail() << "Unable to create input stream to unknown entity!";
				return nullptr;
			}

			// create a matching file stream
			const File& file = files[entry.id];
			return (file.mode == Mode::Binary) ?
				new std::fstream(file.name,std::ios_base::in | std::ios_base::binary) :
				new std::fstream(file.name,std::ios_base::in);
		}

		std::ostream* createOutputStream(Entry entry) {

			// check valid entry id
			if (entry.id >= files.size()) {
				assert_fail() << "Unable to create output stream to unknown entity!";
				return nullptr;
			}

			// create a matching file stream
			const File& file = files[entry.id];
			return (file.mode == Mode::Binary) ?
				new std::fstream(file.name,std::ios_base::out | std::ios_base::binary) :
				new std::fstream(file.name,std::ios_base::out);
		}

		void* createMemoryMappedInput(const Entry& entry) {

			// get a reference to the covered file
			File& file = getFile(entry);

			// check that file is not already mapped
			assert_true(file.base==nullptr)
				<< "Error: file already previously opened!";

			// get the file descriptor
			file.fd = getFileDescriptor(file,true);

			// resolve the file size
			file.size = getFileSize(file);

#ifndef _MSC_VER
			// map file into address space
			file.base = mmap(nullptr,file.size, PROT_READ, MAP_PRIVATE, file.fd, 0);
			// check result of mmap
			if (!checkMappedAddress(file.base)) file.base = nullptr;
#else
			// if no support for memory mapped io, try to read the entire file into a buffer
			file.base = malloc(file.size);
			auto bytesRead = READ_WRAPPER(file.fd, file.base, (unsigned)file.size);
			if (bytesRead < 0) {
				free(file.base);
				file.base = nullptr;
			}
#endif

			// return pointer to base address
			return file.base;
		}

		void* createMemoryMappedOutput(const Entry& entry, std::size_t size) {

			// get a reference to the covered file
			File& file = getFile(entry);

			// check that file is not already mapped
			assert_true(file.base==nullptr)
				<< "Error: file already previously opened!";

			// get the file descriptor
			file.fd = createFile(file,size);

			// fix the file size
			file.size = size;

#ifndef _MSC_VER
			// map file into address space
			file.base = mmap(nullptr,file.size, PROT_READ | PROT_WRITE, MAP_SHARED, file.fd, 0);
			// check result of mmap
			if (!checkMappedAddress(file.base)) file.base = nullptr;
#else
			file.base = malloc(size);
#endif

			// return pointer to base address
			return file.base;
		}

		void close(std::istream& stream) {
			delete &stream;
		}

		void close(std::ostream& stream) {
			delete &stream;
		}

		void close(const MemoryMappedInput& mmi) {
			close(mmi, false);
		}

		void close(const MemoryMappedOutput& mmo) {
			close(mmo, true);
		}

		bool exists(Entry entry) const {
			if (entry.id >= files.size()) return false;
			struct stat buffer;
			return stat(files[entry.id].name.c_str(), &buffer) == 0;
		}

		void remove(Entry entry) {
			if (entry.id >= files.size()) return;
			std::remove(files[entry.id].name.c_str());
		}

	private:

		File& getFile(const Entry& entry) {

			// check valid entry id
			if (entry.id >= files.size()) {
				assert_fail() << "Unknown file entry: " << entry.id;
				return files[0];
			}

			// provide access
			return files[entry.id];
		}

		static file_descriptor createFile(const File& file, std::size_t size) {

			// create the new file
			auto fd = OPEN_WRAPPER(file.name.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR );
			assert_ne(-1,fd) << "Error creating file " << file.name;

			// fix size of file
			LSEEK_WRAPPER(fd,(long)(size-1),SEEK_SET);

			// write a byte at the end
			char data = 0;
			auto res = WRITE_WRAPPER(fd,&data,1);
			assert_eq(1,res) << "Could not write byte at end of file.";
			if (res != 1) return 0;

			// move cursor back to start
			LSEEK_WRAPPER(fd,0,SEEK_SET);

			// return file descriptor
			return fd;
		}

		static file_descriptor getFileDescriptor(const File& file, bool readOnly) {

			// get the register entry
			if (file.fd > 0) return file.fd;

			// get name of file
			const char* name = file.name.c_str();

			// get file descriptor from file name
			auto fd = OPEN_WRAPPER(name, ((readOnly) ? O_RDONLY : O_RDWR ) );
			assert_ne(-1,fd) << "Error opening file " << name;

			// return the obtained file descriptor
			return fd;

		}

		static std::size_t getFileSize(const File& file) {

			// get size of file
			struct stat fileStat;
			auto succ = stat(file.name.c_str(),&fileStat);
			assert_eq(0,succ) << "Unable to obtain size of input file: " << file.name;

			if (succ != 0) return 0;

			// get the file size
			return fileStat.st_size;
		}

		static bool checkMappedAddress(void* addr) {
#ifndef _MSC_VER
			// compare with error token
			if (addr != MAP_FAILED) return true;
			char buffer[2000];
			std::cout << strerror_r(errno,buffer,2000);
#endif
			// fail with message if mapping failed
			// or if mapped address checking was requested on MSVC platforms
			assert_fail() << "Failed to map file into address space!";
			return false;
		}

		void close(const MemoryMappedIO& mmio, bool requiresWrite) {

			auto entry = mmio.getEntry();

			// check valid entry id
			if (entry.id >= files.size()) {
				assert_fail() << "Unable to close memory mapped input to unknown entity!";
				return;
			}

			// get the register entry
			File& file = files[entry.id];
			if (!file.base) return;

			int succ = 0;
#ifndef _MSC_VER
			// unmap the file from the address space
			succ = munmap(file.base, file.size);
			assert_eq(0, succ)
				<< "Unable to unmap file " << file.name;
			// if it was not successful, stop it here
			if (succ != 0) return;
			// silence unused parameter warning
			(void)requiresWrite;
#else
			// if no support for memory mapped io, just write full buffer contents to file and free buffer
			if (requiresWrite) {
				auto bytesWritten = WRITE_WRAPPER(file.fd, file.base, (unsigned)file.size);
				free(file.base);
				assert_le(0, bytesWritten)
					<< "Unable to write to file " << file.name << ", " << strerror(errno) << " " << file.fd;
			}
#endif

			// close the file descriptor
			succ = ::CLOSE_WRAPPER(file.fd);
			assert_eq(0, succ) << "Unable to close file " << file.name;

			// reset the file descriptor
			file.fd = 0;

			// reset the base pointer
			file.base = nullptr;

		}

	};

	class FileIOManager : public IOManager<FileStorageFactory> {
		FileIOManager() {};
	public:
		static FileIOManager& getInstance() {
			static FileIOManager manager;
			return manager;
		}
	};


	/**
	 * Obtains access to the singleton instance of the File IO manager.
	 */
	inline static FileIOManager& getFileIOManager() {
		return FileIOManager::getInstance();
	}


} // end namespace reference
} // end namespace impl
} // end namespace core
} // end namespace api
} // end namespace allscale
