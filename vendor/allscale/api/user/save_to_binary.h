#pragma once

#include "allscale/api/core/io.h"
#include "allscale/api/user/algorithm/pfor.h"


namespace allscale {
namespace api {
namespace user {

// Save vector of vectors to binary in parallel
template<typename T>
void saveVecVecToFile(std::vector<std::vector<T>> vecVec, std::string filename, size_t innerSize) {
	core::FileIOManager& manager = core::FileIOManager::getInstance();
	size_t outerSize = vecVec.size();

	// generate output data
	core::Entry binary = manager.createEntry(filename, core::Mode::Binary);
	auto fout = manager.openOutputStream(binary);

//	fout.write(innerSize);

	std::vector<size_t> idxVec;
	for(size_t i = 0; i < innerSize; ++i)
		idxVec.push_back(i);

	algorithm::pfor(idxVec, [&](size_t& i) {
		fout.atomic([&](auto& out) {
			// write preamble
			out.write(i);

			// write data
			for(size_t j = 0; j < outerSize; ++j) {
				out.write(vecVec[j][i]);
			}
		});
	});

	manager.close(fout);

}

template<typename T>
void saveVecVecToFileMM(std::vector<std::vector<T>> vecVec, std::string filename, unsigned outerSize, unsigned innerSize) {
	core::FileIOManager& manager = core::FileIOManager::getInstance();

	// generate output data
	core::Entry binary = manager.createEntry(filename, core::Mode::Binary);
	core::MemoryMappedOutput fout = manager.openMemoryMappedOutput(binary, sizeof(T)* outerSize*innerSize);

	std::vector<size_t> idxVec;
	for(size_t i = 0; i < innerSize; ++i)
		idxVec.push_back(i);

	auto dataOut = &fout.access<T>();//std::array<T, OuterSize*InnerSize>>();
	algorithm::pfor(idxVec, [&](size_t& i) {
		// write data
		for(size_t j = 0; j < outerSize; ++j) {
			dataOut[i*outerSize + j] = vecVec[j][i];
		}
	});
	manager.close(fout);
}

// Read vector of vectors to binary in parallel
template<typename T>
std::vector<std::vector<T>> readVecVecFromFile(std::string filename, size_t outerSize, size_t innerSize) {
	std::vector<std::vector<T>> vecVec;
	core::FileIOManager& manager = core::FileIOManager::getInstance();

	core::Entry binary = manager.createEntry(filename, core::Mode::Binary);
	auto fin = manager.openInputStream(binary);

	for(size_t j = 0; j < outerSize; ++j) {
		vecVec.push_back(std::vector<T>());
		for(size_t i = 0; i < innerSize; ++i)
			vecVec[j].push_back(T());
	}

	for(size_t i = 0; i < innerSize; ++i) {
		// read position from file
		size_t idx = fin.read<size_t>();

		for(size_t j = 0; j < outerSize; ++j) {
			// read data
			vecVec[j][idx] = (fin.read<T>());
		}
	}

	manager.close(fin);
	return vecVec;
}


// Read vector of vectors to binary in parallel
template<typename T>
std::vector<std::vector<T>> readVecVecFromFileMM(std::string filename, unsigned outerSize, unsigned innerSize) {
	std::vector<std::vector<T>> vecVec;
	core::FileIOManager& manager = core::FileIOManager::getInstance();

	core::Entry binary = manager.createEntry(filename, core::Mode::Binary);
	auto fin = manager.openMemoryMappedInput(binary);
	auto dataIn = &fin.access<T>();//<std::array<T, InnerSize*OuterSize>>();

	for(size_t j = 0; j < outerSize; ++j) {
		vecVec.push_back(std::vector<T>());
		for(size_t i = 0; i < innerSize; ++i)
			vecVec[j].push_back(T());
	}

	for(size_t i = 0; i < innerSize; ++i) {
		for(size_t j = 0; j < outerSize; ++j) {
			// read data
			vecVec[j][i] = dataIn[i*outerSize + j];
		}
	}

	manager.close(fin);
	return vecVec;
}

} // end namespace user
} // end namespace api
} // end namespace allscale
