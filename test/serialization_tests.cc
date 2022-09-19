#include "test_utils.h"

#include <catch2/generators/catch_generators_all.hpp>

namespace celerity::detail {

TEST_CASE("frame_vector iteration reproduces the emplaced values", "[frame_vector]") {
	struct frame {
		using payload_type = uint32_t;
		enum { tag_a, tag_b } tag = tag_b; // non-trivial default ctor
		size_t count;
		payload_type payload[];
	};

	frame_vector_layout<frame> layout;
	layout.reserve_back(from_payload_count, 2);
	layout.reserve_back(from_payload_count, 3);
	layout.reserve_back(from_payload_count, 1);
	CHECK(layout.get_size_bytes()
	      == utils::ceil(4 * sizeof(size_t), alignof(frame))                                    //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 2, alignof(frame)) //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 3, alignof(frame)) //
	             + (sizeof(frame) + sizeof(frame::payload_type) * 1));

	frame_vector_builder<frame> builder(layout);
	{
		auto& f = builder.emplace_back(from_payload_count, 2);
		f.count = 2;
		f.payload[0] = 8;
		f.payload[1] = 9;
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 3);
		f.tag = frame::tag_a;
		f.count = 3;
		f.payload[0] = 2;
		f.payload[1] = 3;
		f.payload[2] = 4;
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 1);
		f.count = 1;
		f.payload[0] = 10;
	}

	auto vector = std::move(builder).into_vector();
	CHECK(std::distance(vector.cbegin(), vector.cend()) == 3);

	auto it = vector.begin();
	REQUIRE(it != vector.end());
	CHECK(it.get_payload_count() == 2);
	CHECK(it.get_size_bytes() == sizeof(frame) + 2 * sizeof(uint32_t));
	CHECK(it->tag == frame::tag_b);
	CHECK(it->count == 2);
	CHECK(it->payload[0] == 8);
	CHECK(it->payload[1] == 9);

	const auto same_it = it++;
	REQUIRE(same_it == vector.begin());
	CHECK(&*same_it == &*vector.begin());

	REQUIRE(it != vector.end());
	CHECK(it.get_payload_count() == 3);
	CHECK(it.get_size_bytes() == sizeof(frame) + 3 * sizeof(uint32_t));
	CHECK(it->tag == frame::tag_a);
	CHECK(it->count == 3);
	CHECK(it->payload[0] == 2);
	CHECK(it->payload[1] == 3);
	CHECK(it->payload[2] == 4);

	++it;
	REQUIRE(it != vector.end());
	CHECK(it.get_payload_count() == 1);
	CHECK(it.get_size_bytes() == sizeof(frame) + 1 * sizeof(uint32_t));
	CHECK(it->tag == frame::tag_b);
	CHECK(it->count == 1);
	CHECK(it->payload[0] == 10);

	++it;
	CHECK(it == vector.end());

	CHECK(vector.begin() == vector.cbegin());
	CHECK(vector.cend() == vector.end());
	CHECK(vector.begin() != vector.cend());
	CHECK(vector.cend() != vector.begin());
}

TEST_CASE("frame_vector correctly lays out frame types with small aligment", "[frame_vector]") {
	struct frame {
		using payload_type = uint16_t;
		payload_type payload[];
	};

	frame_vector_layout<frame> layout;
	layout.reserve_back(from_payload_count, 1);
	layout.reserve_back(from_payload_count, 5);
	layout.reserve_back(from_payload_count, 4);
	CHECK(layout.get_size_bytes()
	      == utils::ceil(4 * sizeof(size_t), alignof(frame))                                    //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 1, alignof(frame)) //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 5, alignof(frame)) //
	             + (sizeof(frame) + sizeof(frame::payload_type) * 4));

	frame_vector_builder<frame> builder(layout);
	{
		auto& f = builder.emplace_back(from_payload_count, 1);
		f.payload[0] = 1;
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 5);
		f.payload[0] = 4;
		f.payload[1] = 5;
		f.payload[2] = 6;
		f.payload[3] = 7;
		f.payload[4] = 8;
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 4);
		f.payload[0] = 10;
		f.payload[1] = 11;
		f.payload[2] = 12;
		f.payload[3] = 13;
	}

	auto vector = std::move(builder).into_vector();

	auto it = vector.begin();
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == 2);
	CHECK(it.get_payload_count() == 1);
	CHECK(it->payload[0] == 1);

	++it;
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == 10);
	CHECK(it.get_payload_count() == 5);
	CHECK(it->payload[0] == 4);
	CHECK(it->payload[1] == 5);
	CHECK(it->payload[2] == 6);
	CHECK(it->payload[3] == 7);
	CHECK(it->payload[4] == 8);

	++it;
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == 8);
	CHECK(it.get_payload_count() == 4);
	CHECK(it->payload[0] == 10);
	CHECK(it->payload[1] == 11);
	CHECK(it->payload[2] == 12);
	CHECK(it->payload[3] == 13);

	++it;
	CHECK(it == vector.end());
}

TEST_CASE("frame_vector correctly lays out frame types with large aligment", "[frame_vector]") {
	struct alignas(16) frame {
		using payload_type = uint64_t;
		int tag;
		alignas(16) payload_type payload[];
	};

	frame_vector_layout<frame> layout;
	layout.reserve_back(from_payload_count, 0);
	layout.reserve_back(from_payload_count, 1);
	layout.reserve_back(from_payload_count, 2);
	layout.reserve_back(from_payload_count, 3);
	layout.reserve_back(from_payload_count, 1);

	CHECK(layout.get_size_bytes()
	      == utils::ceil(6 * sizeof(size_t), alignof(frame))                                    //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 0, alignof(frame)) //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 1, alignof(frame)) //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 2, alignof(frame)) //
	             + utils::ceil(sizeof(frame) + sizeof(frame::payload_type) * 3, alignof(frame)) //
	             + (sizeof(frame) + sizeof(frame::payload_type) * 1));

	frame_vector_builder<frame> builder(layout);
	{
		auto& f = builder.emplace_back(from_payload_count, 0);
		REQUIRE(reinterpret_cast<uintptr_t>(&f) % 16 == 0);
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 1);
		REQUIRE(reinterpret_cast<uintptr_t>(&f) % 16 == 0);
		f.payload[0] = 4;
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 2);
		REQUIRE(reinterpret_cast<uintptr_t>(&f) % 16 == 0);
		f.payload[0] = 2;
		f.payload[1] = 3;
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 3);
		REQUIRE(reinterpret_cast<uintptr_t>(&f) % 16 == 0);
		f.payload[0] = 7;
		f.payload[1] = 8;
		f.payload[2] = 9;
	}
	{
		auto& f = builder.emplace_back(from_payload_count, 1);
		REQUIRE(reinterpret_cast<uintptr_t>(&f) % 16 == 0);
		f.payload[0] = 11;
	}

	auto vector = std::move(builder).into_vector();

	auto it = vector.begin();
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == sizeof(frame));
	CHECK(it.get_payload_count() == 0);

	++it;
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == sizeof(frame) + sizeof(frame::payload_type));
	CHECK(it.get_payload_count() == 1);
	CHECK(it->payload[0] == 4);

	++it;
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == sizeof(frame) + 2 * sizeof(frame::payload_type));
	CHECK(it.get_payload_count() == 2);
	CHECK(it->payload[0] == 2);
	CHECK(it->payload[1] == 3);

	++it;
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == sizeof(frame) + 3 * sizeof(frame::payload_type));
	CHECK(it.get_payload_count() == 3);
	CHECK(it->payload[0] == 7);
	CHECK(it->payload[1] == 8);
	CHECK(it->payload[2] == 9);

	++it;
	REQUIRE(it != vector.end());
	CHECK(it.get_size_bytes() == sizeof(frame) + sizeof(frame::payload_type));
	CHECK(it.get_payload_count() == 1);
	CHECK(it->payload[0] == 11);

	++it;
	CHECK(it == vector.end());
}


TEST_CASE("can create shared_frame_ptrs from frame_vector iterators", "[frame-vector]") {
	struct frame {
		using payload_type = std::byte;
		int tag;
		payload_type payload[];
	};

	frame_vector_layout<frame> layout;
	layout.reserve_back(from_payload_count, 0);
	layout.reserve_back(from_payload_count, 1);
	layout.reserve_back(from_payload_count, 2);

	frame_vector_builder<frame> builder(layout);
	builder.emplace_back(from_payload_count, 0).tag = 6;
	builder.emplace_back(from_payload_count, 1).tag = 5;
	builder.emplace_back(from_payload_count, 2).tag = 4;

	auto vector = std::make_shared<frame_vector<frame>>(std::move(builder).into_vector());
	CHECK(vector.use_count() == 1);

	{
		auto it = vector->begin();
		auto b0 = it.get_shared_from_this();
		CHECK(b0.get_payload_count() == 0);
		CHECK(b0->tag == 6);

		++it;
		auto b1 = it.get_shared_from_this();
		CHECK(b1.get_payload_count() == 1);
		CHECK(b1->tag == 5);

		++it;
		auto b2 = it.get_shared_from_this();
		CHECK(b2.get_payload_count() == 2);
		CHECK(b2->tag == 4);

		++it;
		CHECK(it == vector->end());
		CHECK(vector.use_count() == 4);
	}
	CHECK(vector.use_count() == 1);
}

} // namespace celerity::detail
