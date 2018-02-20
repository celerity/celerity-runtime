#pragma once

#include "allscale/api/user/algorithm/async.h"
#include "allscale/api/core/treeture.h"
#include "allscale/api/user/algorithm/internal/operation_reference.h"

namespace allscale {
namespace api {
namespace user {
namespace algorithm {


	// ---------------------------------------------------------------------------------------------
	//									    Declarations
	// ---------------------------------------------------------------------------------------------


	/**
	 * The VCycle utility enalbes the generic description of a arbitrarily deep V-cycle computation.
	 * Each stage is realized by a different specialization of the VCycleStage class, conducting the
	 * necessary reduction, computation, and prolongation steps to assemble the full VCycle.
	 */

	class vcycle_reference;

	/**
	 * A generic v-cycle implementation enabling the creation of a vcycle solver by providing
	 * an implementation of a v-cycle stage body.
	 */
	template<template<typename M,unsigned L> class StageBody, typename Mesh>
	class VCycle;


	// ---------------------------------------------------------------------------------------------
	//									    Definitions
	// ---------------------------------------------------------------------------------------------


	/**
	 * An entity to reference the full range of a scan. This token
	 * can not be copied and will wait for the completion of the scan upon destruction.
	 */
	class vcycle_reference : public internal::operation_reference {

	public:

		// inherit all constructors
		using operation_reference::operation_reference;

	};


	namespace detail {


		template<
			typename Mesh,
			template<typename M,unsigned L> class StageBody,
			unsigned Level, 												// the level covered by this instance
			unsigned NumLevels												// total number of levels
		>
		class VCycleStage {

			using stage_body = StageBody<Mesh,Level>;

			using nested_stage_type = VCycleStage<Mesh,StageBody,Level-1,NumLevels>;

			using stage_body_type = StageBody<Mesh,Level>;

			stage_body_type body;

			nested_stage_type nested;

		public:

			VCycleStage(const Mesh& mesh)
				: body(mesh), nested(mesh) {}

			/**
			 * A function processing a single V-cycle starting at the current level.
			 */
			void run() {
				// one iteration of the V cycle (actually very simple)
				up();		// going up (fine to coarse)
				down();		// going down (coarse to fine)
			}

			void up() {
				// forward call to nested
				nested.up();
				body.restrictFrom(nested.getBody());
				body.computeFineToCoarse();
			}

			void down() {
				body.prolongateTo(nested.getBody());
				nested.getBody().computeCoarseToFine();
				nested.down();
			}

			stage_body_type& getBody() {
				return body;
			}

			void prolongateFrom(const StageBody<Mesh,Level+1>& parentBody) {
				body.prolongateFrom(parentBody);
			}

			template<unsigned Lvl>
			typename std::enable_if<Level==Lvl,const StageBody<Mesh,Level>&>::type
			getStageBody() const {
				return body;
			}

			template<unsigned Lvl>
			typename std::enable_if<Level!=Lvl,const StageBody<Mesh,Lvl>&>::type
			getStageBody() const {
				return nested.template getStageBody<Lvl>();
			}

			template<unsigned Lvl>
			typename std::enable_if<Level==Lvl,StageBody<Mesh,Level>&>::type
			getStageBody() {
				return body;
			}

			template<unsigned Lvl>
			typename std::enable_if<Level!=Lvl,StageBody<Mesh,Lvl>&>::type
			getStageBody() {
				return nested.template getStageBody<Lvl>();
			}

			template<typename Op>
			void forEachStage(const Op& op) {
				op(Level, this->body);
				nested.forEachStage(op);
			}

			template<typename Op>
			void forEachStage(const Op& op) const {
				op(Level, this->body);
				nested.forEachStage(op);
			}

		};


		template<
			typename Mesh,
			template<typename M,unsigned L> class StageBody,
			unsigned NumLevels												// total number of levels
		>
		class VCycleStage<Mesh,StageBody,0,NumLevels> {

			using stage_body_type = StageBody<Mesh,0>;

			stage_body_type body;

		public:

			VCycleStage(const Mesh& mesh)
				: body(mesh) {}

			/**
			 * A function processing a single V-cycle starting at the current level.
			 */
			void run() {
				// one iteration of the V cycle (actually very simple)
				up();		// going up (fine to coarse)
				down();		// going down (coarse to fine)
			}

			void up() {
				// just compute on this level
				body.computeFineToCoarse();
			}

			void down() {
				// nothing to do
			}

			stage_body_type& getBody() {
				return body;
			}

			void prolongateTo(const StageBody<Mesh,1>& parentBody) {
				body.prolongateTo(parentBody);
			}

			template<unsigned Lvl>
			typename std::enable_if<0==Lvl,const StageBody<Mesh,0>&>::type
			getStageBody() const {
				return body;
			}

			template<unsigned Lvl>
			typename std::enable_if<0==Lvl,StageBody<Mesh,0>&>::type
			getStageBody() {
				return body;
			}

			template<typename Op>
			void forEachStage(const Op& op) {
				op(0, this->body);
			}

			template<typename Op>
			void forEachStage(const Op& op) const {
				op(0, this->body);
			}

		};


	}



	template<
		template<typename M,unsigned L> class StageBody,
		typename Mesh
	>
	class VCycle {

		using top_stage_type = detail::VCycleStage<Mesh,StageBody,Mesh::levels-1,Mesh::levels>;

		top_stage_type topStage;

	public:

		using mesh_type = Mesh;

		const mesh_type& mesh;

		VCycle(const mesh_type& mesh) : topStage(mesh), mesh(mesh) {}

		vcycle_reference run(std::size_t numCycles = 1) {
			return async([&, numCycles]() {
				// run the given number of cycles
				for(std::size_t i = 0; i<numCycles; ++i) {
					topStage.run();
				}
			});
		}

		template<unsigned Level = 0>
		const StageBody<Mesh,Level>& getStageBody() const {
			return topStage.template getStageBody<Level>();
		}

		template<unsigned Level = 0>
		StageBody<Mesh,Level>& getStageBody() {
			return topStage.template getStageBody<Level>();
		}

		template<typename Op>
		void forEachStage(const Op& op) {
			topStage.forEachStage(op);
		}

		template<typename Op>
		void forEachStage(const Op& op) const {
			topStage.forEachStage(op);
		}

	};


} // end namespace algorithm
} // end namespace user
} // end namespace api
} // end namespace allscale
