---
id: core-principles
title: Core Principles
sidebar_label: Core Principles
---

The Celerity API and runtime system follow a small number of core principles,
which help us to focus on the goals of the project and guide our design
decisions.

- **Data flow over control flow** - dependencies in computations using the
  Celerity API should be defined by their respective data flow rather than
  explicit control flow. Avoid creating two potentially redundant
  synchronization paths.

- **Everything is asynchronous** - in the runtime implementation, all command
  generation and data transfers should happen asynchronously. This applies
  both across distributed memory nodes and between individual runtime threads
  on each node.

- **When in doubt, stay close to SYCL** - when introducing new API
  functionality and concepts, try to stick as closely as possible to SYCL, to
  ease the migration path for new users. Of course, some functionality needs
  to be reconsidered for distributed memory.
