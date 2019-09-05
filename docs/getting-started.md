---
id: getting-started
title: Getting Started
sidebar_label: Getting Started
---

Celerity allows you to write highly parallel applications that can be run on
a cluster of accelerator nodes. It focuses on providing a way of scaling
applications to many nodes without having to be an expert in cluster
programming. In fact, the Celerity API does not make it apparent that a program is
(potentially) running on many nodes at all: There is no notion of _ranks_;
partitioning of work and data is taken care of transparently behind the scenes.
This lets you focus on your actual work, without having to concern yourself with the
complexities of modern distributed memory cluster programming.

While ease of use is one of Celerity's main goals, simplicity can only go so
far without sacrificing considerable performance. Proficiency in modern C++
as well as at least a rough understanding of how accelerator (GPU) programming
differs to parallel CPU programming is required to make efficient use of Celerity.
Lastly, you will require a good understanding of the algorithms and techniques you
intend to implement using Celerity in order for the runtime system to be able
to run it on a cluster both correctly and in an efficient manner.

Celerity is built on top of [SYCL](https://www.khronos.org/sycl/), an
open-standard high-level C++ embedded domain specific language for
programming accelerators. SYCL provides a great API that hits a sweet spot
between expressiveness and power as well as ease of use, making it the
perfect starting point for Celerity: We set out to find the minimal set of
extensions required to bring the SYCL API to distributed memory clusters -
thus making it relatively easy to migrate an existing SYCL application to
Celerity. If you don't have any experience with SYCL, don't worry, as we will
introduce the most important concepts along the way.

If this piqued your interest and you would like to try it for yourself, check
out the [Installation](installation.md) section on how to build and install
Celerity. For a complete example of how to set up a Celerity application from
start to finish, see the [Tutorial](tutorial.md).

For more information on the overall Celerity API as well as the driving
principles behind its design, please see our Celerity in-depth
[Overview](overview.md) section.
