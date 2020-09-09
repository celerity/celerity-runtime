---
id: issues-and-limitations
title: Known Issues and Current Limitations
sidebar_label: Issues & Limitations
---

While Celerity can already do a lot, there still are some things it cannot
do. This is usually either because of a SYCL limitation, because we are still
figuring out how to fit certain functionality into the programming model,
or because we simply haven't had the time yet to implement a given feature.
If you are blocked by any of these or other issues, please
[let us know](https://github.com/celerity/celerity-runtime/issues/new).

Here is a (potentially incomplete) list of currently known issues:

## No Reductions

Celerity currently offers no dedicated API for performing distributed
reduction operations. While the experimental support for [collective host
tasks](host-tasks.md#experimental-collective-host-tasks) allows to implement
distributed reductions using e.g. `MPI_Reduce`, the calculations have to be
performed on the host. First-class support for device-accelerated distributed
reductions will be added to Celerity in the future.

We are currently evaluating the reduction functionalities proposed in the
[SYCL 2020 Provisional Specification](https://www.khronos.org/registry/SYCL/),
and how we could build a distributed variant on top of it.

## No Control Flow

In some situations, the number of Celerity tasks required for a computation
may not be known fully in advance. For example, when using an iterative
method, a kernel might be repeated until some error metric threshold is
reached. Celerity currently offers no canonical way of incorporating such
branching decisions into the data flow execution graph.

That being said, it is not impossible to achieve this behavior today. For
example, the branching decision can be made within a [distributed host
task](host-tasks.md#distributed-host-tasks) and then relayed into the main
execution thread. The latter waits using
`celerity::distr_queue::slow_full_sync` until a corresponding predicate has
been set, and then continues submitting Celerity tasks depending on the
predicate.

## Only Basic `parallel_for` Overload

Due to various rather technical issues with the SYCL 1.2.1 standard, Celerity
is currently unable to support the `nd_range` overload of `parallel_for`, as
well as `parallel_for_work_group`. However, thanks to improvements made in
SYCL 2020, Celerity will be able to support the former as soon as SYCL
implementations catch up, giving users explicit control over work group
sizes and access to local shared memory.

---

If you encounter any additional issues, please [let us
know](https://github.com/celerity/celerity-runtime/issues/new).
