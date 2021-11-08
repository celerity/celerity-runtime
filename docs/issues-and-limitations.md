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

---

If you encounter any additional issues, please [let us
know](https://github.com/celerity/celerity-runtime/issues/new).
