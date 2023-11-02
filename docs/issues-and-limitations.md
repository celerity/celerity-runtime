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

## No Built-In Data-Dependent Control Flow Primitives

In some situations, the number of Celerity tasks required for a computation
may not be known fully in advance. For example, when using an iterative
method, a kernel might be repeated until some error metric threshold is
reached. Celerity currently offers no canonical way of incorporating such
branching decisions into the asynchronous execution flow.

That being said, it is not impossible to achieve this behavior today. For
example, the branching decision can be made using a [reduction](reductions.md)
and then observed on the application thread on a `fence`:

```cpp
celerity::distr_queue q;
celerity::buffer<float, 0> error;
for (;;) {
    q.submit([&](celerity::handler& cgh) {
        cgh.parallel_for(celerity::range<1>{1000},
            celerity::reduction(error, cgh, sycl::plus<float>{},
                celerity::property::reduction::initialize_to_identity{});
            [=](celerity::item<1> item, auto& err) { err += ...; });
    });
    // `fence` will capture buffer contents once all writes have completed
    auto future = q.fence(error);
    // optionally submit more work here to avoid stalling the async execution
    const float err = *future.get();
    if (err < epsilon) break;
}
```

---

If you encounter any additional issues, please [let us
know](https://github.com/celerity/celerity-runtime/issues/new).
