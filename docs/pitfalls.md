---
id: pitfalls
title: Common Pitfalls
sidebar_label: Common Pitfalls
---

> **This section is still a work in progress.**

There are a few pitfalls that are commonly encountered when first getting
started with Celerity.

## Diverging Host-Execution on Different Nodes

Each Celerity process that is started as part of a single cluster execution
(i.e., using `mpirun` or similar) _must_ execute the exact same host code (pertaining to any Celerity API interactions).
This means that for example something like this is _illegal_ and will result
in undefined behavior:

```cpp
if(rand() > 1337) {
    celerity::buffer<float, 2> my_buffer(...);
}
```

## Asynchronous Execution - Capturing by Reference

Celerity tasks submitted to the `celerity::distr_queue` are executed
_asynchronously_ at a later point in time. This means that the stack
surrounding a command function ("kernel") may have been unwound by the time it
is being invoked.

While Celerity and the underlying SYCL implementation will try to detect and
prevent certain types of common errors (for example capturing accessors by
reference), not all mistakes can be caught reliably.

In particular when using [host tasks](host-tasks.md), it is important to ensure
that all values that are captured by reference outlive the task:

```cpp
int global_variable = 22;

void some_function(celerity::distr_queue& q) {
    int local_variable = 42;
    q.submit([&](celerity::handler& cgh) {
        cgh.host_task([&] {
            printf("%d\n", global_variable); // safe, global variable outlives task
            printf("%d\n", local_variable); // dangling reference!
        });
    });
}
```

> Celerity supports experimental APIs that can replace most if not all uses for reference captures.
> See `celerity::experimental::host_object`, `celerity::experimental::side_effect` and
> `celerity::experimental::fence`.