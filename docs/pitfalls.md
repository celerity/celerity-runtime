---
id: pitfalls
title: Common Pitfalls
sidebar_label: Common Pitfalls
---

There are a few pitfalls that are commonly encountered when first getting
started with Celerity.

## Incorrectly Specified Range Mappers

Celerity requires [range mappers](range-mappers.md) to be specified on every accessor definition
in order to maintain data coherence across the cluster. When a kernel exhibits a complex data
access pattern, two core requirements are easy to violate from user code – even when using
Celerity's built-in range mappers.

### Out-Of-Bounds Accesses

A work item `item` must never access the buffer outside the range of `range_mapper(chunk)`,
where `chunk` is any chunk of the iteration space that contains `item`:

```cpp
// INCORRECT example: access pattern inside the kernel does not follow the range mapper
celerity::buffer<float, 1> buf({256});
celerity::distr_queue().submit([&](celerity::handler &cgh) {
    celerity::accessor acc(buf, cgh, celerity::access::one_to_one(), celerity::read_write);
    cgh.parallel_for(celerity::range<1>(128), [=](celerity::item<1> item) {
        // OUT-OF-BOUNDS ACCESS: `one_to_one` means `item` must only access `acc[item]`
        acc[item] += acc[item.get_id(0) + 128];
    });
});
```

> Out-of-bounds accesses can be detected at runtime by enabling the
> `CELERITY_ACCESSOR_BOUNDARY_CHECK` CMake option at the cost of some runtime
> overhead (enabled by default in debug builds).

### Overlapping Writes

Range mappers for `write_only` or `read_write` accessors must never produce overlapping buffer
ranges for non-overlapping chunks of the iteration space.

A likely beginner mistake is to violate the second constraint when implementing a **stencil code**.
The first intuition might be to operate on a single buffer using a `read_write` accessor together
with a `neighborhood` range mapper like so:
```cpp
// INCORRECT stencil example
celerity::distr_queue q;
celerity::buffer<float, 2> buf({256, 256});
for (int i = 0; i < N; ++i) {
    q.submit([&](celerity::handler &cgh) {
        // ILLEGAL RANGE MAPPER: `neighborhood` can not be used on a writing access
        celerity::accessor acc(buf, cgh, celerity::access::neighborhood(1, 1),
                celerity::read_write);
        cgh.parallel_for(buf.get_range(), [=](celerity::item<1> item) {
            acc[item] = acc[...] + acc[...] + /* ... stencil code */;
        });
    });
}
```
Intstead, these patterns must be implemented using a separate `neighborhood` read-access and
`one_to_one` write-access *on two separate* buffers:

```cpp
// correct stencil code
celerity::distr_queue q;
celerity::buffer<float, 2> input({256, 256});
celerity::buffer<float, 2> output({256, 256}); // double buffering
for (int i = 0; i < N; ++i) {
    q.submit([&](celerity::handler &cgh) {
        celerity::accessor read(input, cgh, celerity::access::neighborhood(1, 1),
                celerity::read_only);
        celerity::accessor write(output, cgh, celerity::access::one_to_one(),
                celerity::write_only, celerity::no_init);
        cgh.parallel_for(output.get_range(), [=](celerity::item<2> item) {
            write[item] = read[...] + read[...] + /* ... stencil code */;
        });
    });
    if (i + 1 < N) std::swap(input, output); // swapping buffers has trivial cost!
}
```

Note that this is not just a Celerity limitation, but inherent to the implementation of stencils
on GPUs, which must avoid races between reads and writes through a strategy like double-buffering
anyway.

> None of the `all`, `fixed`, `neighborhood` or `slice` built-in range mappers
> can be used for a writing access (unless the kernel only operates on a single
> work item).

## Illegal Reference Captures in Kernel and Host Task Functions

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

> Celerity supports APIs that can replace most if not all uses for reference captures.
> See `celerity::distr_queue::fence`, `celerity::experimental::host_object` and
> `celerity::experimental::side_effect`.

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
