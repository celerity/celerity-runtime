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
surrounding a Celerity command group function might
have been unwound by the time it is being called.

For this reason Celerity by default enforces that tasks only capture
surrounding variables by value, rather than by reference. If you know what
you are doing and would like to disable this check, you can use the
`distr_queue::submit` overload accepting reference captures:

```cpp
celerity::distr_queue q;
bool flag = false;
q.submit(celerity::allow_by_ref, [&flag](celerity::handler &cgh) {...});
```

Also, similar to SYCL kernels, both host and device kernels must not capture accessors
or other values from the command group function by reference since the kernels will
outlive it. To avoid mistakes, make it a habit to **never use by-reference capture
defaults**, even when using `allow_by_ref`:

```cpp
celerity::distr_queue q;
celerity::buffer<int, 1> buf;
bool flag = false;
q.submit(celerity::allow_by_ref, [=, &flag](celerity::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::read,
        cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<1>());
    cgh.host_task(celerity::on_master_node, [=, &flag] {
        flag = acc[0] == 42;
    });
});
```