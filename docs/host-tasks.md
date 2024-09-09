---
id: host-tasks
title: I/O and Verification using Host Tasks
sidebar_label: Host Tasks
---

While Celerity is focused on accelerator computations, real-world applications will contain host code as part of their
data flow. This may include calls to specialized libraries, distributed I/O operations or verification.

To integrate such tasks into an asynchronous distributed program, Celerity offers **host tasks** with semantics
specialized for these different applications. Similar to compute tasks, they are scheduled through the command group
handler using the `celerity::handler::host_task` family of methods.

Host tasks are executed in a background thread pool on each participating node and may execute concurrently.

## Simple Host Tasks

The simplest kind of host task executes once on exactly one node. It is selected by calling `host_task` with the `once` tag:

```cpp
cgh.host_task(celerity::once, []{ ... });
```

Buffers can be accessed in the usual fashion, although there is no `item` structure passed into the kernel. Instead,
when constructing an accessor, a `fixed` or `all` range mapper is used to specify the range of interest. Also,
the `*_host_task` selector must be used for selecting the access mode.

```cpp
celerity::queue q;
celerity::buffer<float, 1> result;
q.submit([&](celerity::handler &cgh) {
	celerity::accessor acc{buffer, cgh, celerity::access::all{},
			celerity::read_only_host_task};
    cgh.host_task(celerity::once, [=]{
        printf("The result is %g\n", acc[0]);
    });
});
```

## Master-Node Host Tasks

By passing `on_master_node` instead of the `once` tag, we can enforce that all invocations of the host task run on the same cluster node
(MPI rank 0, dubbed the "master node"). This is useful when state other than buffers (such as a host object) is shared between multiple host tasks,
and we need to guarantee that all tasks access the same object.

```cpp
cgh.host_task(celerity::on_master_node, []{ ... });
```

## Distributed Host Tasks

If a computation involving host code is to be distributed across a cluster, Celerity can split the iteration space
accordingly. Such a distributed host task is created by passing a global size to `host_task`:

```cpp
cgh.host_task(global_size, [](celerity::partition<Dims>) { ... });
cgh.host_task(global_size, global_offset, [](celerity::partition<Dims>) { ... });
```

Instead of the per-item kernel invocation of `handler::parallel_for` that is useful for accelerator
computations, ther host kernel will receive _partitions_ of the iteration space. They describe the iteration sub-space
this node receives:

```cpp
celerity::queue q;
q.submit([&](celerity::handler &cgh) {
    cgh.host_task(celerity::range<1>(100), [=](celerity::partition<1> part) {
        printf("This node received %zu items\n", part.get_subrange().range[0]);
    });
});
```

In distributed host tasks, buffers can be accessed using the same range mappers as in device computations with the
expected semantics.

Celerity makes no guarantees about the granularity of the split. Also, some nodes may receive multiple concurrent
invocations of the kernel while others might not participate in the host task at all.

## Experimental: Collective Host Tasks

> **Note:** This feature is subject to change.

Efficient distributed I/O routines need to perform _collective_ operations accross a cluster, i.e. operations
in which all nodes participate simultaneously. A popular API that makes use of this feature is
[_Parallel HDF5_](https://support.hdfgroup.org/HDF5/PHDF5/), providing access to array data on the file system
through an API using MPI I/O as the underlying driver (See the Celerity `distr_io` example for a demonstration).

Invoking collective operations from a Celerity program requires additional support from the runtime to guarantee
proper ordering of MPI calls, the participation of each node in the operation and absence of race conditions between
concurrent host tasks on the same queue. To request a host task satisfying these conditions, use the
`experimental::collective` tag:

```cpp
cgh.host_task(celerity::experimental::collective,
    [](celerity::experimental::collective_partition part) { ... });
```

`collective_partition` is a specialization of the one-dimensional `partition`. Note how no global size is passed
to the host task. Instead, the runtime creates a one-dimensional iteration space where the size is the number of
participating nodes, and the single-element subrange on each node is the node index. Additionally,
`collective_partition` provides access to the MPI communicator for this task:

```cpp
celerity::queue q;
q.submit([](celerity::handler &cgh) {
    cgh.host_task(celerity::experimental::collective,
            [](celerity::experimental::collective_partition> part) {
        MPI_Comm comm = part.get_collective_mpi_comm();
        MPI_Barrier(comm);
    });
});
```

Third-party APIs using MPI collectives will have a `MPI_Comm` parameter where this communicator can be passed in.
Celerity guarantees the communicator to be free of race conditions with other operations for the duration of the
host task. If multiple collective tasks are scheduled, they receive the same MPI communicator.

### Collective Groups

To guarantee proper ordering of MPI operations across the cluster, collective host tasks on the same communicator
must neither be run concurrently nor be reordered on one node. In case there are multiple independent collective
operations eligible to be run concurrently, Celerity can be notified of this by using _collective groups_:

```cpp
celerity::queue q;
celerity::experimental::collective_group first_group;
celerity::experimental::collective_group second_group;
q.submit([&](celerity::handler &cgh) {
    cgh.host_task(celerity::experimental::collective(first_group), []...);
});
q.submit([&](celerity::handler &cgh) {
    cgh.host_task(celerity::experimental::collective(second_group), []...);
});
```

Since these two host tasks use different collective groups and are also independent with regards to their buffer
accesses, they can now be executed concurrently. For this purpose, each kernel receives a MPI communicator unique to its
collective group. The prior example without explicit mentions of a collective group implicitly binds to
`celerity::experimental::default_collective_group`.

### Buffer Access from a Collective Host Task

Collective host tasks are special in that they receive an implicit one-dimensional iteration space that just identifies
the participating nodes. To access buffers in a meaningful way, these node indices must be translated to buffer regions.
In the typical Celerity fashion, this is handled via range mappers.

The `celerity::experimental::even_split` range mapper maps a one-dimensional range onto arbitrary-dimensional buffers by
splitting them along the first (slowest) dimension into contiguous memory portions.
`celerity::accessor::get_allocation_window` can then be used to retrieve the host-local chunk of the buffer:

```cpp
celerity::queue q;
celerity::buffer<float, 2> buf;
q.submit([&](celerity::handler& cgh) {
	celerity::accessor acc{buffer, cgh,
			celerity::experimental::access::even_split<2>{},
			celerity::read_only_host_task};
    cgh.host_task(celerity::experimental::collective,
            [=](celerity::experimental::collective_partition part) {
        auto aw = acc.get_allocation_window(part);
        // ...
    });
});
```
