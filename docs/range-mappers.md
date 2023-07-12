---
id: range-mappers
title: Expressing Data Dependencies with Range Mappers
sidebar_label: Range Mappers
---

In order for Celerity to be able to split kernel executions across an
arbitrary number of worker nodes, it needs to be able to determine how a
kernel accesses a buffer. More specifically, it needs a way of knowing how
each individual work item intends to access the buffer both spatially (i.e.,
at which indices) as well as in what way (e.g. for reading, writing and so
on). While for the latter SYCL's access modes are sufficient, the former
requires an API extension that we call **range mappers**.

## Overview

Range mappers are functors that map a portion of kernel execution, a
so-called **chunk**, to a subrange of a buffer that is being accessed by the
kernel. More concretely, for a kernel with iteration-range dimensionality
`KernelDims` that accesses a buffer with dimensionality `BufferDims`, the range
mapper must be callable with at least one of the following signatures:
```cpp
celerity::subrange<BufferDims> range_mapper(celerity::chunk<KernelDims> chnk);
celerity::subrange<BufferDims> range_mapper(celerity::chunk<KernelDims> chnk,
        celerity::range<BufferDims> buffer_range);
```

`KernelDims` and `BufferDims` can either be constants matching the kernel
and buffer definitions or originate from template parameters.
In general, the dimensionality of a kernel does not have to match that of
the buffer(s) it operates on. For example, a two-dimensional kernel might
access a one-dimensional buffer, or vice-versa.

### Usage

Range mappers are passed as the third argument when constructing a
`celerity::accessor`. This means that the spatial accessing
behavior of a kernel can vary from buffer to buffer. For example, the
following command group specifies two different range mappers (whose
definition is omitted) for buffers `buf_a` and `buf_b`:

```cpp
queue.submit([&](celerity::handler& cgh) {
	celerity::accessor r_a{cgh, buf_a, my_mapper, celerity::read_only};
	celerity::accessor dw_b{cgh, buf_b, other_mapper, celerity::write_only, celerity::no_init};

    cgh.parallel_for(...);
});
```

### Getting an Intuition

A useful way of thinking about kernel chunks is as a collection of individual
kernel threads, or _work items_. Each work item is associated with a unique
one-, two-, or three-dimensional index. Together, they span the entire global
execution range for a kernel call. If you can determine the spatial buffer
access behavior for each work item, all that's left to do is to express it in
such a way that it can be computed for entire collections of work items (i.e.,
chunks) at a time. The resulting subrange should then be the union of each
indiviual work items' requirements.

Arguably the simplest of such mappings would be for a kernel that only
accesses a buffer at the very same index as the index of its work item. Such
a one-to-one mapping could be implemented by simply returning the entire
chunk without changes:

```cpp
template<int Dims>
celerity::subrange<Dims> my_one_to_one(celerity::chunk<Dims> chnk) {
    return celerity::subrange<Dims>(chnk);
}
```

Note that in order for this to work, the dimensionality of both the kernel
and the buffer must match.

Since this is such a common pattern, Celerity provides _built-in range
mappers_ for one-to-one mappings and other frequently used patterns. See
[Built-in Range Mappers](range-mappers.md#built-in-range-mappers).

## Built-in Range Mappers

Celerity currently ships with the following built-in range mappers (all under
the `celerity::access` namespace):

### One-to-one

The `one_to_one` range mapper directly maps the offset and range of a given
chunk to a subrange. This requires that the dimensionality of the kernel and
buffer matches.

```cpp
struct celerity::access::one_to_one {
    template <int Dims>
    subrange<Dims> operator()(chunk<Dims> chnk) const;
};
```

### Slice

The `slice` range mapper allows to extend the range of a chunk along a given
dimension indefinitely, thus selecting an entire slice of a buffer in that
dimension. This requires that the dimensionality of the kernel and buffer
matches. A common use case for this range mapper is dense matrix
multiplication.

```cpp
template <int Dims>
struct celerity::access::slice {
    explicit slice(size_t dim_idx);

    subrange<Dims> operator()(chunk<Dims> chnk) const;
};
```

### Neighborhood

The neighborhood range mapper allows to select a specified number of indices
around a given index in every dimension. Neighborhoods are clamped to the
boundaries of the buffer. This range mapper is commonly used for stencil
codes.

```cpp
template <int Dims>
struct celerity::access::neighborhood {
    neighborhood(size_t dim0);
    /* only available if Dims >= 2 */
    neighborhood(size_t dim0, size_t dim1);
    /* only available if Dims == 3 */
    neighborhood(size_t dim0, size_t dim1, size_t dim2);

    subrange<Dims> operator()(chunk<Dims> chnk) const;
};
```

### Fixed

The `fixed` range mapper allows to specify a fixed subrange that each chunk
requires, i.e., independently of the input chunk. This range mapper is
commonly used in situations where all worker nodes need access to a fixed
section of a buffer, for example a filter to apply during a convolution.

```cpp
template <int BufferDims>
struct celerity::access::fixed {
    fixed(subrange<BufferDims> sr);

    template <int KernelDims>
    subrange<BufferDims> operator()(chunk<KernelDims>) const;
};
```

### All

The `all` range mapper selects the entire buffer, regardless of the input
chunk. This is a special case of the `fixed` range mapper and is provided for
convenience.

```cpp
struct celerity::access::all {
    template <int KernelDims, int BufferDims>
    subrange<BufferDims> operator()(chunk<KernelDims>, range<BufferDims>) const;
};
```

## Validity Requirements

Range mappers must fulfill certain requirements in order to be considered
valid.

- A range mapper must be _monotonic_, meaning that the result for any given
  work item must not change depending on the other items, i.e., the given input
  chunk. Given the kernel domain `K`, a buffer domain `B` and a range mapper
  `r: K -> B`, it must hold that `for all a,b that are subsets of K: if a is a subset of b, then r(a) is a subset of r(b)`.
  In other words, if two chunks share one or more work items, their resulting
  data requirements must both include the requirements of the shared items.
- A range mapper must never assume a particular number of chunks. Part of the
  reason range mappers exist in the first place is to alleviate users of having
  to think about how work and data is to be split.
- For producer accesses (that is, everything except
  `celerity::access_mode::read`), the output of a range mapper must not overlap.

Range mappers that do not satisfy all of the above points cause undefined
behavior. Note that it is perfectly valid for range mappers to return an
empty subrange for certain chunks.
