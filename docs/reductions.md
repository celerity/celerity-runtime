---
id: reductions
title: Reductions
sidebar_label: Reductions
---

Celerity implements cluster-wide reductions in the spirit of
[SYCL 2020 reduction variables](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:reduction).

The following distributed program computes the sum from 0 to 999 in `sum_buf` using a reduction:

```c++
celerity::distr_queue q;
celerity::buffer<size_t, 1> sum_buf{{1}};
q.submit([&](celerity::handler& cgh) {
    auto rd = celerity::reduction(sum_buf, cgh, sycl::plus<size_t>{},
                                  celerity::property::reduction::initialize_to_identity{});
    cgh.parallel_for(celerity::range<1>{1000}, rd,
                     [=](celerity::item<1> item, auto& sum) { sum += item.get_id(0); });
});
```

A reduction output buffer can have any dimensionality, but must be unit-sized. Like buffer requirements, the result of
such a reduction is made available on nodes as needed, so the Celerity API does not distinguish between `Reduce` and
`Allreduce` operations like MPI does.

In Celerity, every reduction operation must have a known identity. For
[SYCL function objects](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:function-objects)
these are known implicitly, for user-provided functors like lambdas, an explicit identity must be provided:

```c++
auto parity = [](unsigned a, unsigned b) { return (a ^ b) & 1u; };
auto rd = celerity::reduction(buf, cgh, parity, 0u /* explicit identity */,
                              celerity::property::reduction::initialize_to_identity{});
```

## Limitations

### Only Scalar Reductions

Currently, the SYCL standard only mandates scalar reductions, i.e. reductions that produce a single scalar value.
While that is useful for synchronization work like terminating a loop on a stopping criterion, it is not enough for
other common operations like histogram construction. Since Celerity delegates to SYCL for intra-node reductions,
higher-dimensional reduction outputs will only become available once SYCL supports them.
