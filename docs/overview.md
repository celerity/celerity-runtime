---
id: overview
title: Overview
sidebar_label: Overview
---

> **This section is still a work in progress.**

## Celerity Execution Timeline

The following animation illustrates the basics of how programs are executed
internally by the Celerity runtime system:

<video autoplay loop width="100%">
  <source src="assets/celerity_overview.mp4" type="video/mp4">
  <p>
    Your browser doesn't support HTML5 video.
    Here is a <a href="assets/celerity_overview.mp4">link to the video</a> instead.
  </p>
</video>

Everything starts with the _Prepass_, which is similar to normal program
execution. However, whenever a relevant Celerity queue operation -- such as
`submit` -- is encountered, this pass does not
actually fully execute the related _kernel_ code (marked in color in the
video). Instead, this code, together with some dependency and scheduling
metainformation, is recorded in the Celerity **task graph**.

Concurrently with the Prepass, a _Scheduler_ thread constructs a more
detailed **command graph** from existing task graph nodes. This command graph
includes individual instructions for each node in the system, and also
encodes all necessary data transfers to maintain the consistent view of data
which would be expected if the program were to be executed on a single node.

On each _Worker_ node the command graph is asynchronously consumed and
executed, which finally leads to actually running the computations described
by each kernel in the input program, distributed among the GPUs in the
cluster.

In order to make the entire process more clear, the animation is simplified
in a few important aspects:

- The task graph shown is strictly sequential -- in practice independent
  tasks can exist at various points in the timeline.
- For this illustration, there are only two nodes in the system. The command
  graph would of course increase in size with additional nodes.
- While in the animation workers proceed seemingly sequentially over data send
  and receive operations, all of these are in fact processed asynchronously
  in the actual implementation.

Despite these simplifications, the animation captures the basic execution
principle of the Celerity runtime, and hopefully also makes the reasons for
some of the pitfalls described in the [Common Pitfalls](pitfalls.md) section
more clear.
