---
id: tutorial
title: Complete Application Tutorial
sidebar_label: Tutorial
---

This section gives a walkthrough of how a simple Celerity application can be
set up from start to finish. Before you begin, make sure you have [built and
installed](installation.md) Celerity and all of its dependencies.

We are going to implement a simple image processing kernel that performs edge
detection on an input image and writes the resulting image back to the
filesystem. Here you can see how a result might look (white parts):

![](assets/tutorial_edge_detection_flower_by_reinhold_moeller.jpg)

[Original image](https://commons.wikimedia.org/wiki/File:Wiesenknopf_Bl%C3%BCte_6260037-PSD-PSD.jpg) by Reinhold Möller (CC-BY-SA 4.0).

## Setting Up a CMake Project

The first thing you typically want to do when writing a Celerity application
is to set up a CMake project. For this, create a new folder for your project
and in it create a file `CMakeLists.txt` with the following contents:

```cmake
cmake_minimum_required(VERSION 3.5.1)
project(celerity_edge_detection)

find_package(Celerity CONFIG REQUIRED)

add_executable(edge_detection edge_detection.cpp)
add_celerity_to_target(TARGET edge_detection SOURCES edge_detection.cpp)
```

With this simple CMake configuration file we've created a new executable
called `edge_detection` that links to Celerity. The important section is the
call to `add_celerity_to_target`, where we specify both the target that we
want to turn into a Celerity executable, as well as all source files that
should be compiled for accelerator execution.

Create an empty file `edge_detection.cpp` next to your `CMakeLists.txt`.
Then, create a new folder `build` inside your project directory, navigate
into it and simply run `cmake ..` to configure your project. Just as during
[installation](installation.md), you might have to provide some additional
parameters to CMake in order for it to find and/or configure Celerity and its
dependencies.

## Image Handling Boilerplate

We're going to start by adding the necessary code to load (and later save) an
image file. To this end, we'll use the [stb](https://github.com/nothings/stb)
single file libraries. Download `stb_image.h` and `stb_image_write.h` from
GitHub and drop them next to our source file.

Next, add the following code to `edge_detection.cpp`:

```cpp
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {
    if(argc != 2) return EXIT_FAILURE;
    int img_width, img_height;
    uint8_t* img_data = stbi_load(argv[1], &img_width, &img_height, nullptr, 1);
    stbi_image_free(img_data);
    return EXIT_SUCCESS;
}
```

First we check that the user provided an image file name and if so, we load
the corresponding file using `stbi_load()`. The last parameter tells stb that
we want to load the image as grayscale. The result is then stored in an array
of `uint8_t`, which consists of `img_height` lines of size `img_width` each.
We then immediately free the image again and exit.

> Now might be a good time to compile and run the program to make sure
> everything works so far.

## Celerity Queue and Buffers

With everything set up, we can now begin to implement the Celerity portion of
our application. The first thing that we will require in any Celerity program
is the **distributed queue**. Similar to how a SYCL queue allows you to
submit work to a compute device, the Celerity distributed queue allows you to
submit work to the distributed runtime system -- which will subsequently be
split transparently across all available worker nodes.

Additionally, we will require **buffers** to store our input image as well as
the resulting edge-detected image in a way that can be efficiently accessed
by the GPU. Let's create our two buffers and the distributed queue now:

```cpp
#include <celerity/celerity.h>
...
uint8_t* img_data = stbi_load(argv[1], &img_width, &img_height, nullptr, 1);
celerity::buffer<uint8_t, 2> input_buf(img_data, cl::sycl::range<2>(img_height, img_width));
stbi_image_free(img_data);
celerity::buffer<uint8_t, 2> edge_buf(cl::sycl::range<2>(img_height, img_width));
celerity::distr_queue queue;
...
```

With this we've created a couple of two-dimensional buffers, `input_buf` and
`edge_buf`, that both store values of type `uint8_t` and are of size
`(img_height, img_width)`. Notice that we initialize `input_buf` using the
image data that we've just read. We can then immediately free the raw image,
as we no longer need it. `edge_buf` on the other hand is not being
initialized with any existing data, as it will be used to store the result of
our image processing kernel.

## Detecting Those Edges

Now we are ready to do the actual edge detection. For this we will write a
**kernel function** that will be executed on one or more GPUs. The way kernels
are specified in Celerity is very similar to how it is done in SYCL:

```cpp
queue.submit([=](celerity::handler& cgh) {
    // TODO: Buffer accessors
    cgh.parallel_for<class MyEdgeDetectionKernel>(
        cl::sycl::range<2>(img_height - 2, img_width - 2),
        cl::sycl::id<2>(1, 1),
        [=](cl::sycl::item<2> item) {
            // TODO: Kernel code
        }
    );
});
```

We call `queue.submit()` to inform the Celerity runtime that we want to
execute a new kernel function. As an argument, we pass a so-called **command
group**; a C++11 lambda function. Command groups themselves are not being
executed on an accelerator. Instead, they serve as a way of tying kernels
to buffers, informing the runtime system exactly how we plan to access
different buffers from within our kernels. This is done through **buffer
accessors**, which we will create in a minute.

> Attentive readers might have noticed that unlike in SYCL, Celerity command
> groups capture surrounding variables _by value_ rather than by reference.
> For more information on why this is important, see [Common
> Pitfalls](pitfalls.md).

The actual kernel code that will be executed on our compute device(s) resides
within the last argument to the `celerity::handler::parallel_for` function -
again concisely written as a lambda expression. Let us continue by fleshing
out the kernel code. Replace the TODO with the following code:

```cpp
int sum = r_input[{item[0] + 1, item[1]}] + r_input[{item[0] - 1, item[1]}]
        + r_input[{item[0], item[1] + 1}] + r_input[{item[0], item[1] - 1}];
dw_edge[item] = 255 - std::max(0, sum - (4 * r_input[item]));
```

This kernel computes a [discrete Laplace
filter](https://en.wikipedia.org/wiki/Discrete_Laplace_operator) - a simple
type of edge detection filter - by summing up the four pixel values along the
main axes surrounding the current result pixel and computing the difference
to the current pixel value. We then subtract the resulting value from the
maximum value a `uint8_t` can store (255) in order to get a white image with
black edges. The current pixel position is described by the
`cl::sycl::item<2>` we receive as an argument to our kernel function. This
two-dimensional item corresponds to a `y/x` position in our input and output
images and can be used to index into the respective buffers. However, we're
not using the buffers directly; instead we are indexing into the
aforementioned buffer accessors. Let's create these now - replace the TODO
before the kernel function with the following:

```cpp
auto r_input = input_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));
auto dw_edge = edge_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
```

If you have worked with SYCL before, these buffer accessors will look
familiar to you. The template parameter is called the **access mode** and
declares the type of access we inted to make on each buffer: We want to
`read` from our `input_buf`, and want to write to our `edge_buf`. While
there is a `write` access mode, we do not care at all about preserving any of
the previous contents of `edge_buf`, which is why we choose to discard them
and use the `discard_write` access mode.

So far everything works exactly as it would in a SYCL application. However,
there is a second parameter passed into the `celerity::buffer::get_access`
function that is not present in its SYCL counterpart. In fact, this parameter
represents one of Celerity's most important API additions: While access modes
tell the runtime system how a kernel intends to access a buffer, it does not
include any information about _where_ a kernel will access said buffer. In
order for Celerity to be able to split a single kernel execution across
potentially many different worker nodes, it needs to know how each of those
**kernel chunks** will interact with the input and output buffers of a kernel
-- i.e., which node requires which parts of the input, and produces which
parts of the output. This is where Celerity's so-called **range mappers**
come into play.

Let us first discuss the range mapper for `edge_buf`, as it represents the
simpler of the two cases. Looking at the kernel function, you can see that
for each invocation of the kernel -- i.e., for each work item, we only ever
access the output buffer once: at exactly the current location represented by
the `item`. This means there exists a one-to-one mapping of the kernel index
and the accessed buffer index. For this reason we pass a
`celerity::access::one_to_one<2>` range mapper (where the `2` simply means
that we are operating on two-dimensional kernels and buffers).

The range mapper for our `input_buf` is a bit more complicated, but not by
much: Remember that for computing the Laplace filter, we are summing up the
pixel values of the four surrounding pixels along the main axes and
calculating the difference to the current pixel. This means that in addition
to reading the pixel value associated with each item, each kernel thread also
reads a 1-pixel _neighborhood_ around the current item. This being another
very common pattern, it can be expressed with the
`celerity::access::neighborhood` range mapper. The parameters `(1, 1)`
signify that we want to access a 1-item boundary in each dimension
surrounding the current work item.

> While we are using built-in range mappers provided by the Celerity API,
> they can in fact also be user-defined functions! For more information on
> range mappers, see [Range Mappers](range-mappers.md).

Lastly, there are two more things of note for the call to `parallel_for`: The
first is the **kernel name**. Just like in SYCL, each kernel function in
Celerity has to have a unique name in the form of a template type parameter.
Here we chose `MyEdgeDetectionKernel`, but this can be anything you like.
Finally, the first two parameters to the `parallel_for` function tell
Celerity how many individual GPU threads (or work items) we want to execute.
In our case we want to execute one thread for each pixel of our image, except
for a 1-pixel border on the outside of the image - which is why we subtract
2 from our image size in both dimensions and additionally specify the
execution offset of `(1, 1)`. _Why_ this is a good idea is left as an
exercise for the reader ;-).

...and that's it, we successfully submitted a kernel to compute an edge
detection filter on our input image and store the result in an output buffer.
The only thing that remains to do now is to save the resulting image back to
a file.

## Saving The Result

To write the image resulting from our kernel execution back to a file, we need
to pass the contents of `edge_buf` back to the host. Similar to SYCL 2020,
Celerity offers _host tasks_ for this purpose. In the distributed memory setting,
we opt for the simple solution of transferring the entire image to one node
and writing the output file from there.

Just like the _compute tasks_ we created above by calling
`celerity::handler::parallel_for`, we can instantiate a host task on the command group
handler by calling `celerity::handler::host_task`. Add the following code at the end of
your `main()` function:


```cpp
queue.submit([=](celerity::handler& cgh) {
    auto out = edge_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
    cgh.host_task(celerity::on_master_node, [=]() {
        stbi_write_png("result.png", img_width, img_height, 1, out.get_pointer(), 0);
    });
});
```

Just as in compute kernel command groups, we first obtain accessors for the
buffers we want to operate on within this task. Since we need access
to the entire buffer, we pass an instance of the `all` range mapper.

Then we supply the code to be run on the host as a lambda function to
`celerity::handler::host_task`. As the tag `celerity::on_master_node`
implies, we select the overload that calls our host task on a single node
-- the master node. Since the code is executed on the host, we are able to
use it for things such as result verification and I/O. In this case, we call
`stbi_write_png` to write our resulting image into a file called `result.png`.

> **Note:** While master-node tasks are easy to use, they do not scale
> to larger problems. For real-world applications, transferring all data
> to a single node may be either prohibitively expensive or impossible
> altogether. Instead, _collective host tasks_ can be used to perform distributed
> I/O with libraries like HDF5. This feature is currently experimental.

## Running The Application

After you've built the executable, you can try and run it by passing an image
file as a command line argument like so:

```
./edge_detection ./my_image.jpg
```

If all goes well, a result file named `result.png` should then be located in
your working directory.

Since Celerity applications are built on top of MPI internally, you can now
also try and run multiple nodes:

```
mpirun -n 4 edge_detection ./my_image.jpg
```
