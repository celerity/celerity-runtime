# Wave Simulation

This is based on the [AllScale wave simulation
example](https://github.com/allscale/allscale_api/tree/50b551b4a79ef5cf08d1c9f76e3a8051c7d47239/code/tutorials/src/adaptivegrid).

Computes a wave simulation (2D wave equation) over multiple time steps,
optionally writing intermediate results to a CSV file which can subsequently be
plotted.

The behavior of this simulation can be altered through various command line
arguments:

- `-N <integer>` sets the size of the simulated grid (should be a power of 2).
- `-T <integer>` sets the time at the end of the simulation.
- `--dt <float>` sets the delta time increments used for each simulation step.
  This, together with `-T`, effectively dictates the number of steps.
- `--sample-rate <integer>` controls the rate at which intermediate results
  are recorded for output. For example, `--sample-rate 10` means that every
  10th time step will be recorded. Setting this to 0 means that no output file
  will be produced.

## Plotting the output using GNUPlot

This requires `ruby` and the following gems: `gnuplot`, `parallel`, `rmagick`.

To produce a series of images from the generated CSV data run
`plot.rb wave_sim_result.csv`.

To combine the images into an animated GIF, run `animate.rb` in the same
directory.

