# Wave Simulation

This is based on the [AllScale wave simulation
example](https://github.com/allscale/allscale_api/tree/50b551b4a79ef5cf08d1c9f76e3a8051c7d47239/code/tutorials/src/adaptivegrid).

By default, the program produces a CSV file containing the resulting simulation
data. This can then be used to render an animated GIF using `gnuplot`.
Alternatively, the results can be immediately displayed in the terminal window,
by passing the `--ascii-plot` command line parameter.

## Plotting the output using GNUPlot

This requires `ruby` and the following gems: `gnuplot`, `parallel`, `rmagick`.

To produce a series of images from the generated CSV data run
`plot.rb wave_sim_result.csv`.

To combine the images into an animated GIF, run `animate.rb` in the same
directory.

