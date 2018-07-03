#!/usr/bin/env ruby
# Usage: ruby plot.rb wave_sim_result.csv
require "gnuplot"
require "csv"
require "parallel"

data = CSV.read(ARGV[0])

work = (1..(data.size - 1)).to_a

Parallel.each(work) { |i|

    x = Array.new
    y = Array.new
    z = Array.new

    data[i].each_index do |index|
        if index == 0
            next
        end
        element = data[i][index]

        pos = data[0][index].split(":")
        if y[-1] != pos[0]
            x.push nil
            y.push nil
            z.push nil
        end
        x.push pos[1]
        y.push pos[0]
        z.push element
    end

    Gnuplot.open do |gp|
        Gnuplot::SPlot.new(gp) do |plot|
            plot.grid
            plot.zrange "[-0.5:1]"
            plot.cbrange "[-0.5:1]"
            plot.palette "defined (0 \"blue\", 0.33 \"green\", 0.66 \"red\", 1 \"yellow\")"
            plot.ylabel "y"
            plot.xlabel "x"
            plot.zlabel "u"
            plot.tics "font \", 10\""
            plot.title "Wave Simulation"
            plot.terminal "pngcairo"
            plot.output "wave_sim_plot_#{i}.png"

            plot.data << Gnuplot::DataSet.new([x, y, z]) do |ds|
                ds.with = "pm3d"
                ds.notitle
            end
        end
    end
}
