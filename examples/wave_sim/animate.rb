#!/usr/bin/env ruby
require 'rmagick'
include Magick

images = Dir["wave_sim_plot_*.png"]
images = images.sort { |a, b| Integer(a.split('_')[3].split('.')[0]) <=> Integer(b.split('_')[3].split('.')[0]) }

if images.size > 0 then
    animation = ImageList.new(*images)
    animation.delay = 1
    animation.write("wave_sim.gif")
else
    puts "No \"wave_sim_plot_*.png\" files found"
end
