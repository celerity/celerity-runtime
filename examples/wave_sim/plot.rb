#!/usr/bin/env ruby
# Usage: ruby plot.rb [wave_sim_result.bin [-keep]]

require "gnuplot"
require "parallel"
require "fileutils"
require 'tmpdir'

# check if ffmpeg is available
`ffmpeg -version`
if !$?.success?
    puts "'ffmpeg' not found in path (required)"
    exit
end

PREFIX = "#{Dir.tmpdir()}/wave_sim_plot_"

keep_pngs = ARGV.size>1 && ARGV[1] == "-keep" 

filename = ARGV.size>0 ? ARGV[0] : "wave_sim_result.bin"
data = File.read(filename)

N, T, *values = data.unpack("QQf*")

puts "Generating #{T} animation frames..."

# gnuplot wants a "nil" element whenever a new line starts
# and also needs exhaustive indexing of every element

# build the x and y indices
x = ([nil, (0...N).map(&:to_s)] * N).flatten
y = (0...N).map{|i| [nil] + [i.to_s]*N}.flatten

generated_pngs = Parallel.map(1..(T - 1)) { |i|
    fn = "#{PREFIX}#{i}.png"

    # extract the relevant data range and add in the required "nil"s
    n2d = N*N
    z = values[i*n2d...(i+1)*n2d].map &:to_s
    z = z.each_slice(N).inject([]) {|a,i| a + [nil] + i}

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
            plot.output fn

            plot.data << Gnuplot::DataSet.new([x, y, z]) do |ds|
                ds.with = "pm3d"
                ds.notitle
            end
        end
    end

    fn
}

# make animation
puts "Generating video..."
image_string = generated_pngs.map{|p| "\'#{p}\'"}.join(" ")
ffmpeg_output = `cat #{image_string} | ffmpeg -y -framerate 60 -f image2pipe -i - -c:v libvpx-vp9 -crf 15 -b:v 0 -pix_fmt yuv444p wave_sim.mp4 2>&1`
if !$?.success?
    puts "'ffmpeg' command failed:\n#{ffmpeg_output}"
    exit
end

# cleanup
FileUtils.rm generated_pngs unless keep_pngs
