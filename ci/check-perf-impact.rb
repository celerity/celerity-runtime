# This script is used to check if a PR has a significant impact on the performance of the
# Celerity benchmarks. It is intended to be run as a GitHub action as part of CI.
# It expects benchmark results to be present in a csv file (see 'BENCH_FN' below).
# It will generate a message which can be posted as a comment to the PR, including a list of
# benchmarks which were significantly affected, and corresponding box plots.

# Note that if you want to test it outside the GitHub action context, you need to provide some
# result data and a baseline branch name. You probably also want to clean up the generated pngs.
# Example run:
# $ GITHUB_BASE_REF=origin/master ruby ci/check-perf-impact.rb
# $ rm box_*.png

require 'csv'
require 'gruff'
require 'json'
require 'digest'

# Monkey patch Gruff to allow left-aligning labels in the legend; only allow one entry per line.
module Gruff
  class Base
    def draw_legend
      return if @hide_legend

      legend_labels = store.data.map(&:label)
      legend_square_width = @legend_box_size # small square with color of this item
      line_height = [legend_caps_height, legend_square_width].max + @legend_margin

      current_y_offset = begin
        if @legend_at_bottom
          @graph_bottom + @legend_margin + labels_caps_height + @label_margin + (@x_axis_label ? (@label_margin * 2) + marker_caps_height : 0)
        else
          hide_title? ? @top_margin + @title_margin : @top_margin + @title_margin + title_caps_height
        end
      end

      current_x_offset = @left_margin

      legend_labels.each_with_index do |legend_label, index|
        unless legend_label.empty?
          legend_label_width = calculate_width(@legend_font, legend_label)

          # Draw label
          text_renderer = Gruff::Renderer::Text.new(renderer, legend_label, font: @legend_font)
          text_renderer.add_to_render_queue(legend_label_width,
                                            legend_square_width,
                                            current_x_offset + (legend_square_width * 1.7),
                                            current_y_offset,
                                            Magick::WestGravity) # Change gravity to Magick::WestGravity for left alignment

          # Now draw box with color of this dataset
          rect_renderer = Gruff::Renderer::Rectangle.new(renderer, color: store.data[index].color)
          rect_renderer.render(current_x_offset,
                               current_y_offset,
                               current_x_offset + legend_square_width,
                               current_y_offset + legend_square_width)

          current_y_offset += line_height
        end
      end
    end

    def calculate_legend_height
      return 0.0 if @hide_legend

      legend_labels = store.data.map(&:label)
      line_height = [legend_caps_height, @legend_box_size].max

      # Divide by two because we always draw two box plots for each benchmark (before / after); one has an empty label
      ((line_height * legend_labels.count) + (@legend_margin * (legend_labels.count - 1))) / 2
    end
  end
end

# information regarding the benchmark file
BENCH_FN = 'ci/perf/gpuc2_bench.csv'
NAME_COL_1 = "test case"         # first name column
NAME_COL_2 = "benchmark name"    # second name column
TAG_COL = "tags"                 # tag column (quoted, comma separated list of tags)
RAW_DATA_COL = "raw"             # raw data column (array of runs)

# customizing chart generation
MAX_CHARTS_PER_IMAGE = 10   # maximum number of comparisons in a single image
CHART_COLOR_WHEEL=[         # colors to use for boxes (cyclic)
'#AA1111','#BB4444','#11AA11','#44BB44','#1111AA','#4444BB',
'#AAAA11','#BBBB44','#AA11AA','#BB44BB','#11AAAA','#44BBBB',
'#111111','#444444']
GRAPH_WIDTH = 1200          # width of the generated image
THUMB_WIDTH = 120           # width of the thumbnail shown in the PR comment

# reporting thresholds for explicitly calling out benchmarks
THRESHOLD_SLOW = 1.25
THRESHOLD_FAST = 1 / THRESHOLD_SLOW
# graphing and per-group thresholds (should be smaller than the above)
MINOR_THRESHOLD_SLOW = 1.1
MINOR_THRESHOLD_FAST = 1 / MINOR_THRESHOLD_SLOW
# a small offset to reduce the reporting impact of really short-running (nanosecond-range) benchmarks
FLAT_THRESHOLD_OFFSET = 10
MAX_BENCHMARKS_TO_LIST = 3  # if more than this number of benchmarks is affected, just report the count

# file name for the message that will be posted
MESSAGE_DIR = "#{ENV['GITHUB_WORKSPACE']}/"
MESSAGE_FN = "check_perf_message.txt"

# check if the expected files and env variables are present
if !File.exist?(BENCH_FN)
  puts "Benchmark file #{BENCH_FN} not found.\nExecute this script from the repo root directory."
  exit(-1)
end
if !ENV.key?("GITHUB_BASE_REF")
  puts "GITHUB_BASE_REF needs to be set"
  exit(-1)
end

# support "remote/branch" format for GITHUB_BASE_REF
github_remote = "origin"
github_base_ref = ENV['GITHUB_BASE_REF']
if github_base_ref =~ /(.*?)\/(.*)/
  github_remote = $1
  github_base_ref = $2
end

# prevent duplicate comments for the same benchmark results (by comparing csv hash)
bench_file_digest = Digest::MD5.file(BENCH_FN).hexdigest
if /Check-perf-impact results:[^(]*\((.*)\)/ =~ ENV["PREV_COMMENT_BODY"]
  if bench_file_digest == $1
    puts "Same csv already processed, early exit"
    `echo "done=true" >> $GITHUB_OUTPUT`
    exit
  end
end

# in this case, we are just being invoked again to append the previously uploaded images to the message
if File.exist?(MESSAGE_FN)
  throw "Expected PLOT_IMG_URLS environment variable to be set" if !ENV.key?("PLOT_IMG_URLS")
  exit if ENV["PLOT_IMG_URLS"].empty? # no images were generated, i.e. there was no change
  img_urls = JSON.parse(ENV["PLOT_IMG_URLS"])
  File.open(MESSAGE_FN, 'a') do |file|
    file.puts
    img_urls.each do |img_url|
      file.print "[<img src=\"#{img_url}\" width=\"#{THUMB_WIDTH}\">](#{img_url}) "
    end
    file.puts
  end
  exit
end

# helper function which retrieves benchmark data from the csv at a specific git version (or the current one)
# returns a map from [category,benchmark_name] to raw data array, and the full data set
def get_data_for_version(version = nil)
  csv_file_contents = IO.read(BENCH_FN)
  # if a different version is supplied, get that data instead
  if version
    cmd = "git show #{version}:#{BENCH_FN}"
    csv_file_contents = `#{cmd}`
    throw "failed git checkout (cmd: #{cmd})!" unless $?.success?
  end
  # parse the data
  data = CSV.parse(csv_file_contents, headers: true)
  bench_data_map = {}
  data.each do |row|
    name = row[NAME_COL_1]+" / "+row[NAME_COL_2]
    # determine the category of the benchmark
    # (this is the tag which starts with "group:", without the prefix)
    category = row[TAG_COL].split(",").find { |tag| tag.start_with?("group:") }
    if category.nil? || category.empty?
      puts "\e[33mWARNING\e[0m: benchmark #{name} has no category - tags: #{row[TAG_COL]}!"
      category = "group:other"
    end
    category = category.delete_prefix("group:")
    raw_data = row[RAW_DATA_COL].delete_prefix('"').delete_suffix('"').split(",").map(&:to_f)
    bench_data_map[[category,name]] = raw_data
  end
  return bench_data_map, data
end

# retrieve current and previous data from respective csv files

`git fetch #{github_remote} #{github_base_ref}`
throw "failed git fetch for #{github_remote}/#{github_base_ref}!" unless $?.success?
new_data_map, new_data = get_data_for_version()
old_data_map, old_data = get_data_for_version(github_remote + "/" + github_base_ref)

# build a list of all categories in the new data
categories = new_data_map.keys.map { |k| k[0] }.uniq.sort

# statistics utilities
def mean(array)
  return nil if array.empty?
  array.reduce(:+) / array.size
end
def median(array)
  return nil if array.empty?
  sorted = array.sort
  len = sorted.length
  (sorted[(len - 1) / 2] + sorted[len / 2]) / 2.0
end
def scalar_add(array, val)
  array.map { |elem| elem+val}
end

significantly_slower_benchmarks = []
significantly_faster_benchmarks = []
relative_times_per_category = Hash.new { |h,k| h[k] = [] }

# perform further analysis and generate box plots if the data changed
if new_data != old_data
  $wheel_col_idx = 0
  def get_wheel_color
    ret = CHART_COLOR_WHEEL[$wheel_col_idx]
    $wheel_col_idx += 1
    $wheel_col_idx = 0 if $wheel_col_idx >= CHART_COLOR_WHEEL.size
    return ret
  end

  in_chart = false
  significant_perf_improvement_in_this_chart = false
  significant_perf_reduction_in_this_chart = false
  cur_chart_start_mean = 0
  cur_chart_idx = 0
  cur_img_idx = 0
  cur_chart_category = nil
  consecutive_category_count = 1
  g = nil

  # closure for completing the current in-progress chart
  finish_chart = Proc.new do
    # generate image
    img = g.to_image()
    # if there was a significant change, add border to image
    if significant_perf_improvement_in_this_chart && significant_perf_reduction_in_this_chart
      img.border!(8, 8, '#FFFF00')
    elsif significant_perf_improvement_in_this_chart
      img.border!(8, 8, '#00FF00')
    elsif significant_perf_reduction_in_this_chart
      img.border!(8, 8, '#FF0000')
    end
    img.write("box_%03d.png" % cur_img_idx)
    cur_img_idx += 1
    in_chart = false
  end

  all_keys = old_data_map.keys | new_data_map.keys
  all_keys.each do |bench_key|
    # gather some important information
    old_bench_raw = old_data_map[bench_key] || nil
    new_bench_raw = new_data_map[bench_key] || nil
    is_new_or_removed = old_bench_raw.nil? || new_bench_raw.nil?
    bench_category = bench_key[0]
    bench_name = bench_key[1]
    if old_bench_raw.nil?
      bench_name = "ADDED: " + bench_name
    elsif new_bench_raw.nil?
      bench_name = "REMOVED: " + bench_name
    end
    # If both old and new data is available, we default to old - this is an arbitrary choice
    bench_mean = old_bench_raw.nil? ? mean(new_bench_raw) : mean(old_bench_raw)

    # check if there was a highly significant difference
    rel_difference = 0
    if !is_new_or_removed
      new_median = median(scalar_add(new_bench_raw, FLAT_THRESHOLD_OFFSET))
      old_median = median(scalar_add(old_bench_raw, FLAT_THRESHOLD_OFFSET))
      rel_difference = new_median / old_median
      relative_times_per_category[bench_category] << rel_difference
      # we output these for easy inspection in the CI log
      puts "%3.2f <= %s" % [rel_difference, bench_name]
      if rel_difference > THRESHOLD_SLOW
        significantly_slower_benchmarks << bench_name
      elsif rel_difference < THRESHOLD_FAST
        significantly_faster_benchmarks << bench_name
      end
    end

    # add old and new boxes to chart if they are significant according to the charting thresholds
    if is_new_or_removed || rel_difference > MINOR_THRESHOLD_SLOW || rel_difference < MINOR_THRESHOLD_FAST
      # finish current chart if the category has changed, the relative difference on the y-axis is too large,
      # or we've reached the maximum number of plots per image
      ratio = [cur_chart_start_mean, bench_mean].max / [cur_chart_start_mean, bench_mean].min
      if in_chart && (cur_chart_category != bench_category || ratio > 50 || cur_chart_idx >= MAX_CHARTS_PER_IMAGE)
        finish_chart.()
      end

      # start a new chart if necessary
      if !in_chart
        g = Gruff::Box.new(GRAPH_WIDTH)
        g.theme_pastel
        g.title = bench_category
        g.marker_font_size = 15
        g.legend_at_bottom = true
        g.legend_font_size = 9
        g.legend_box_size = 10
        g.legend_margin = 2
        g.y_axis_label = "Time (nanoseconds)"
        g.stroke_width = 0.5

        in_chart = true
        significant_perf_improvement_in_this_chart = false
        significant_perf_reduction_in_this_chart = false
        cur_chart_start_mean = bench_mean
        cur_chart_idx = 0
        if cur_chart_category == bench_category
          consecutive_category_count += 1
          g.title = bench_category + " (#{consecutive_category_count})"
        else
          consecutive_category_count = 1
        end
        cur_chart_category = bench_category
      end

      if !is_new_or_removed && rel_difference > THRESHOLD_SLOW
        significant_perf_reduction_in_this_chart = true
      elsif !is_new_or_removed && rel_difference < THRESHOLD_FAST
        significant_perf_improvement_in_this_chart = true
      end

      # plot the data
      g.data bench_name, old_bench_raw, get_wheel_color
      g.data nil, new_bench_raw, get_wheel_color
      cur_chart_idx += 1
    end
  end
  # don't forget to finish the last chart!
  finish_chart.()
end

# helper for message generation
def report_benchmark_list(list)
  if list.size <= MAX_BENCHMARKS_TO_LIST
    list.join(", ")
  else
    "#{list.size} individual benchmarks affected"
  end
end

# generate PR message
message = "**Check-perf-impact results:** (#{bench_file_digest})\n\n"
if new_data == old_data
  message += ":question: No new benchmark data submitted. :question:  \n"
  message += "Please re-run the microbenchmarks and include the results if your commit could potentially affect performance."
else
  if !significantly_slower_benchmarks.empty?
    message += ":warning: Significant **slowdown** (>%3.2fx) in some microbenchmark results: " % THRESHOLD_SLOW
    message += report_benchmark_list(significantly_slower_benchmarks) + "  \n"
  end
  if !significantly_faster_benchmarks.empty?
    message += ":rocket: Significant **speedup** (<%3.2fx) in some microbenchmark results: " % THRESHOLD_FAST
    message += report_benchmark_list(significantly_faster_benchmarks) + "  \n"
  end
  added_benchmarks = new_data_map.keys - old_data_map.keys
  if !added_benchmarks.empty?
    message += ":heavy_plus_sign: **Added** microbenchmark(s): "
    message += report_benchmark_list(added_benchmarks) + "  \n"
  end
  removed_benchmarks = old_data_map.keys - new_data_map.keys
  if !removed_benchmarks.empty?
    message += ":heavy_minus_sign: **Removed** microbenchmark(s): "
    message += report_benchmark_list(removed_benchmarks) + "  \n"
  end
  if significantly_slower_benchmarks.empty? && significantly_faster_benchmarks.empty?
    message += ":heavy_check_mark: No significant performance change in the microbenchmark set. You are good to go!  \n"
  end
  message += "\nRelative execution time per category: (mean of relative medians)\n"
  categories.each do |category|
    category_mean = mean(relative_times_per_category[category])
    if category_mean.nil?
      message += "* **#{category}** : **new** :star2:\n"
    else
      symbol = category_mean > MINOR_THRESHOLD_SLOW ? ":warning:" : (category_mean < MINOR_THRESHOLD_FAST ? ":rocket:" : "")
      message += "* **#{category}** : **%3.2fx** %s\n" % [category_mean, symbol]
    end
  end
end
puts "\n" + message

# write message to workspace file for subsequent step
if File.writable?(MESSAGE_DIR)
  File.write(MESSAGE_DIR + MESSAGE_FN, message)
else
  puts "\nCannot write to '#{MESSAGE_DIR}', skipping message generation"
end
