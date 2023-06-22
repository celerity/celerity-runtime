require 'csv'
require 'gruff'
require 'json'
require 'digest'

# information regarding the benchmark file
BENCH_FN = 'ci/perf/gpuc2_bench.csv'
MEAN_COL = 5       # column containing the mean value
NAME_COL_1 = 0     # first name column
NAME_COL_2 = 1     # second name column
RAW_DATA_COL = 11  # raw data column (array of runs)

# customizing chart generation
MAX_CHARTS_PER_IMAGE = 10   # maximum number of comparisons in a single image
CHART_COLOR_WHEEL=[         # colors to use for boxes (cyclic)
'#AA1111','#BB4444','#11AA11','#44BB44','#1111AA','#4444BB',
'#AAAA11','#BBBB44','#AA11AA','#BB44BB','#11AAAA','#44BBBB',
'#111111','#444444']
GRAPH_WIDTH = 1200          # width of the generated image
THUMB_WIDTH = 120           # width of the thumbnail shown in the PR comment

# reporting thresholds
THRESHOLD_SLOW = 1.1
THRESHOLD_FAST = 0.9
MAX_BENCHMARKS_TO_LIST = 3  # if more than this nubmer of benchmarks is affected, just report the count

# file name for the message that will be posted
MESSAGE_FN = "#{ENV['GITHUB_WORKSPACE']}/check_perf_message.txt"

if !File.exists?(BENCH_FN)
  puts "Benchmark file #{BENCH_FN} not found.\nExecute this script from the repo root directory."
  exit(-1)
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
if File.exists?(MESSAGE_FN) 
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

# helper function which retrieves benchmark data from the csv at a specific version
def get_data_for_version(version = nil)
  # if a different version is supplied, check it out
  if version
    cmd = "git checkout #{version} -- #{BENCH_FN}"
    `#{cmd}`
    throw "failed git checkout (cmd: #{cmd})!" unless $?.success?
  end
  # read the data
  data = CSV.read(BENCH_FN)
  bench_data_map = {}
  data[1..].each do |row|
    raw_data = row[RAW_DATA_COL].delete_prefix('"').delete_suffix('"').split(",").map(&:to_f)
    bench_data_map[row[NAME_COL_1]+" / "+row[NAME_COL_2]] = raw_data
  end
  # restore the file if it was checked out in a different version
  if version
    `git restore --staged #{BENCH_FN}`
    `git restore #{BENCH_FN}`
  end
  # return the geometric mean, benchmark data map, and full data set
  return bench_data_map, data
end

# retrieve current and previous data from respective csv files
base_ver = ENV['GITHUB_BASE_REF']
`git fetch origin #{base_ver}`
throw "failed git fetch for #{base_ver}!" unless $?.success?
new_data_map, new_data = get_data_for_version()
old_data_map, old_data = get_data_for_version("origin/" + base_ver)

significantly_slower_benchmarks = []
significantly_faster_benchmarks = []

# statistics utility
def mean(x)
  x.reduce(:+) / x.size
end

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
  significant_change_in_this_chart = false
  cur_chart_start_mean = 0
  cur_chart_idx = 0
  cur_img_idx = 0
  g = nil
  old_data_map.sort_by { |k,v| mean(v) }.each do |bench_name, old_bench_raw|
    # skip deleted benchmarks
    next unless new_data_map.key?(bench_name)

    # finish the current chart if we have reached the maximum per image
    # or if the relative y axis difference becomes too large
    if in_chart && (cur_chart_start_mean < mean(old_bench_raw) / 20 ||
                    cur_chart_idx >= MAX_CHARTS_PER_IMAGE)

      # ~ 10 subdivisions
      g.y_axis_increment = mean(old_bench_raw) / 10
      # generate image
      img = g.to_image()
      if significant_change_in_this_chart
        # if there was a significant change, add border to image
        img.border!(8, 8, '#FF0000')
      end
      img.write("box_#{cur_img_idx}.png")
      cur_img_idx += 1
      in_chart = false
    end

    # start a new chart
    if !in_chart
      g = Gruff::Box.new(GRAPH_WIDTH)
      g.theme_pastel
      g.hide_title = true
      g.marker_font_size = 15
      g.legend_at_bottom = true
      g.legend_font_size = 9
      g.legend_box_size = 10
      g.legend_margin = 2

      in_chart = true
      significant_change_in_this_chart = false
      cur_chart_start_mean = mean(old_bench_raw)
      cur_chart_idx = 0
    end

    # add old and new boxes to chart
    g.data bench_name, old_bench_raw, get_wheel_color
    new_bench_raw = new_data_map[bench_name]
    g.data nil, new_bench_raw, get_wheel_color

    # check if there was a significant difference
    rel_difference = mean(new_bench_raw) / mean(old_bench_raw)
    if rel_difference > THRESHOLD_SLOW
      significantly_slower_benchmarks << bench_name
      significant_change_in_this_chart = true
    elsif rel_difference < THRESHOLD_FAST
      significantly_faster_benchmarks << bench_name
      significant_change_in_this_chart = true
    end
    cur_chart_idx += 1
  end
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
message = "**Check-perf-impact results:** (#{bench_file_digest})  \n"
if new_data == old_data
  message += ":question: No new benchmark data submitted. :question:  \nPlease re-run the microbenchmarks and include the results if your commit could potentially affect performance."
else
  if !significantly_slower_benchmarks.empty?
    message += ":warning: Significant **slowdown** in some microbenchmark results: "
    message += report_benchmark_list(significantly_slower_benchmarks) + "  \n"
  end
  if !significantly_faster_benchmarks.empty?
    message += ":rocket: Significant **speedup** in some microbenchmark results: "
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
    message += ":heavy_check_mark: No significant performance change in the microbenchmark set. You are good to go!"
  end
end
puts message

# write message to workspace file for subsequent step
File.write(MESSAGE_FN, message)
