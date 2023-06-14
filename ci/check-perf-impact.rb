require 'csv'

BENCH_FN='ci/perf/gpuc2_bench.csv'
MEAN_COL=5

THRESHOLD_SLOW=1.1
THRESHOLD_FAST=0.9

if !File.exists?(BENCH_FN)
  puts "Benchmark file #{BENCH_FN} not found.\nExecute this script from the repo root directory."
  exit(-1)
end

def geomean(x)
  sum = x.inject(0) { |memo, v| memo + Math.log(v) }
  sum /= x.size
  Math.exp(sum).round(2)
end

def compute_geomean_for_version(version)
  cmd = "git checkout #{version} -- #{BENCH_FN}"
  `#{cmd}`
  throw "failed git checkout (cmd: #{cmd})!" if !$?.success?
  data = CSV.read(BENCH_FN)
  means = []
  data[1..].each do |row|
    means << row[MEAN_COL].to_f
  end
  return geomean(means), data
end

# compute metrics on current and previous data

base_ver = ENV['GITHUB_BASE_REF']
`git fetch origin #{base_ver}`
throw "failed git fetch for #{base_ver}!" if !$?.success?
prev_geomean, prev_data = compute_geomean_for_version("origin/" + base_ver)
post_geomean, post_data = compute_geomean_for_version("HEAD")

# generate message

message = "**Check-perf-impact results:**  \n"
if prev_data == post_data
  message += ":question: No new benchmark data submitted. :question:  \nPlease re-run the microbenchmarks and include the results if your commit could potentially affect performance."
else
  if post_geomean / prev_geomean > THRESHOLD_SLOW
    message += ":warning: Significant slowdown in microbenchmark results. Needs Investigation. :warning:"
  elsif post_geomean / prev_geomean < THRESHOLD_FAST
    message += ":rocket: Significant speedup in microbenchmark results. Nice! :rocket:"
  else
    message += ":heavy_check_mark: No significant performance change in the microbenchmark set. You are good to go!"
  end
  message += format("\n\n```\ngeomean before: %10.2f  \ngeomean after : %10.2f\n```", prev_geomean, post_geomean)
end
puts message

# write message to workspace file for subsequent step

File.write("#{ENV['GITHUB_WORKSPACE']}/check_perf_message.txt", message)
