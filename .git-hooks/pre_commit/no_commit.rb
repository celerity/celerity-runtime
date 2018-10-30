# This is a small pre-commit hook, checking for "NOCOMMIT" comments in staged files.
# If a "NOCOMMIT" comment is found, the commit is aborted.
module Overcommit::Hook::PreCommit
  class NoCommit < Base
    def run
      error_lines = []
      root_dir = Pathname.new(File.dirname(__FILE__)).join('../../')

      applicable_files.each do |file|

        File.open(file, 'r').each_with_index do |line, index|
          if line.include? "NOCOMMIT"
            relative_path = Pathname.new(file).relative_path_from(root_dir).to_s
            message = "#{relative_path}:#{index + 1}: NOCOMMIT found"
            error_lines << message
          end
        end
      end

      return :fail, error_lines.join("\n") if error_lines.any?
      :pass
    end
  end
end
