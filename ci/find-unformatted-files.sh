#!/bin/bash

if [[ ! -d src ]]; then
    echo "Warning: This script should be run from within the repository root directory" 1>&2
fi

SOURCES=$(find examples include src test \( -name "*.h" -o -name "*.cc" \) ! -name "stb*")

for s in $SOURCES; do
    # Since clang-format does not provide an option to check whether formatting is required,
    # we use the XML replacements output together with grep as a workaround.
    NUM_REPLACEMENTS=$(clang-format-8 -output-replacements-xml "$s" | grep -v -c -E "<(/?replacements|\?xml)")
    if [[ $NUM_REPLACEMENTS -ne 0 ]]; then
        echo $s
    fi
done
