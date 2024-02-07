#!/bin/bash

set -o pipefail -o nounset

if [[ ! -d src ]]; then
    echo "Warning: This script should be run from within the repository root directory" >&2
fi

if [[ ! -x "$(which clang-format)" ]]; then
    echo "Error: clang-format is not installed" >&2
    exit 1
fi

SOURCES=$(find examples include src test \( -name "*.h" -o -name "*.cc" \) ! -name "stb*")

for s in $SOURCES; do
    # Since clang-format does not provide an option to check whether formatting is required,
    # we use the XML replacements output together with grep as a workaround.
    NUM_REPLACEMENTS=$(clang-format -output-replacements-xml "$s" | grep -v -c -E "<(/?replacements|\?xml)")
    if [[ $NUM_REPLACEMENTS -ne 0 ]]; then
        echo "$s"
    fi
done
