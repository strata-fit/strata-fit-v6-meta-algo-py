#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <node_count> <data_dir> <output_nodes_env>" >&2
  exit 1
fi

node_count="$1"
data_dir="$2"
out_file="$3"

if ! [[ "$node_count" =~ ^[0-9]+$ ]]; then
  echo "node_count must be an integer" >&2
  exit 1
fi

if [ "$node_count" -lt 1 ] || [ "$node_count" -gt 8 ]; then
  echo "node_count must be between 1 and 8" >&2
  exit 1
fi

mkdir -p "$(dirname "$out_file")"

node_names=(alpha beta gamma delta epsilon zeta eta theta)
api_keys=(
  844a7d92-1cc9-4856-bf33-0613252d5b3c
  57143784-19ef-456b-94c9-ba68c8cb079b
  57143784-19ef-456b-94c9-ba68c8cb079c
  9f2f57be-4f26-46a8-b227-49d8467b8fd1
  b4ff3af3-9d05-4db2-b8ef-dcb4d14e6f7f
  1e72c7d2-589b-4e8c-9b7d-6c7cfe43b001
  3b7f7a88-42f7-4e8d-a655-2d7ea7318c02
  6d654bb0-0c56-4f13-a5ff-d957fdf64003
)

: > "$out_file"
for ((i=0; i<node_count; i++)); do
  name="${node_names[$i]}"
  key="${api_keys[$i]}"
  csv_path="$data_dir/data_bucket$((i+1)).csv"

  if [ ! -f "$csv_path" ]; then
    echo "Missing CSV for node '$name': $csv_path" >&2
    exit 1
  fi

  echo "$name|$key|$csv_path|csv|default" >> "$out_file"
done

echo "Generated $node_count node specs at $out_file"
