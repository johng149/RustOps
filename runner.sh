#!/bin/bash

# Function to prompt for input with a message
prompt_input() {
    local message=$1
    local variable_name=$2
    local default_value=$3
    local input_value

    if [ -n "$default_value" ]; then
        read -p "$message [$default_value]: " input_value
        # If input is empty, use the default value
        if [ -z "$input_value" ]; then
            input_value=$default_value
        fi
    else
        read -p "$message: " input_value
        # Ensure required input is provided
        while [ -z "$input_value" ]; do
            echo "Input cannot be empty."
            read -p "$message: " input_value
        done
    fi
    # Assign the value to the variable name passed as argument
    eval "$variable_name='$input_value'"
}

# Ask user which command to run
echo "Which command would you like to run?"
echo "1) chunk - Process an image into chunks"
echo "2) benchmark - Run a benchmark of optimize and predict"
read -p "Enter choice (1 or 2): " choice

# Validate choice
if [[ "$choice" != "1" && "$choice" != "2" ]]; then
    echo "Invalid choice. Exiting."
    exit 1
fi

# Build the base command parts
cargo_cmd="cargo run"
app_args=""
prefix_cmd=""

# Handle chunk command
if [ "$choice" == "1" ]; then
    echo "--- Configuring Image Chunking ---"
    prompt_input "Path to the input image file" input_path
    prompt_input "Target width for the image" target_width
    prompt_input "Target height for the image" target_height
    prompt_input "Width of each chunk" chunk_width
    prompt_input "Height of each chunk" chunk_height
    prompt_input "Directory to save the output chunks" output_dir

    # Construct the chunk arguments
    app_args="chunk --input \"$input_path\" --target-width $target_width --target-height $target_height --chunk-width $chunk_width --chunk-height $chunk_height --output-dir \"$output_dir\""

# Handle benchmark command
elif [ "$choice" == "2" ]; then
    echo "--- Configuring Benchmark ---"
    prompt_input "Batch size for the input tensor" batch_size "6"
    prompt_input "Number of chunks (fields/nodes in outer layer)" num_chunks "8"
    prompt_input "Dimension of each chunk (feature dimension)" chunk_dim "7"
    read -p "Use release mode (optimized build)? [y/N]: " use_release
    read -p "Run with samply profiler? [y/N]: " use_samply

    # Add --release flag if requested
    if [[ "${use_release,,}" == "y" || "${use_release,,}" == "yes" ]]; then
        cargo_cmd="$cargo_cmd --release"
    fi

    # Add samply prefix if requested
    if [[ "${use_samply,,}" == "y" || "${use_samply,,}" == "yes" ]]; then
        prefix_cmd="samply record"
        echo "Note: Ensure samply is installed and configured."
    fi

    # Construct the benchmark arguments
    app_args="benchmark --batch-size $batch_size --num-chunks $num_chunks --chunk-dim $chunk_dim"
fi

# Combine the command parts
# Note the structure: [prefix] cargo run [--release] -- [app_args]
cmd="$prefix_cmd $cargo_cmd -- $app_args"

# Execute the command
echo "Running command:"
# Use printf for safer printing of potentially complex commands
printf "%s\n" "$cmd"
echo "---"
eval "$cmd"

echo "---"
echo "Script finished."