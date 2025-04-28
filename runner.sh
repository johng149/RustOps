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

# Build the base command
base_cmd="cargo run --"

# Handle chunk command
if [ "$choice" == "1" ]; then
    echo "--- Configuring Image Chunking ---"
    prompt_input "Path to the input image file" input_path
    prompt_input "Target width for the image" target_width
    prompt_input "Target height for the image" target_height
    prompt_input "Width of each chunk" chunk_width
    prompt_input "Height of each chunk" chunk_height
    prompt_input "Directory to save the output chunks" output_dir

    # Construct the chunk command
    cmd="$base_cmd chunk --input \"$input_path\" --target-width $target_width --target-height $target_height --chunk-width $chunk_width --chunk-height $chunk_height --output-dir \"$output_dir\""

# Handle benchmark command
elif [ "$choice" == "2" ]; then
    echo "--- Configuring Benchmark ---"
    prompt_input "Batch size for the input tensor" batch_size "6"
    prompt_input "Number of chunks (fields/nodes in outer layer)" num_chunks "8"
    prompt_input "Dimension of each chunk (feature dimension)" chunk_dim "7"

    # Construct the benchmark command
    cmd="$base_cmd benchmark --batch-size $batch_size --num-chunks $num_chunks --chunk-dim $chunk_dim"
fi

# Execute the command
echo "Running command:"
echo "$cmd"
echo "---"
eval "$cmd"

echo "---"
echo "Script finished."