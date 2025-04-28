use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use image::{GenericImageView, ImageBuffer, Rgb, imageops};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use ndarray_npy::write_npy;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::path::PathBuf;
use std::time::Instant;

// Import necessary functions from your library
use RustOps::functions::{optimize, predict}; // Assuming optimize and predict are in lib.rs or functions/mod.rs

// --- Define Layer Structure (Example based on root_down.py) ---
const LAYER_0_MEMS: usize = 16;
const LAYER_1_NODES: usize = 4;
const LAYER_1_MEMS: usize = 32;
const LAYER_2_NODES: usize = 1;
const LAYER_2_MEMS: usize = 10;

// --- Optimization Parameters ---
const T_INITIAL: usize = 0;
const ALPHA: f32 = 16.0;
const RHO: f32 = 1e-8;
const EPS: f32 = 1e-6;
const COEFF: f32 = 0.5;
const MARK: i64 = -2; // Marker for growth_argmaxi

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Process an image into chunks.
    Chunk(ChunkArgs),
    /// Run a benchmark of the optimize and predict functions.
    Benchmark(BenchmarkArgs),
}

#[derive(Parser, Debug)]
struct ChunkArgs {
    /// Path to the input image file.
    #[arg(short, long)]
    input: PathBuf,

    /// Target width for the image.
    #[arg(long)]
    target_width: u32,

    /// Target height for the image.
    #[arg(long)]
    target_height: u32,

    /// Width of each chunk.
    #[arg(long)]
    chunk_width: u32,

    /// Height of each chunk.
    #[arg(long)]
    chunk_height: u32,

    /// Directory to save the output chunks.
    #[arg(short, long)]
    output_dir: PathBuf,
}

#[derive(Parser, Debug)]
struct BenchmarkArgs {
    /// Batch size for the input tensor.
    #[arg(long, default_value_t = 6)]
    batch_size: usize,

    /// Number of chunks (fields/nodes in the outer layer).
    #[arg(long, default_value_t = 8)]
    num_chunks: usize,

    /// Dimension of each chunk (feature dimension).
    #[arg(long, default_value_t = 7)]
    chunk_dim: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chunk(args) => run_chunking(args),
        Commands::Benchmark(args) => run_benchmark(args),
    }
}

// --- Image Chunking Logic ---
fn run_chunking(args: ChunkArgs) -> Result<()> {
    println!("--- Running Image Chunking ---");
    // --- 1. Load Image ---
    println!("Loading image from: {:?}", args.input);
    let img = image::open(&args.input)
        .with_context(|| format!("Failed to open image: {:?}", args.input))?;
    println!("Original dimensions: {:?}", img.dimensions());

    // --- 2. Resize Image ---
    println!(
        "Resizing image to: {}x{}",
        args.target_width, args.target_height
    );
    let img_rgb = img.to_rgb8();
    let resized_img = imageops::resize(
        &img_rgb,
        args.target_width,
        args.target_height,
        imageops::FilterType::Lanczos3,
    );
    let (resized_w, resized_h) = resized_img.dimensions();
    println!("Resized dimensions: {}x{}", resized_w, resized_h);

    // --- 3. Calculate Padding & Padded Dimensions ---
    let chunk_w = args.chunk_width;
    let chunk_h = args.chunk_height;
    let num_channels = 3; // Assuming RGB

    let padded_w = (resized_w + chunk_w - 1) / chunk_w * chunk_w;
    let padded_h = (resized_h + chunk_h - 1) / chunk_h * chunk_h;
    println!("Padded dimensions: {}x{}", padded_w, padded_h);

    // --- 4. Create Padded Image ---
    let mut padded_img = ImageBuffer::<Rgb<u8>, _>::new(padded_w, padded_h);
    imageops::overlay(&mut padded_img, &resized_img, 0, 0);

    // --- 5. Extract Chunks ---
    let num_chunks_x = padded_w / chunk_w;
    let num_chunks_y = padded_h / chunk_h;
    let total_chunks = (num_chunks_x * num_chunks_y) as usize;
    let chunk_dim = (chunk_w * chunk_h * num_channels) as usize;
    println!(
        "Total chunks: {}, Chunk dimension: {}",
        total_chunks, chunk_dim
    );

    let mut all_chunk_data: Vec<u8> = Vec::with_capacity(total_chunks * chunk_dim);

    for cy in 0..num_chunks_y {
        for cx in 0..num_chunks_x {
            let start_x = cx * chunk_w;
            let start_y = cy * chunk_h;
            let chunk_view = imageops::crop_imm(&padded_img, start_x, start_y, chunk_w, chunk_h);
            for pixel in chunk_view.pixels() {
                all_chunk_data.extend_from_slice(&pixel.2.0);
            }
        }
    }

    // --- 6. Create ArrayD ---
    let flat_array = Array::from_vec(all_chunk_data);
    let chunked_array = flat_array.into_shape((total_chunks, chunk_dim))?;
    let final_array: ArrayD<u8> = chunked_array.insert_axis(Axis(0)).into_dyn();
    println!("Final array shape: {:?}", final_array.shape());

    // --- 7. Save Output ---
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", args.output_dir))?;
    let output_filename = args.output_dir.join("image_chunks.npy");
    println!("Saving chunks to: {:?}", output_filename);
    write_npy(&output_filename, &final_array)
        .with_context(|| format!("Failed to write NPY file: {:?}", output_filename))?;

    println!("Image chunking complete.");
    Ok(())
}

// --- Benchmark Logic ---
fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    println!("--- Running Benchmark ---");
    println!("Args: {:?}", args);

    // --- Type Aliases ---
    type T = f32; // Float type for activations/weights
    type U = i64; // Integer type for counts

    // --- Initialize Network State ---
    let fields = args.num_chunks;
    let field_dim = args.chunk_dim;

    // Layer 0
    let layer0_shape = IxDyn(&[fields, LAYER_0_MEMS, field_dim]);
    let layer0_mm = ArrayD::<T>::from_elem(layer0_shape, 0.5);
    let layer0_counts = ArrayD::<U>::ones(IxDyn(&[fields, LAYER_0_MEMS]));

    // Layer 1
    let prev_nodes_1 = fields;
    let prev_mems_1 = LAYER_0_MEMS;
    if prev_nodes_1 % LAYER_1_NODES != 0 {
        anyhow::bail!(
            "Layer 0 nodes ({}) must be divisible by Layer 1 nodes ({})",
            prev_nodes_1,
            LAYER_1_NODES
        );
    }
    let children_per_node_1 = prev_nodes_1 / LAYER_1_NODES;
    let layer1_shape = IxDyn(&[
        LAYER_1_NODES,
        children_per_node_1,
        LAYER_1_MEMS,
        prev_mems_1,
    ]);
    let layer1_mm = ArrayD::<T>::from_elem(layer1_shape, 0.0); // Typically initialized differently, using 0 for simplicity
    let layer1_counts = ArrayD::<U>::ones(IxDyn(&[LAYER_1_NODES, LAYER_1_MEMS]));

    // Layer 2
    let prev_nodes_2 = LAYER_1_NODES;
    let prev_mems_2 = LAYER_1_MEMS;
    if prev_nodes_2 % LAYER_2_NODES != 0 {
        anyhow::bail!(
            "Layer 1 nodes ({}) must be divisible by Layer 2 nodes ({})",
            prev_nodes_2,
            LAYER_2_NODES
        );
    }
    let children_per_node_2 = prev_nodes_2 / LAYER_2_NODES;
    let layer2_shape = IxDyn(&[
        LAYER_2_NODES,
        children_per_node_2,
        LAYER_2_MEMS,
        prev_mems_2,
    ]);
    let layer2_mm = ArrayD::<T>::from_elem(layer2_shape, 0.0); // Typically initialized differently, using 0 for simplicity
    let layer2_counts = ArrayD::<U>::ones(IxDyn(&[LAYER_2_NODES, LAYER_2_MEMS]));

    let mut layers = vec![layer0_mm, layer1_mm, layer2_mm];
    let mut layer_counts = vec![layer0_counts, layer1_counts, layer2_counts];
    let mut t = T_INITIAL;
    let mut growth_threshold = ALPHA / (T::from(t as f32) + ALPHA);

    println!("Initialized {} layers.", layers.len());
    println!(
        "Layer 0 MM shape: {:?}, Counts shape: {:?}",
        layers[0].shape(),
        layer_counts[0].shape()
    );
    println!(
        "Layer 1 MM shape: {:?}, Counts shape: {:?}",
        layers[1].shape(),
        layer_counts[1].shape()
    );
    println!(
        "Layer 2 MM shape: {:?}, Counts shape: {:?}",
        layers[2].shape(),
        layer_counts[2].shape()
    );

    // --- Generate Initial Input ---
    let input_shape = IxDyn(&[args.batch_size, fields, field_dim]);
    let sensory_input_optim = ArrayD::<T>::random(input_shape.clone(), Uniform::new(0.0, 1.0));
    println!(
        "Generated sensory input for optimize with shape: {:?}",
        sensory_input_optim.shape()
    );

    // --- Time Optimize ---
    println!("Running optimize...");
    let start_optimize = Instant::now();
    let (updated_layers, updated_counts, next_t, next_growth_threshold) = optimize::optimize(
        &layers,             // Pass current layers (as slice)
        &layer_counts,       // Pass current counts
        sensory_input_optim, // Pass input (consumed)
        t,
        ALPHA,
        RHO,
        EPS,
        COEFF,
        growth_threshold,
        MARK,
    );
    let duration_optimize = start_optimize.elapsed();
    println!("Optimize completed in: {:?}", duration_optimize);

    // Update state
    layers = updated_layers;
    layer_counts = updated_counts;
    t = next_t;
    growth_threshold = next_growth_threshold;

    // --- Generate New Input for Predict ---
    let sensory_input_predict = ArrayD::<T>::random(input_shape, Uniform::new(0.0, 1.0));
    println!(
        "Generated sensory input for predict with shape: {:?}",
        sensory_input_predict.shape()
    );

    // --- Time Predict ---
    println!("Running predict...");
    let start_predict = Instant::now();
    let prediction = predict::predict(
        &layers,               // Pass updated layers (as slice)
        sensory_input_predict, // Pass new input (consumed)
        RHO,
        EPS,
        COEFF,
    );
    let duration_predict = start_predict.elapsed();
    println!("Predict completed in: {:?}", duration_predict);
    println!("Prediction output shape: {:?}", prediction.shape());

    println!("Benchmark complete.");
    Ok(())
}
