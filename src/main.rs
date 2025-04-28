use anyhow::{Context, Result};
use clap::Parser;
use image::{GenericImageView, ImageBuffer, Rgb, imageops};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use ndarray_npy::write_npy; // Or use image::save_buffer if saving chunks as images
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
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

fn main() -> Result<()> {
    let args = Args::parse();

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
    // Using Lanczos3 for quality resizing. Convert to RGB first.
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
    // Create a black background image
    let mut padded_img = ImageBuffer::<Rgb<u8>, _>::new(padded_w, padded_h);
    // Copy the resized image onto the padded background
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

            // Extract chunk view
            let chunk_view = imageops::crop_imm(&padded_img, start_x, start_y, chunk_w, chunk_h);

            // Flatten chunk data (row by row, pixel by pixel, channel by channel)
            for pixel in chunk_view.pixels() {
                all_chunk_data.extend_from_slice(&pixel.2.0); // pixel.0 is [R, G, B]
            }
        }
    }

    // --- 6. Create ArrayD ---
    // Convert Vec<u8> to Array
    let flat_array = Array::from_vec(all_chunk_data);
    // Reshape to (total_chunks, chunk_dim)
    let chunked_array = flat_array.into_shape((total_chunks, chunk_dim))?;
    // Add the leading dimension (for num_images = 1) and make it ArrayD
    let final_array: ArrayD<u8> = chunked_array.insert_axis(Axis(0)).into_dyn();

    println!("Final array shape: {:?}", final_array.shape());

    // --- 7. Save Output ---
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", args.output_dir))?;

    let output_filename = args.output_dir.join("image_chunks.npy"); // Or choose another name/format
    println!("Saving chunks to: {:?}", output_filename);

    // Example: Saving as a single .npy file
    write_npy(&output_filename, &final_array)
        .with_context(|| format!("Failed to write NPY file: {:?}", output_filename))?;

    // // --- Alternative: Save each chunk as an image ---
    // let mut chunk_index = 0;
    // for cy in 0..num_chunks_y {
    //     for cx in 0..num_chunks_x {
    //         let start_x = cx * chunk_w;
    //         let start_y = cy * chunk_h;
    //         let chunk_view = imageops::crop_imm(&padded_img, start_x, start_y, chunk_w, chunk_h);
    //         let chunk_filename = args.output_dir.join(format!("chunk_{}.png", chunk_index));
    //         chunk_view.to_image().save(&chunk_filename)?;
    //         chunk_index += 1;
    //     }
    // }
    // println!("Saved {} individual chunk images.", chunk_index);

    println!("Processing complete.");
    Ok(())
}
