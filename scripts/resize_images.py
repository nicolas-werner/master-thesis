import os
import argparse
from PIL import Image
import concurrent.futures

def resize_image(input_path, output_path, max_size=2000, quality=80, convert_grayscale=False):
    """
    Resize an image to reduce its size while maintaining readability.

    Args:
        input_path: Path to the original image
        output_path: Path to save the resized image
        max_size: Maximum dimension (width or height) in pixels
        quality: JPEG quality (1-100)
        convert_grayscale: Whether to convert to grayscale
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Open the image
        img = Image.open(input_path)

        # Convert to grayscale if requested (often sufficient for document images)
        if convert_grayscale and img.mode != 'L':
            img = img.convert('L')

        # Calculate new dimensions while preserving aspect ratio
        width, height = img.size
        if width > height:
            if width > max_size:
                new_width = max_size
                new_height = int(height * (max_size / width))
        else:
            if height > max_size:
                new_height = max_size
                new_width = int(width * (max_size / height))

        # Skip resizing if image is already smaller than max_size
        if width <= max_size and height <= max_size:
            new_width, new_height = width, height

        # Resize the image
        if new_width != width or new_height != height:
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save with specified quality
        img.save(output_path, 'JPEG', quality=quality, optimize=True)

        # Calculate size reduction
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # in MB
        new_size = os.path.getsize(output_path) / (1024 * 1024)  # in MB
        return {
            'filename': os.path.basename(input_path),
            'original_size': original_size,
            'new_size': new_size,
            'reduction': (1 - (new_size / original_size)) * 100
        }
    except Exception as e:
        return {
            'filename': os.path.basename(input_path),
            'error': str(e)
        }

def process_directory(input_dir, output_dir, max_size=2000, quality=80, convert_grayscale=False, num_workers=4):
    """
    Process all images in a directory.

    Args:
        input_dir: Input directory containing original images
        output_dir: Output directory for resized images
        max_size: Maximum dimension (width or height) in pixels
        quality: JPEG quality (1-100)
        convert_grayscale: Whether to convert to grayscale
        num_workers: Number of parallel workers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]

    results = []

    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(
                resize_image,
                os.path.join(input_dir, f),
                os.path.join(output_dir, f),
                max_size,
                quality,
                convert_grayscale
            ): f for f in image_files
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            results.append(result)

            # Print progress
            if 'error' in result:
                print(f"Error processing {result['filename']}: {result['error']}")
            else:
                print(f"Processed {result['filename']}: {result['original_size']:.2f}MB → {result['new_size']:.2f}MB ({result['reduction']:.1f}% reduction)")

    # Calculate overall statistics
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        total_original = sum(r['original_size'] for r in successful_results)
        total_new = sum(r['new_size'] for r in successful_results)
        print(f"\nSummary:")
        print(f"Total files processed: {len(successful_results)}")
        print(f"Total original size: {total_original:.2f}MB")
        print(f"Total new size: {total_new:.2f}MB")
        print(f"Overall reduction: {(1 - (total_new / total_original)) * 100:.1f}%")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images for more efficient API usage")
    parser.add_argument("--input-dir", required=True, help="Directory containing original images")
    parser.add_argument("--output-dir", required=True, help="Directory to save resized images")
    parser.add_argument("--max-size", type=int, default=2000, help="Maximum dimension (width or height) in pixels")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (1-100)")
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir,
        args.max_size,
        args.quality,
        args.grayscale,
        args.workers
    )
