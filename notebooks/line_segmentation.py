import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    import sys
    import os
    import io
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    import base64

    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import functions from file_utils.py
    from src.file_utils import extract_line_coords_from_xml, crop_line_images, parse_coords_points
    
    # Define a wrapper function for debugging purposes
    def debug_extract_line_coords_from_xml(xml_path):
        """
        Wrapper around extract_line_coords_from_xml that adds debugging
        
        Args:
            xml_path: Path to the PAGE XML file
            
        Returns:
            List of dictionaries with line coordinates and metadata
        """
        print(f"Processing XML file: {xml_path}")
        
        # Call the original function
        results = extract_line_coords_from_xml(xml_path)
        
        # Log results
        print(f"Found {len(results)} lines with coordinates")
        
        return results
            
    # Title and Description
    mo.md("""
    # Line Segmentation Visualization Tool
    Upload an image and its XML annotation file to visualize line segmentation.
    """)

    # File Upload UI
    file_section = mo.md("## Upload Files")
    image_path = mo.ui.file(label="Select image file", kind="area", filetypes=[".jpg", ".png", ".tif", ".tiff"])
    xml_path = mo.ui.file(label="Select XML file", kind="area", filetypes=[".xml"])

    # Options
    options_section = mo.md("## Visualization Options")
    show_text = mo.ui.checkbox(label="Show text content", value=True)
    show_ids = mo.ui.checkbox(label="Show line IDs", value=True)
    padding = mo.ui.slider(label="Line padding (px)", start=0, stop=20, value=0, step=1)
    process_button = mo.ui.run_button(label="Process Files")

    # UI Layout
    ui = mo.vstack([
        file_section,
        mo.hstack([image_path]),
        mo.hstack([xml_path]),
        options_section,
        show_text,
        show_ids,
        padding,
        process_button
    ])

    ui
    return (
        Image,
        ImageDraw,
        ImageFont,
        base64,
        crop_line_images,
        debug_extract_line_coords_from_xml,
        extract_line_coords_from_xml,
        file_section,
        image_path,
        io,
        mo,
        options_section,
        os,
        padding,
        parse_coords_points,
        process_button,
        project_root,
        show_ids,
        show_text,
        sys,
        tempfile,
        ui,
        xml_path,
    )


@app.cell
def _(ImageDraw, ImageFont):
    # Line Segmentation Functions
    def draw_bounding_boxes(image, lines, show_text=True, show_ids=True):
        """Draw bounding boxes on the image"""
        # Ensure we have a standard Image object
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
    
        # Make a copy to avoid modifying the original
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image, "RGBA")

        # Use default font
        font = ImageFont.load_default()

        colors = [
            (255, 0, 0, 128),    # Red
            (0, 255, 0, 128),    # Green
            (0, 0, 255, 128),    # Blue
            (255, 255, 0, 128),  # Yellow
            (255, 0, 255, 128),  # Magenta
            (0, 255, 255, 128)   # Cyan
        ]

        for i, line in enumerate(lines):
            color = colors[i % len(colors)]

            # Draw rectangle for the line
            if 'coords' in line and line['coords']:
                x_min, y_min, x_max, y_max = line['coords']
                draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)], 
                    outline=color, 
                    fill=(color[0], color[1], color[2], 50)
                )

                # Draw line ID
                if show_ids:
                    line_num = i + 1
                    reading_order = line.get('reading_order', line_num)
                    label = f"Line {reading_order}"
                    draw.text((x_min, y_min - 15), label, fill=(0, 0, 0), font=font)

                # Draw text
                if show_text and line.get('text'):
                    text = line['text']
                    draw.text((x_min, y_max + 5), text[:40], fill=(0, 0, 0), font=font)

        return annotated_image
    return (draw_bounding_boxes,)


@app.cell
def _(
    Image,
    base64,
    crop_line_images,
    debug_extract_line_coords_from_xml,
    draw_bounding_boxes,
    image_path,
    io,
    mo,
    os,
    padding,
    process_button,
    show_ids,
    show_text,
    tempfile,
    xml_path,
):
    output = None
    # Visualization Logic based on UI values
    if not image_path.value or not xml_path.value:
       output = mo.md("Select files above and click 'Process Files' to visualize line segmentation.")
    elif process_button.value:
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img_file:
                temp_img_file.write(image_path.contents())
                temp_img_path = temp_img_file.name

            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp_xml_file:
                temp_xml_file.write(xml_path.contents())
                temp_xml_path = temp_xml_file.name

            # Process files
            process_info = mo.md(f"Processing files: {image_path.name()} and {xml_path.name()}")

            # Extract line coordinates - using the debug wrapper
            line_data = debug_extract_line_coords_from_xml(temp_xml_path)

            if not line_data:
                # Load the image
                image = Image.open(temp_img_path).convert('RGB')
                no_lines_msg = mo.md("### No lines detected in XML file")
                
                # Convert image to bytes for display
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                img_display = mo.image(img_bytes)
                
                # Combine outputs in a vertical stack
                output = mo.vstack([process_info, no_lines_msg, img_display])
            else:
                # Extract line coordinates
                image = Image.open(temp_img_path).convert('RGB')

                # Add reading order based on index if not present
                for i, line in enumerate(line_data):
                    if 'reading_order' not in line:
                        line['reading_order'] = i

                # Create annotated image
                annotated_image = draw_bounding_boxes(
                    image, 
                    line_data,
                    show_text=show_text.value,
                    show_ids=show_ids.value
                )

                # Extract individual line images
                line_images = crop_line_images(temp_img_path, line_data, padding=padding.value)

                # Display results
                result_title = mo.md(f"### Line Segmentation Results")
                stats = mo.md(f"**Total lines detected:** {len(line_data)}")
                
                # Convert annotated image to bytes for display
                img_byte_arr = io.BytesIO()
                annotated_image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()  
                annotated_img_display = mo.image(img_bytes)

                # Create table of line images with transcriptions
                table_rows = []

                # Limit the number of lines to avoid exceeding output size
                max_lines_to_show = min(10, len(line_images))
                for i, line_info in enumerate(line_images[:max_lines_to_show]):
                    # Convert line image to bytes for display
                    line_img_bytes = io.BytesIO()
                    line_info['image'].convert('RGB').save(line_img_bytes, format='PNG')
                    line_img_data = line_img_bytes.getvalue()
                    
                    # Add to table
                    order = line_info.get('reading_order', i)
                    table_rows.append({
                        "Line": f"Line {order}",
                        "Image": mo.image(line_img_data, width=300),
                        "Text": line_info.get('text', '')
                    })

                lines_title = mo.md("### Line Images and Transcriptions")
                if len(line_images) > max_lines_to_show:
                    lines_title = mo.md(f"### Line Images and Transcriptions (showing first {max_lines_to_show} of {len(line_images)})")
                
                table = mo.ui.table(
                    table_rows,
                    pagination=True,
                    page_size=5
                )
                
                # Combine all outputs in a vertical stack
                output = mo.vstack([
                    process_info,
                    result_title,
                    stats,
                    annotated_img_display,
                    lines_title,
                    table
                ])

            # Clean up temporary files
            os.unlink(temp_img_path)
            os.unlink(temp_xml_path)

        except Exception as e:
           output = mo.md(f"**Error:** {str(e)}")
           print(f"Error details: {e}")
    else:
        output = mo.md("👆 Select files above and click 'Process Files' to visualize line segmentation.")

    output


    return (
        annotated_image if 'annotated_image' in locals() else None,
        annotated_img_display if 'annotated_img_display' in locals() else None,
        i if 'i' in locals() else None,
        image if 'image' in locals() else None,
        img_byte_arr if 'img_byte_arr' in locals() else None,
        img_bytes if 'img_bytes' in locals() else None,
        img_display if 'img_display' in locals() else None,
        line if 'line' in locals() else None,
        line_data if 'line_data' in locals() else None,
        line_images if 'line_images' in locals() else None,
        line_img_bytes if 'line_img_bytes' in locals() else None,
        line_img_data if 'line_img_data' in locals() else None,
        line_info if 'line_info' in locals() else None,
        lines_title if 'lines_title' in locals() else None,
        no_lines_msg if 'no_lines_msg' in locals() else None,
        order if 'order' in locals() else None,
        output,
        process_info if 'process_info' in locals() else None,
        result_title if 'result_title' in locals() else None,
        stats if 'stats' in locals() else None,
        table if 'table' in locals() else None,
        table_rows if 'table_rows' in locals() else None,
        temp_img_path if 'temp_img_path' in locals() else None,
        temp_xml_path if 'temp_xml_path' in locals() else None,
    )


if __name__ == "__main__":
    app.run()
