import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Line Segmentation for [Reichenauer Inkunabeln](https://zenodo.org/records/11046062)
        The dataset is not line segmented by default. However the Annotation Files provide Coordinates for each line on a page.
        This notebook visualizes the line bounding boxes from the XML annotation files.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Example Process""")
    return


@app.cell(hide_code=True)
def _():
    import os
    from PIL import Image, ImageDraw, ImageFont
    import xml.etree.ElementTree as ET
    import re
    import numpy as np

    def load_image_and_xml(image_path, xml_path):
        """Load image and XML annotation file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        image = Image.open(image_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        return image, root

    def extract_line_coordinates(root):
        """Extract line coordinates from XML file"""
        # Find all TextLine elements considering the namespace
        lines = []

        # Define the namespace
        ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        # Find all TextLine elements using the namespace
        text_lines = root.findall('.//page:TextLine', namespaces=ns)

        print(f"Found {len(text_lines)} text lines in the XML")

        for line in text_lines:
            coords_elem = line.find('./page:Coords', namespaces=ns)

            if coords_elem is not None:
                points_str = coords_elem.get('points')

                # Parse points string into list of (x, y) tuples
                # Format is typically "x1,y1 x2,y2 x3,y3 ..."
                points = []
                for point in points_str.split():
                    x, y = map(int, point.split(','))
                    points.append((x, y))

                # Get line ID and custom information 
                line_id = line.get('id', '')
                custom = line.get('custom', '')

                # Try to extract reading order if available
                reading_order = None
                if custom:
                    match = re.search(r'readingOrder\s*\{index:(\d+);\}', custom)
                    if match:
                        reading_order = int(match.group(1))

                # Get text content if available
                text_equiv = line.find('.//page:TextEquiv/page:Unicode', namespaces=ns)
                text = text_equiv.text if text_equiv is not None else ""

                lines.append({
                    'id': line_id,
                    'points': points,
                    'reading_order': reading_order,
                    'text': text
                })

        # Sort lines by reading order if available
        if all(line['reading_order'] is not None for line in lines):
            lines.sort(key=lambda x: x['reading_order'])

        return lines

    def draw_bounding_boxes(image, lines, show_text=True, show_ids=False):
        """Draw bounding boxes on the image"""
        # Make a copy to avoid modifying the original
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image, "RGBA")

        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            # Fallback to default font
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
            # Draw polygon for the line
            color = colors[i % len(colors)]
            draw.polygon(line['points'], outline=color, fill=(color[0], color[1], color[2], 50))

            # Calculate bounding box
            xs = [p[0] for p in line['points']]
            ys = [p[1] for p in line['points']]
            x_min, y_min = min(xs), min(ys)

            # Draw line ID or reading order
            if show_ids:
                label = f"Line {i+1}"
                if line['reading_order'] is not None:
                    label += f" (Order: {line['reading_order']})"
                draw.text((x_min, y_min - 20), label, fill=(0, 0, 0), font=font)

            # Draw text
            if show_text and line['text']:
                # Create a text background
                text_width, text_height = draw.textbbox((0, 0), line['text'], font=font)[2:]
                text_x = x_min
                text_y = y_min - 20 - text_height if show_ids else y_min - text_height
                draw.rectangle(
                    [(text_x, text_y), (text_x + text_width, text_y + text_height)],
                    fill=(255, 255, 255, 200)
                )
                draw.text((text_x, text_y), line['text'], fill=(0, 0, 0), font=font)

        return annotated_image

    def extract_line_images(image, lines):
        """Extract individual line images based on their polygon coordinates"""
        line_images = []

        for line in lines:
            # Get bounding box coordinates from the polygon points
            xs = [p[0] for p in line['points']]
            ys = [p[1] for p in line['points']]
            x_min, y_min = max(0, min(xs)), max(0, min(ys))
            x_max, y_max = min(image.width, max(xs)), min(image.height, max(ys))

            # Instead of adding margin, use exact polygon boundaries
            # This will provide a precise crop matching the colored boxes

            # Create a mask for this specific line to avoid overlap
            mask = Image.new('L', (image.width, image.height), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(line['points'], fill=255)

            # Crop the relevant region
            line_bbox = image.crop((x_min, y_min, x_max, y_max))

            # Also crop the mask to the same region
            mask_bbox = mask.crop((x_min, y_min, x_max, y_max))

            # Create a new RGBA image with transparent background
            line_image = Image.new('RGBA', (x_max - x_min, y_max - y_min), (255, 255, 255, 0))

            # Paste only the masked region
            line_image.paste(line_bbox, (0, 0), mask_bbox)

            # Store the line image along with its data
            line_info = {
                'image': line_image,
                'text': line['text'],
                'id': line['id'],
                'reading_order': line['reading_order'],
                'bbox': (x_min, y_min, x_max, y_max),
                'points': [(x - x_min, y - y_min) for x, y in line['points']]  # Adjust points to new coordinate system
            }

            line_images.append(line_info)

        return line_images

    # def save_line_images(line_images, output_dir, image_basename):
    #     """Save extracted line images to files with their transcriptions"""
    #     # Create output directory if it doesn't exist
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Create a transcription file
    #     transcription_path = os.path.join(output_dir, f"{image_basename}_transcription.txt")

    #     with open(transcription_path, 'w', encoding='utf-8') as f:
    #         for i, line_info in enumerate(line_images):
    #             # Generate filename for this line
    #             order = line_info['reading_order'] if line_info['reading_order'] is not None else i
    #             line_filename = f"{image_basename}_line_{order:02d}.png"
    #             line_path = os.path.join(output_dir, line_filename)

    #             # Save the line image
    #             line_info['image'].save(line_path)

    #             # Write the transcription
    #             f.write(f"{line_filename}\t{line_info['text']}\n")

    #     return transcription_path

    def visualize_line_segmentation(image_path, xml_path, show_text=True, show_ids=True):
        """Visualize line segmentation for a given image and XML file"""
        # Load image and XML
        image, root = load_image_and_xml(image_path, xml_path)

        # Extract line coordinates
        lines = extract_line_coordinates(root)

        # Draw bounding boxes
        annotated_image = draw_bounding_boxes(image, lines, show_text, show_ids)

        # Extract individual line images
        line_images = extract_line_images(image, lines)

        return annotated_image, lines, line_images, image
    return (
        ET,
        Image,
        ImageDraw,
        ImageFont,
        draw_bounding_boxes,
        extract_line_coordinates,
        extract_line_images,
        load_image_and_xml,
        np,
        os,
        re,
        visualize_line_segmentation,
    )


@app.cell(hide_code=True)
def _(mo):
    # Create UI elements using vstack for better organization

    # File upload section
    file_section = mo.md("## Input Files")
    image_path = mo.ui.file(label="Select image file", kind="area", filetypes=[".jpg"])
    xml_path = mo.ui.file(label="Select XML file", kind="area", filetypes=[".xml"])

    file_inputs = mo.vstack([
        file_section,
        mo.hstack([
            image_path
        ]),
        mo.hstack([
            xml_path
        ])
    ])

    # Advanced options section
    options_section = mo.md("## Visualization Options")
    show_text = mo.ui.checkbox(label="Show text content", value=False)
    show_ids = mo.ui.checkbox(label="Show line IDs", value=False)

    options_inputs = mo.vstack([
        options_section,
        show_text,
        show_ids
    ])

    # Process button
    process_button = mo.ui.run_button(label="Process Files")

    # Combine all UI elements in a vstack
    ui_elements = mo.vstack([
        file_inputs,
        options_inputs,
        process_button
    ])
    return (
        file_inputs,
        file_section,
        image_path,
        options_inputs,
        options_section,
        process_button,
        show_ids,
        show_text,
        ui_elements,
        xml_path,
    )


@app.cell
def _(mo, ui_elements):
    mo.callout(ui_elements)
    return


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell(hide_code=True)
def _(
    image_path,
    mo,
    os,
    process_button,
    show_ids,
    show_text,
    visualize_line_segmentation,
    xml_path,
):
    # Initialize a cell-local variable to track if we have results
    _has_results = False
    _output = None

    if process_button.value:  # Button has been clicked
        if not image_path.value or not xml_path.value:
            _output = mo.md("⚠️ Please select both an image file and its corresponding XML annotation file.")
        else:
            try:
                import tempfile
                import io

                # Create temporary files for processing
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img_file:
                    temp_img_file.write(image_path.contents())
                    temp_img_path = temp_img_file.name

                with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp_xml_file:
                    temp_xml_file.write(xml_path.contents())
                    temp_xml_path = temp_xml_file.name

                # Show file info
                print(f"Processing files: {image_path.name()} and {xml_path.name()}")

                # Process the files using the temporary file paths
                annotated_image, lines, line_images, original_image = visualize_line_segmentation(
                    temp_img_path, 
                    temp_xml_path, 
                    show_text=show_text.value,
                    show_ids=show_ids.value
                )

                # Clean up temporary files
                os.unlink(temp_img_path)
                os.unlink(temp_xml_path)

                # Indicate we have results
                _has_results = True

                # Convert annotated image to bytes for display
                img_byte_arr = io.BytesIO()
                annotated_image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                # Create initial output with overview
                _output = mo.vstack([
                    mo.md(f"### Line Segmentation Results"),
                    mo.md(f"**Total lines detected:** {len(lines)}"),
                    mo.image(img_bytes)
                ])

                # Create table of line images with their transcriptions
                table_rows = []

                for i, line_info in enumerate(line_images):
                    # Convert line image to bytes for display
                    line_img_bytes = io.BytesIO()
                    line_info['image'].save(line_img_bytes, format='PNG')

                    # Add to table
                    order = line_info['reading_order'] if line_info['reading_order'] is not None else i
                    line_num = f"Line {order}"
                    line_image_elem = mo.image(line_img_bytes.getvalue(), width=300)

                    table_rows.append({
                        "Line Number": line_num,
                        "Image": line_image_elem,
                        "Text": line_info['text']
                    })

                # Display table of line images and transcriptions
                line_table = mo.ui.table(
                    table_rows,
                    pagination=True,
                    page_size=10,
                )

                _output = mo.vstack([
                    _output,
                    mo.md("### Line Images and Transcriptions"),
                    line_table
                ])

                print("Processing completed successfully!")
            except Exception as e:
                _output = mo.md(f"**Error:** {str(e)}")
                print(f"Error occurred: {str(e)}")

    # If no results and no error message, show initial message
    if _output is None:
        _output = mo.md("👆 Select files above and click 'Process Files' to visualize line segmentation.")

    _output
    return (
        annotated_image,
        i,
        img_byte_arr,
        img_bytes,
        io,
        line_image_elem,
        line_images,
        line_img_bytes,
        line_info,
        line_num,
        line_table,
        lines,
        order,
        original_image,
        table_rows,
        temp_img_file,
        temp_img_path,
        temp_xml_file,
        temp_xml_path,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Batch Line Segmentation""")
    return


@app.cell(hide_code=True)
def _(mo):
    import glob

    # Path input fields
    input_path = mo.ui.text(
        label="Input folder path", 
        placeholder="Path to folder with JPG files and page/ subfolder",
        value="./train_set"
    )

    output_path = mo.ui.text(
        label="Output folder path", 
        placeholder="Path to save line images",
        value="./output"
    )

    # Process button
    batch_process_button = mo.ui.run_button(label="Process All Files")

    # Display the UI elements
    batch_ui = mo.vstack([
        mo.md("Enter the path to a folder containing JPG files with corresponding XML files in 'page' subfolder:"),
        input_path,
        mo.md("Enter the output folder for line segments:"),
        output_path,
        batch_process_button
    ])

    mo.callout(batch_ui)
    return batch_process_button, batch_ui, glob, input_path, output_path


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell(hide_code=True)
def _(ET, Image, ImageDraw, glob, mo, os, re):
    # Functions


    def q_find_matching_files(input_dir):
        """Find matching JPG and XML files in the given directory structure."""
        # Validate input directory
        if not os.path.isdir(input_dir):
            return [], f"Error: {input_dir} is not a valid directory"

        # Find all JPG files in the main directory
        jpg_files = glob.glob(os.path.join(input_dir, "*.jpg"))

        # Check for page subfolder
        page_dir = os.path.join(input_dir, "page")
        if not os.path.isdir(page_dir):
            return [], f"Error: 'page' subfolder not found in {input_dir}"

        # Find matching file pairs
        matched_pairs = []

        for jpg_path in jpg_files:
            # Get the base filename without extension
            basename = os.path.splitext(os.path.basename(jpg_path))[0]

            # Look for matching XML file in page subfolder
            xml_path = os.path.join(page_dir, f"{basename}.xml")

            if os.path.exists(xml_path):
                matched_pairs.append((jpg_path, xml_path))

        return matched_pairs, f"Found {len(matched_pairs)} matching file pairs"


    def q_load_image_and_xml(image_path, xml_path):
        """Load image and XML annotation file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        image = Image.open(image_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        return image, root

    def q_extract_line_coordinates(root):
        """Extract line coordinates from XML file"""
        # Find all TextLine elements considering the namespace
        lines = []

        # Define the namespace
        ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        # Find all TextLine elements using the namespace
        text_lines = root.findall('.//page:TextLine', namespaces=ns)

        for line in text_lines:
            coords_elem = line.find('./page:Coords', namespaces=ns)

            if coords_elem is not None:
                points_str = coords_elem.get('points')

                # Parse points string into list of (x, y) tuples
                # Format is typically "x1,y1 x2,y2 x3,y3 ..."
                points = []
                for point in points_str.split():
                    x, y = map(int, point.split(','))
                    points.append((x, y))

                # Get line ID and custom information 
                line_id = line.get('id', '')
                custom = line.get('custom', '')

                # Try to extract reading order if available
                reading_order = None
                if custom:
                    match = re.search(r'readingOrder\s*\{index:(\d+);\}', custom)
                    if match:
                        reading_order = int(match.group(1))

                # Get text content if available
                text_equiv = line.find('.//page:TextEquiv/page:Unicode', namespaces=ns)
                text = text_equiv.text if text_equiv is not None else ""

                lines.append({
                    'id': line_id,
                    'points': points,
                    'reading_order': reading_order,
                    'text': text
                })

        # Sort lines by reading order if available
        if all(line['reading_order'] is not None for line in lines):
            lines.sort(key=lambda x: x['reading_order'])

        return lines

    def q_extract_line_images(image, lines):
        """Extract individual line images based on their polygon coordinates"""
        line_images = []

        for line in lines:
            # Get bounding box coordinates from the polygon points
            xs = [p[0] for p in line['points']]
            ys = [p[1] for p in line['points']]
            x_min, y_min = max(0, min(xs)), max(0, min(ys))
            x_max, y_max = min(image.width, max(xs)), min(image.height, max(ys))

            # Create a mask for this specific line to avoid overlap
            mask = Image.new('L', (image.width, image.height), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(line['points'], fill=255)

            # Crop the relevant region
            line_bbox = image.crop((x_min, y_min, x_max, y_max))

            # Also crop the mask to the same region
            mask_bbox = mask.crop((x_min, y_min, x_max, y_max))

            # Create a new RGB image with white background
            line_image = Image.new('RGB', (x_max - x_min, y_max - y_min), (255, 255, 255))

            # Paste only the masked region
            line_image.paste(line_bbox, (0, 0), mask_bbox)

            # Store the line image along with its data
            line_info = {
                'image': line_image,
                'text': line['text'],
                'id': line['id'],
                'reading_order': line['reading_order'],
                'bbox': (x_min, y_min, x_max, y_max)
            }

            line_images.append(line_info)

        return line_images

    def q_save_line_images(line_images, output_dir, image_basename):
        """Save extracted line images to files with their individual transcriptions"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Keep track of saved files for reporting
        saved_files = []

        # Process each line
        for i, line_info in enumerate(line_images):
            # Generate filename for this line
            order = line_info['reading_order'] if line_info['reading_order'] is not None else i
            line_basename = f"{image_basename}_line_{order:02d}"

            # Image file path
            line_img_path = os.path.join(output_dir, f"{line_basename}.png")

            # Transcription file path (same basename but with .txt extension)
            line_txt_path = os.path.join(output_dir, f"{line_basename}.txt")

            # Save the line image
            line_info['image'].save(line_img_path)

            # Save individual transcription file for this line
            with open(line_txt_path, 'w', encoding='utf-8') as f:
                f.write(line_info['text'])

            saved_files.append((line_img_path, line_txt_path))

        # Also create a master transcription file for reference
        master_transcription_path = os.path.join(output_dir, f"{image_basename}_transcription.txt")
        with open(master_transcription_path, 'w', encoding='utf-8') as f:
            for i, line_info in enumerate(line_images):
                order = line_info['reading_order'] if line_info['reading_order'] is not None else i
                line_filename = f"{image_basename}_line_{order:02d}.png"
                f.write(f"{line_filename}\t{line_info['text']}\n")

        return saved_files


    def process_all_files(input_dir, output_dir):
        """Process all matching file pairs and extract line segments"""
        # Find matching files
        matched_pairs, message = q_find_matching_files(input_dir)

        if not matched_pairs:
            return mo.md(f"⚠️ {message}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Track progress
        results = []
        progress_status = mo.status.progress_bar(matched_pairs)

        for jpg_path, xml_path in progress_status:
            try:
                # Get base filename
                basename = os.path.splitext(os.path.basename(jpg_path))[0]

                # Create document folder in output directory
                doc_output_dir = os.path.join(output_dir, basename)
                os.makedirs(doc_output_dir, exist_ok=True)

                # Process the files
                image, root = q_load_image_and_xml(jpg_path, xml_path)
                lines = q_extract_line_coordinates(root)
                line_images = q_extract_line_images(image, lines)

                # Save line images and transcription
                saved_files = q_save_line_images(line_images, doc_output_dir, basename)

                # Create result entry
                results.append({
                    "filename": basename,
                    "lines_found": len(lines),
                    "status": "✅ Success",
                    "output_path": doc_output_dir
                })

            except Exception as e:
                results.append({
                    "filename": basename if 'basename' in locals() else os.path.basename(jpg_path),
                    "lines_found": 0,
                    "status": f"❌ Error: {str(e)}",
                    "output_path": ""
                })

        # Create a results table
        results_table = mo.ui.table(
            results,
            pagination=True,
            page_size=10
        )

        # Create a summary
        successful = sum(1 for r in results if "✅" in r["status"])
        failed = len(results) - successful
        total_lines = sum(r["lines_found"] for r in results)

        summary = mo.md(f"""
        ## Batch Processing Summary

        - **Total documents processed:** {len(results)}
        - **Successfully processed:** {successful}
        - **Failed:** {failed}
        - **Total lines extracted:** {total_lines}
        - **Output directory:** `{output_dir}`
        """)

        return mo.vstack([
            summary,
            mo.md("### Processing Results"),
            results_table
        ])
    return (
        process_all_files,
        q_extract_line_coordinates,
        q_extract_line_images,
        q_find_matching_files,
        q_load_image_and_xml,
        q_save_line_images,
    )


@app.cell(hide_code=True)
def _(batch_process_button, input_path, mo, output_path, process_all_files):
    batch_result = None

    if batch_process_button.value:
        # Validate input and output paths
        if not input_path.value:
            batch_result = mo.md("⚠️ Please enter an input folder path")
        elif not output_path.value:
            batch_result = mo.md("⚠️ Please enter an output folder path")
        else:
            # Process all files
            try:
                input_dir = input_path.value.strip()
                output_dir = output_path.value.strip()
                batch_result = process_all_files(input_dir, output_dir)
            except Exception as e:
                batch_result = mo.md(f"**Error occurred during processing**: {str(e)}")

    # Display initial message or results
    if batch_result is None:
        batch_result = mo.md("👆 Enter the input and output folders, then click 'Process All Files'")

    batch_result
    return batch_result, input_dir, output_dir


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
