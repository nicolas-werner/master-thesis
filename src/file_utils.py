import os
import re
import xml.etree.ElementTree as ET
import unicodedata
from typing import List, Dict, Optional, Tuple, Any
import base64
from PIL import Image
import numpy as np
import io


NAMESPACES = {
    'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
}


def normalize_text(text: str) -> str:
    """
    Normalize text with NFC Unicode normalization. OCR-D Guidelines recommend this.

    Args:
        text: The text to normalize

    Returns:
        Normalized text
    """
    return unicodedata.normalize('NFC', text)


def extract_text_from_xml(xml_path: str) -> List[str]:
    """
    Extract text lines from PAGE XML format, supporting multiple namespace versions.

    Args:
        xml_path: Path to the PAGE XML file

    Returns:
        List of text lines
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Support multiple PAGE format namespaces
        namespaces = {
            # 2013 namespace (used in Reichenau dataset)
            'ns2013': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
            # 2010 namespace (used in Bentham dataset)
            'ns2010': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'
        }

        text_lines = []

        # Try each namespace until we find matching elements
        for prefix, ns in namespaces.items():
            # Find all TextLine elements with this namespace
            for text_line in root.findall(f'.//{{{ns}}}TextLine'):
                # Look for Unicode element within this TextLine
                unicode_elem = text_line.find(f'.//{{{ns}}}Unicode')
                if unicode_elem is not None and unicode_elem.text is not None:
                    line_text = unicode_elem.text.strip()
                    text_lines.append(normalize_text(line_text))

            # If we found text lines with this namespace, no need to try others
            if text_lines:
                break

        # If no text found with namespaces, try without namespace (fallback)
        if not text_lines:
            # Some XML files might not use namespaces consistently
            for text_line in root.findall('.//TextLine'):
                # Try different paths to Unicode element
                unicode_elem = (text_line.find('.//Unicode') or
                               text_line.find('.//TextEquiv/Unicode'))

                if unicode_elem is not None and unicode_elem.text is not None:
                    line_text = unicode_elem.text.strip()
                    text_lines.append(normalize_text(line_text))

        return text_lines
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []


def extract_lines_with_metadata(xml_path: str) -> List[Dict[str, Any]]:
    """
    Extract text lines with metadata from PAGE XML format.

    Args:
        xml_path: Path to the PAGE XML file

    Returns:
        List of dictionaries containing line text and metadata:
        {
            'id': Line ID,
            'text': Normalized text content,
            'coords': Coordinates (if available),
            'index': Index of the line in the document
        }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines_with_metadata = []

        # Extract text lines
        for index, text_line in enumerate(root.findall('.//ns:TextLine', NAMESPACES)):
            line_id = text_line.get('id')

            # Extract text content
            unicode_elem = text_line.find('.//ns:Unicode', NAMESPACES)
            text = ""
            if unicode_elem is not None and unicode_elem.text is not None:
                text = unicode_elem.text.strip()
                text = normalize_text(text)

            coords = None
            coords_elem = text_line.find('.//ns:Coords', NAMESPACES)
            if coords_elem is not None:
                points = coords_elem.get('points')
                if points:
                    coords = points

            # Add line data to results
            lines_with_metadata.append({
                'id': line_id,
                'text': text,
                'coords': coords,
                'index': index
            })

        return lines_with_metadata
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []

def extract_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract the document ID from filenames with various patterns.

    Patterns supported:
    - 7474184.xml (plain numeric ID)
    - 072_066_002.xml (compound ID with underscores)
    - 0001_7474184.xml (prefix + numeric ID)
    - 0001_072_066_002.xml (prefix + compound ID)
    - transkribus_7474184.xml (tool prefix + ID)

    Args:
        filename: Filename to extract ID from

    Returns:
        Document ID or None if no match
    """
    # Remove file extension
    base_name = os.path.splitext(filename)[0]

    # Case 1: Simple ground truth files (7474184.xml)
    if re.match(r'^\d+$', base_name):
        return base_name

    # Case 2: Compound ground truth files (072_066_002.xml)
    if re.match(r'^\d+(?:_\d+)+$', base_name):
        return base_name

    # Case 3: Files with numeric prefix (0001_7474184.xml or 0001_072_066_002.xml)
    match = re.match(r'^\d{4}_(.+)$', base_name)
    if match:
        return match.group(1)

    # Case 4: Files with tool name prefix (transkribus_7474184.xml, gemini_7474184.xml)
    match = re.match(r'^[a-zA-Z]+_(.+)$', base_name)
    if match:
        return match.group(1)

    return None


def find_matching_files(ground_truth_dir: str, other_dir: str) -> List[Tuple[str, str]]:
    """
    Match ground truth files with files in another directory.

    Args:
        ground_truth_dir: Directory containing ground truth XML files
        other_dir: Directory to find matching files in (e.g., transkribus)

    Returns:
        List of tuples (ground_truth_path, matching_path)
    """
    matched_pairs = []

    # Get all ground truth files and their IDs
    gt_files = {}
    for f in os.listdir(ground_truth_dir):
        if f.endswith('.xml'):
            # For ground truth, just use the filename without extension
            doc_id = os.path.splitext(f)[0]
            gt_files[doc_id] = os.path.join(ground_truth_dir, f)
            print(f"Ground truth file: {f} → ID: {doc_id}")  # Debug print

    # Find matching files in the other directory
    for f in os.listdir(other_dir):
        if f.endswith('.xml'):
            # For transkribus files, extract the numeric part after prefix
            match = re.match(r'^\d{4}_(\d+)\.xml$', f)
            if match:
                doc_id = match.group(1)
                print(f"Transkribus file: {f} → ID: {doc_id}")  # Debug print
                if doc_id in gt_files:
                    matched_pairs.append((
                        gt_files[doc_id],
                        os.path.join(other_dir, f)
                    ))
                    print(f"Matched: {os.path.basename(gt_files[doc_id])} with {f}")  # Debug print

    return matched_pairs

def find_file_for_id(doc_id: str, directory: str, extensions: List[str] = ['.xml']) -> Optional[str]:
    """
    Find a file for a document ID in a directory.

    Args:
        doc_id: Document ID to search for
        directory: Directory to search in
        extensions: List of file extensions to look for

    Returns:
        Path to matching file or None if not found
    """
    # Try direct match first (e.g., 7474184.xml)
    for ext in extensions:
        direct_path = os.path.join(directory, f"{doc_id}{ext}")
        if os.path.exists(direct_path):
            return direct_path

    # Try pattern matches (e.g., 0001_7474184.xml or transkribus_7474184.xml)
    for f in os.listdir(directory):
        # Only consider files with matching extensions
        if not any(f.endswith(ext) for ext in extensions):
            continue

        # Check if this file contains the document ID
        file_id = extract_id_from_filename(f)
        if file_id == doc_id:
            return os.path.join(directory, f)

    # If nothing found, look for any file containing the ID
    for f in os.listdir(directory):
        if any(f.endswith(ext) for ext in extensions) and doc_id in f:
            return os.path.join(directory, f)

    return None


def get_document_paths(doc_id: str, dataset_dir: str) -> Dict[str, Optional[str]]:
    """
    Get all paths related to a document.

    Args:
        doc_id: Document ID
        dataset_dir: Base directory for the dataset

    Returns:
        Dictionary with paths to ground_truth, transkribus, and image files
    """
    paths = {
        'ground_truth': find_file_for_id(
            doc_id,
            os.path.join(dataset_dir, 'ground_truth'),
            ['.xml']
        ),
        'transkribus': find_file_for_id(
            doc_id,
            os.path.join(dataset_dir, 'transkribus'),
            ['.xml']
        ),
        'image': find_file_for_id(
            doc_id,
            os.path.join(dataset_dir, 'images'),
            ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        )
    }

    return paths


def list_available_documents(dataset_dir: str) -> List[str]:
    """
    Get a list of all available document IDs in a dataset.

    Args:
        dataset_dir: Base directory for the dataset

    Returns:
        List of document IDs
    """
    ground_truth_dir = os.path.join(dataset_dir, 'ground_truth')
    doc_ids = []

    for f in os.listdir(ground_truth_dir):
        if f.endswith('.xml'):
            doc_id = extract_id_from_filename(f)
            if doc_id:
                doc_ids.append(doc_id)

    return doc_ids



def encode_image(image_path: str) -> str:
    """
    Encode an image to base64.

    Args:
        image_path: Path to the image

    Returns:
        Base64-encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_prompt_from_file(prompt_path: str) -> str:
    """
    Load a prompt from a text file.

    Args:
        prompt_path: Full path to the prompt file

    Returns:
        The prompt as a string
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error loading prompt from {prompt_path}: {e}")
        raise e

def extract_line_coords_from_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    Extract line coordinates and text from PAGE XML format.

    Args:
        xml_path: Path to the PAGE XML file

    Returns:
        List of dictionaries containing line info:
        {
            'id': Line ID,
            'text': Normalized text content,
            'coords': Bounding box coordinates as (x_min, y_min, x_max, y_max),
            'index': Index of the line in the document
        }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Support multiple PAGE format namespaces
        namespaces = {
            # 2013 namespace (used in Reichenau dataset)
            'ns2013': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
            # 2010 namespace (used in Bentham dataset)
            'ns2010': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'
        }

        lines_with_coords = []

        # Try each namespace until we find matching elements
        for prefix, ns in namespaces.items():
            found_lines = False

            # Find all TextLine elements with this namespace
            for index, text_line in enumerate(root.findall(f'.//{{{ns}}}TextLine')):
                line_id = text_line.get('id')

                # Extract text content
                unicode_elem = text_line.find(f'.//{{{ns}}}Unicode')
                text = ""
                if unicode_elem is not None and unicode_elem.text is not None:
                    text = unicode_elem.text.strip()
                    text = normalize_text(text)

                # Extract coordinates
                coords_elem = text_line.find(f'.//{{{ns}}}Coords')
                if coords_elem is not None:
                    points_str = coords_elem.get('points')
                    if points_str:
                        # Parse points string to get bounding box
                        coords = parse_coords_points(points_str)

                        lines_with_coords.append({
                            'id': line_id,
                            'text': text,
                            'coords': coords,
                            'index': index
                        })
                        found_lines = True

            if found_lines:
                break

        # If no lines found with namespaces, try without namespace
        if not lines_with_coords:
            for index, text_line in enumerate(root.findall('.//TextLine')):
                line_id = text_line.get('id', f"line_{index}")

                # Try different paths to Unicode element
                unicode_elem = (text_line.find('.//Unicode') or
                               text_line.find('.//TextEquiv/Unicode'))

                text = ""
                if unicode_elem is not None and unicode_elem.text is not None:
                    text = unicode_elem.text.strip()
                    text = normalize_text(text)

                # Extract coordinates
                coords_elem = text_line.find('.//Coords')
                if coords_elem is not None:
                    points_str = coords_elem.get('points')
                    if points_str:
                        coords = parse_coords_points(points_str)

                        lines_with_coords.append({
                            'id': line_id,
                            'text': text,
                            'coords': coords,
                            'index': index
                        })

        return lines_with_coords

    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []


def parse_coords_points(points_str: str) -> Tuple[int, int, int, int]:
    """
    Parse coordinate points string from PAGE XML format.

    Args:
        points_str: String with coordinates in format 'x1,y1 x2,y2 x3,y3...'

    Returns:
        Tuple with bounding box (x_min, y_min, x_max, y_max)
    """
    try:
        # Split the points string and convert to integer pairs
        point_pairs = [p.split(',') for p in points_str.split()]
        points = [(int(p[0]), int(p[1])) for p in point_pairs]

        # Calculate bounding box
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]

        # Return bounding box as (x_min, y_min, x_max, y_max)
        return (min(x_values), min(y_values), max(x_values), max(y_values))

    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        # Return a default empty bounding box
        return (0, 0, 0, 0)


def crop_line_images(image_path: str, line_coords: List[Dict[str, Any]], padding: int = 5) -> List[Dict[str, Any]]:
    """
    Crop line images from a full page image based on line coordinates.

    Args:
        image_path: Path to the full page image
        line_coords: List of dictionaries with line coordinates and metadata
        padding: Padding to add around each line (in pixels)

    Returns:
        List of dictionaries with line images and metadata:
        {
            'id': Line ID,
            'text': Line text,
            'image': PIL Image object of the cropped line,
            'coords': Original coordinates,
            'index': Original index
        }
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            line_images = []

            # Get image dimensions
            img_width, img_height = img.size

            # Process each line
            for line in line_coords:
                if 'coords' not in line or not line['coords']:
                    continue

                x_min, y_min, x_max, y_max = line['coords']

                # Apply padding, but stay within image bounds
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(img_width, x_max + padding)
                y_max = min(img_height, y_max + padding)

                # Crop the line image
                line_img = img.crop((x_min, y_min, x_max, y_max))

                # Create result with the line image and metadata
                line_result = line.copy()
                line_result['image'] = line_img

                line_images.append(line_result)

            return line_images

    except Exception as e:
        print(f"Error cropping line images from {image_path}: {e}")
        return []


def optimize_line_image(line_img, max_width=1000, max_height=200, quality=85) -> Image.Image:
    """
    Optimize a line image for API processing.

    Args:
        line_img: PIL Image object of a text line
        max_width: Maximum width for resizing
        max_height: Maximum height for resizing
        quality: JPEG quality for compression

    Returns:
        Optimized PIL Image object
    """
    try:
        # Get current size
        width, height = line_img.size

        # Check if resizing is needed
        if width > max_width or height > max_height:
            # Calculate new dimensions while preserving aspect ratio
            if width > max_width:
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:
                new_height = max_height
                new_width = int(width * (max_height / height))

            # Resize the image
            line_img = line_img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to JPEG in memory with specified quality
        img_byte_arr = io.BytesIO()
        line_img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)

        # Return the optimized image
        img_byte_arr.seek(0)
        return Image.open(img_byte_arr)

    except Exception as e:
        print(f"Error optimizing line image: {e}")
        return line_img


def process_page_by_lines(image_path: str, xml_path: str, optimize: bool = True) -> Dict[str, Any]:
    """
    Process a document page by extracting individual line images and text.

    Args:
        image_path: Path to the page image
        xml_path: Path to the PAGE XML file
        optimize: Whether to optimize line images

    Returns:
        Dictionary with page information:
        {
            'doc_id': Document ID,
            'lines': List of line dictionaries (with images and text),
            'gt_text': Full ground truth text
        }
    """
    # Extract document ID from filename
    doc_id = os.path.splitext(os.path.basename(image_path))[0]

    # Extract line coordinates and text
    line_coords = extract_line_coords_from_xml(xml_path)

    # No lines found
    if not line_coords:
        print(f"No lines found in {xml_path}")
        return {'doc_id': doc_id, 'lines': [], 'gt_text': ''}

    # Crop line images
    line_images = crop_line_images(image_path, line_coords)

    # Optimize line images if requested
    if optimize:
        for line in line_images:
            line['image'] = optimize_line_image(line['image'])

    # Extract full ground truth text
    gt_text = '\n'.join([line['text'] for line in line_coords])

    return {
        'doc_id': doc_id,
        'lines': line_images,
        'gt_text': gt_text
    }


def encode_image_object(img: Image.Image) -> str:
    """
    Encode a PIL Image object to base64.

    Args:
        img: PIL Image object

    Returns:
        Base64-encoded image
    """
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode('utf-8')
