import os
import re
import xml.etree.ElementTree as ET
import unicodedata
from typing import List, Dict, Optional, Tuple, Any
import base64


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
    Extract text lines from PAGE XML format.

    Args:
        xml_path: Path to the PAGE XML file

    Returns:
        List of text lines
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        text_lines = []
        for text_line in root.findall('.//ns:TextLine', NAMESPACES):
            unicode_elem = text_line.find('.//ns:Unicode', NAMESPACES)
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
