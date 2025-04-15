import os
import pandas as pd
from typing import Dict, List, Tuple, Any
import xml.etree.ElementTree as ET

from .file_utils import (
    extract_text_from_xml,
    normalize_text,
    find_matching_files,
    extract_id_from_filename
)
from .metrics import (
    calculate_wer,
    calculate_cer,
    calculate_wer_difference,
    evaluate_transcription,
    save_results,
    calculate_aggregate_metrics
)


def evaluate_transkribus(
    ground_truth_dir: str,
    transkribus_dir: str,
    output_dir: str,
    save_transcriptions: bool = True
) -> pd.DataFrame:
    """
    Evaluate Transkribus transcriptions against ground truth.

    Args:
        ground_truth_dir: Directory containing ground truth XML files
        transkribus_dir: Directory containing Transkribus XML files
        output_dir: Directory to save evaluation results
        save_transcriptions: Whether to save the raw transcriptions

    Returns:
        DataFrame with document-level evaluation results
    """
    print(f"Ground truth directory exists: {os.path.exists(ground_truth_dir)}")
    if os.path.exists(ground_truth_dir):
        print(f"Ground truth files: {os.listdir(ground_truth_dir)}")

    print(f"Transkribus directory exists: {os.path.exists(transkribus_dir)}")
    if os.path.exists(transkribus_dir):
        print(f"Transkribus files: {os.listdir(transkribus_dir)}")

    print(f"Finding matching files between ground truth and Transkribus...")
    matched_pairs = find_matching_files(ground_truth_dir, transkribus_dir)
    print(f"Found {len(matched_pairs)} matching documents")

    if not matched_pairs:
        print("No matching documents found. Check your directory paths.")
        return pd.DataFrame()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # Process each document
    for gt_path, transkribus_path in matched_pairs:
        doc_id = extract_id_from_filename(os.path.basename(gt_path))
        print(f"Processing document: {doc_id}")

        # Extract text from both XML files
        gt_lines = extract_text_from_xml(gt_path)
        transkribus_lines = extract_text_from_xml(transkribus_path)

        if not gt_lines:
            print(f"Warning: No ground truth text found for {doc_id}")
            continue

        if not transkribus_lines:
            print(f"Warning: No Transkribus text found for {doc_id}")
            continue

        # Evaluate transcription
        results = evaluate_transcription(gt_lines, transkribus_lines)

        # Add transcription lines to results for saving
        if save_transcriptions:
            results['transcription_lines'] = transkribus_lines

        # Save results
        save_results(results, output_dir, doc_id, method="transkribus")

        # Add to all results
        doc_metrics = results['document_metrics']
        doc_metrics['document_id'] = doc_id
        all_results.append(doc_metrics)

    # Compile all document results into a DataFrame
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(output_dir, "transkribus_all_results.csv"), index=False)

        # Calculate and save aggregate metrics
        calculate_aggregate_metrics(output_dir, "transkribus")

        return all_results_df
    else:
        print("No results were generated.")
        return pd.DataFrame()


def get_transkribus_raw_text(transkribus_dir: str) -> Dict[str, List[str]]:
    """
    Extract raw text from all Transkribus files in a directory.

    Args:
        transkribus_dir: Directory containing Transkribus XML files

    Returns:
        Dictionary mapping document IDs to lists of text lines
    """
    results = {}

    for filename in os.listdir(transkribus_dir):
        if not filename.endswith('.xml'):
            continue

        doc_id = extract_id_from_filename(filename)
        if not doc_id:
            continue

        transkribus_path = os.path.join(transkribus_dir, filename)
        lines = extract_text_from_xml(transkribus_path)

        if lines:
            results[doc_id] = lines

    return results
