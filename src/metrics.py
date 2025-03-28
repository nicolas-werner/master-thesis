import os
import pandas as pd
import numpy as np
from jiwer import wer, cer
from jiwer import Compose, RemovePunctuation, RemoveMultipleSpaces, ToLowerCase
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
import json

from .file_utils import normalize_text


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.

    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (transcription)

    Returns:
        Character Error Rate (0.0 to 1.0)
    """
    # Handle empty strings
    if not reference or not hypothesis:
        return 1.0

    try:
        # Clean both texts similarly to the transform, but manually
        ref_clean = normalize_text(reference)
        hyp_clean = normalize_text(hypothesis)

        # Calculate character error rate
        return cer(ref_clean, hyp_clean)
    except Exception as e:
        print(f"Error calculating CER: {e}")
        return 1.0


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (transcription)

    Returns:
        Word Error Rate (0.0 to 1.0)
    """
    # Handle empty strings
    if not reference or not hypothesis:
        return 1.0

    try:
        # Define transformation pipeline
        transform = Compose([
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            ToLowerCase()
        ])

        # Calculate word error rate
        return wer(reference, hypothesis,
                  truth_transform=transform,
                  hypothesis_transform=transform)
    except ValueError:
        # Fallback: calculate without transforms if there are issues
        try:
            return wer(reference, hypothesis)
        except Exception as e:
            print(f"Error calculating WER: {e}")
            return 1.0



def calculate_bwer(reference, hypothesis):
    """
    Calculate Bag of Words Error Rate which is independent of reading order.

    Formula from the paper: bWER(X,Y) = (1/2N)(|N-|Y|| + ∑|fX(v)-fY(v)|) Vidal et al. 2023

    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (transcription)

    Returns:
        Bag of Words Error Rate (0.0 to 1.0)
    """
    # Handle empty strings
    if not reference or not hypothesis:
        return 1.0

    # Count word frequencies
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    N = len(ref_words)

    if N == 0:
        return 1.0

    ref_counter = Counter(ref_words)
    hyp_counter = Counter(hyp_words)

    # Calculate differences in word frequencies
    all_words = set(ref_counter.keys()).union(hyp_counter.keys())
    word_diff_sum = sum(abs(ref_counter[w] - hyp_counter[w]) for w in all_words)

    # Calculate unavoidable operations (difference in length)
    unavoidable = abs(len(ref_words) - len(hyp_words))

    # Calculate bWER according to the paper's formula
    bwer = (unavoidable + (word_diff_sum - unavoidable)/2) / N

    return bwer

def calculate_wer_difference(reference, hypothesis):
    """
    Calculate the difference between WER and bWER, which correlates with
    reading order errors.

    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (transcription)

    Returns:
        WER-bWER difference (measure of reading order errors)
    """
    wer_value = calculate_wer(reference, hypothesis)
    bwer_value = calculate_bwer(reference, hypothesis)
    return wer_value - bwer_value


def evaluate_transcription(
    gt_lines: List[str],
    transcription_lines: List[str]
) -> Dict[str, Any]:
    """
    Evaluate a transcription against ground truth.

    Args:
        gt_lines: List of ground truth text lines
        transcription_lines: List of transcribed text lines

    Returns:
        Dictionary with evaluation metrics (line-level and document-level)
    """
    # Handle empty inputs
    if not gt_lines or not transcription_lines:
        return {
            'line_metrics': [],
            'document_metrics': {
                'cer': 1.0,
                'wer': 1.0,
                'bwer': 1.0,
                'line_count_match': False,
                'gt_line_count': len(gt_lines),
                'transcription_line_count': len(transcription_lines)
            }
        }

    # Calculate line-level metrics
    line_metrics = []
    max_lines = min(len(gt_lines), len(transcription_lines))

    for i in range(max_lines):
        gt_line = gt_lines[i]
        transcription_line = transcription_lines[i]

        # Calculate metrics for this line
        line_cer = calculate_cer(gt_line, transcription_line)
        line_wer = calculate_wer(gt_line, transcription_line)
        line_bwer = calculate_bwer(gt_line, transcription_line)

        line_metrics.append({
            'line_number': i+1,
            'ground_truth': gt_line,
            'transcription': transcription_line,
            'cer': line_cer,
            'wer': line_wer,
            'bwer': line_bwer
        })

    # Calculate document-level metrics
    gt_full = " ".join(gt_lines)
    transcription_full = " ".join(transcription_lines[:max_lines])

    doc_cer = calculate_cer(gt_full, transcription_full)
    doc_wer = calculate_wer(gt_full, transcription_full)
    doc_bwer = calculate_bwer(gt_full, transcription_full)

    document_metrics = {
        'cer': doc_cer,
        'wer': doc_wer,
        'bwer': doc_bwer,
        'line_count_match': len(gt_lines) == len(transcription_lines),
        'gt_line_count': len(gt_lines),
        'transcription_line_count': len(transcription_lines)
    }

    return {
        'line_metrics': line_metrics,
        'document_metrics': document_metrics
    }


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    doc_id: str,
    method: str
) -> None:
    """
    Save evaluation results to files.

    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save results in
        doc_id: Document ID
        method: Transcription method name
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    # Save line-level results
    line_df = pd.DataFrame(results['line_metrics'])
    line_df['document_id'] = doc_id
    line_df['method'] = method
    line_df.to_csv(os.path.join(method_dir, f"{doc_id}_line_results.csv"), index=False)

    # Save document-level metrics
    doc_metrics = results['document_metrics']
    doc_metrics['document_id'] = doc_id
    doc_metrics['method'] = method

    # Append to document results file if it exists, otherwise create new
    doc_results_path = os.path.join(output_dir, f"{method}_document_results.csv")

    if os.path.exists(doc_results_path):
        doc_df = pd.read_csv(doc_results_path)
        # Remove existing entry for this document if it exists
        doc_df = doc_df[doc_df['document_id'] != doc_id]
        doc_df = pd.concat([doc_df, pd.DataFrame([doc_metrics])], ignore_index=True)
    else:
        doc_df = pd.DataFrame([doc_metrics])

    doc_df.to_csv(doc_results_path, index=False)

    # Save raw transcription lines for reference
    with open(os.path.join(method_dir, f"{doc_id}_transcription.json"), 'w') as f:
        json.dump({
            'document_id': doc_id,
            'method': method,
            'transcription_lines': results.get('transcription_lines', [])
        }, f, indent=2, ensure_ascii=False)


def calculate_aggregate_metrics(results_dir: str, method: str) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across all documents for a method.

    Args:
        results_dir: Directory containing results
        method: Transcription method name

    Returns:
        Dictionary of aggregate metrics
    """
    # Load document results
    doc_results_path = os.path.join(results_dir, f"{method}_document_results.csv")

    if not os.path.exists(doc_results_path):
        return {}

    doc_df = pd.read_csv(doc_results_path)

    # Calculate aggregate metrics
    aggregates = {
        'method': method,
        'mean_cer': doc_df['cer'].mean(),
        'median_cer': doc_df['cer'].median(),
        'std_cer': doc_df['cer'].std(),
        'mean_wer': doc_df['wer'].mean(),
        'median_wer': doc_df['wer'].median(),
        'std_wer': doc_df['wer'].std(),
        'mean_bwer': doc_df['bwer'].mean(),
        'median_bwer': doc_df['bwer'].median(),
        'std_bwer': doc_df['bwer'].std(),
        'line_match_rate': doc_df['line_count_match'].mean(),
        'num_documents': len(doc_df)
    }

    # Save aggregate results
    pd.DataFrame([aggregates]).to_csv(
        os.path.join(results_dir, f"{method}_aggregate_results.csv"),
        index=False
    )

    return aggregates
