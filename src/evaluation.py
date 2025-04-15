import os
import time
import pandas as pd
import json
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import random
from difflib import SequenceMatcher

from .models.openai_compatible import OpenAICompatibleModel
from .file_utils import encode_image, extract_text_from_xml, find_file_for_id, process_page_by_lines, encode_image_object
from .metrics import evaluate_transcription, save_results, calculate_aggregate_metrics


# Define Pydantic model for structured output
class Transcription(BaseModel):
    correct_transcription: str = Field(
        "", 
        description="The exact transcription of the text as it appears, preserving historical spellings, abbreviations, and special characters. Return an empty string if the image contains no text or shows non-textual elements."
    )


def align_lines(gt_lines, pred_lines):
    """
    Align prediction lines with ground truth lines using a dynamic programming approach
    to find the best match based on string similarity.
    
    Args:
        gt_lines: List of ground truth lines
        pred_lines: List of prediction lines
        
    Returns:
        aligned_pred_lines: List of aligned prediction lines (may contain empty strings)
    """
    # If either list is empty, return empty list with same length as ground truth
    if not gt_lines or not pred_lines:
        return [""] * len(gt_lines)
    
    # Calculate similarity scores between each pair of lines
    similarity_matrix = []
    for gt_line in gt_lines:
        line_scores = []
        for pred_line in pred_lines:
            similarity = SequenceMatcher(None, gt_line, pred_line).ratio()
            line_scores.append(similarity)
        similarity_matrix.append(line_scores)
    
    # Initialize aligned predictions with empty strings
    aligned_pred_lines = [""] * len(gt_lines)
    
    # Track which prediction lines have been used
    used_pred_lines = set()
    
    # Assign prediction lines to ground truth lines based on similarity
    # Start with highest similarity matches
    matches = []
    for gt_idx, scores in enumerate(similarity_matrix):
        for pred_idx, score in enumerate(scores):
            matches.append((score, gt_idx, pred_idx))
    
    # Sort by similarity score (highest first)
    matches.sort(reverse=True)
    
    # Assign predictions to ground truth lines
    for score, gt_idx, pred_idx in matches:
        # Use a threshold to avoid very poor matches
        if score < 0.3:  # Skip very poor matches
            continue
            
        # Skip if this prediction line has already been used
        if pred_idx in used_pred_lines:
            continue
            
        # Assign prediction line to ground truth line
        aligned_pred_lines[gt_idx] = pred_lines[pred_idx]
        used_pred_lines.add(pred_idx)
        
        # If all prediction lines have been assigned, break
        if len(used_pred_lines) == len(pred_lines):
            break
    
    # For debugging purposes
    # for i, (gt, pred) in enumerate(zip(gt_lines, aligned_pred_lines)):
    #     print(f"Line {i+1}:")
    #     print(f"GT: {gt}")
    #     print(f"PRED: {pred}")
    #     print()
        
    return aligned_pred_lines


def process_document(
    provider: str,
    model_name: str,
    doc_id: str,
    gt_path: str,
    image_path: str,
    output_dir: str,
    messages: List[Dict],
    use_structured_output: bool = False
):
    """
    Process a single document with the specified model for page-wise OCR evaluation.

    Args:
        provider: Model provider name
        model_name: Model name
        doc_id: Document ID
        gt_path: Path to ground truth file
        image_path: Path to the image file
        output_dir: Output directory for results
        messages: List of message dictionaries for the API call
        use_structured_output: Whether to use structured output with Pydantic models

    Returns:
        Dictionary with processing result and metrics
    """
    result = {
        "provider": provider,
        "doc_id": doc_id,
        "status": "error",
        "message": "",
        "metrics": None
    }

    try:
        # Initialize model
        model = OpenAICompatibleModel(provider, model_name)

        # Verify files exist
        if not os.path.exists(image_path):
            result["message"] = f"Image not found: {image_path}"
            return result

        # Extract ground truth from the ground truth XML file
        gt_lines = extract_text_from_xml(gt_path)
        if not gt_lines:
            result["message"] = f"No ground truth text found in {gt_path}"
            return result

        # Call the model with either structured or regular output
        max_retries = 5
        base_wait_time = 1  # Start with 1 second
        
        for retry_count in range(max_retries):
            try:
                if use_structured_output:
                    # Use structured output with Pydantic
                    response = model.client.beta.chat.completions.parse(
                        model=model.model_name,
                        messages=messages,
                        response_format=Transcription,
                        temperature=0
                    )
                    
                    # Extract the transcription text from the structured response
                    transcription_text = response.choices[0].message.parsed.correct_transcription.strip()
                else:
                    # Use regular output (original approach)
                    response = model.client.chat.completions.create(
                        model=model.model_name,
                        messages=messages,
                        temperature=0,
                        seed=42
                    )
                    
                    # Extract the transcription text from the regular response
                    transcription_text = response.choices[0].message.content.strip()
                
                # If we got here, the call succeeded
                break
                
            except Exception as api_error:
                error_str = str(api_error)
                
                # Check if it's a rate limit error
                if "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                    if retry_count < max_retries - 1:  # Don't sleep on the last retry
                        # Calculate wait time with exponential backoff
                        wait_time = base_wait_time * (2 ** retry_count)
                        # Add some randomness to avoid all workers retrying at the same time
                        wait_time = wait_time * (0.75 + 0.5 * random.random())
                        print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds (attempt {retry_count+1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        # If we've exhausted all retries, re-raise the exception
                        raise
                else:
                    # If it's not a rate limit error, re-raise immediately
                    raise
        
        # Process response into lines
        transcription_lines = transcription_text.split('\n')

        # Save raw transcription
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{doc_id}_model_transcription.txt"), 'w', encoding='utf-8') as f:
            f.write(transcription_text)
        
        # Save ground truth for reference
        with open(os.path.join(output_dir, f"{doc_id}_ground_truth.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(gt_lines))

        # Calculate segmentation metrics
        segmentation_metrics = {
            'line_count_match': len(gt_lines) == len(transcription_lines),
            'gt_line_count': len(gt_lines),
            'transcription_line_count': len(transcription_lines),
            'segmentation_accuracy': min(len(transcription_lines), len(gt_lines)) / max(len(transcription_lines), len(gt_lines)) if max(len(transcription_lines), len(gt_lines)) > 0 else 1.0
        }

        # Create detailed comparison CSV
        with open(os.path.join(output_dir, f"{doc_id}_comparison.csv"), 'w', encoding='utf-8') as f:
            f.write("line_number,ground_truth,prediction,cer,wer,bwer\n")
            
            # Process all lines, handling mismatches properly
            max_lines = max(len(gt_lines), len(transcription_lines))
            
            for i in range(max_lines):
                # Get text for each source, using empty string if index is out of bounds
                gt_text = gt_lines[i] if i < len(gt_lines) else ""
                pred_text = transcription_lines[i] if i < len(transcription_lines) else ""
                
                # Clean commas for CSV
                gt_clean = gt_text.replace(",", " ") if gt_text else ""
                pred_clean = pred_text.replace(",", " ") if pred_text else ""
                
                # Ensure "None" string values are converted to empty strings, only for model predictions
                if pred_clean == "None":
                    pred_clean = ""
                
                # Calculate metrics based on text content
                if not gt_text and not pred_text:
                    # Both are empty - this is a perfect match
                    cer = 0.0
                    wer = 0.0
                    bwer = 0.0
                elif gt_text and pred_text:
                    # Both have content - calculate actual metrics
                    line_result = evaluate_transcription([gt_text], [pred_text])
                    line_metrics = line_result['line_metrics'][0]
                    cer = line_metrics['cer']
                    wer = line_metrics['wer']
                    bwer = line_metrics['bwer']
                else:
                    # One is empty and one has content - worst case
                    cer = 1.0
                    wer = 1.0
                    bwer = 1.0
                
                # Add line to CSV
                line_status = ""
                if i >= len(gt_lines) and pred_clean:  # Only mark as extra if prediction has content
                    line_status = "[extra]"
                elif i >= len(transcription_lines):
                    line_status = "[missed]"
                
                f.write(f"{i+1},{gt_clean},{pred_clean}{line_status},{cer},{wer},{bwer}\n")

        # Evaluate with alignment-based approach
        # Calculate metrics on the overlapping lines
        min_lines = min(len(gt_lines), len(transcription_lines))
        results = evaluate_transcription(gt_lines[:min_lines], transcription_lines[:min_lines])
        
        # Add segmentation metrics to results
        results['document_metrics'].update(segmentation_metrics)
        
        # Calculate Bag of Words metrics as specified in OCR-D
        from collections import Counter
        
        # Tokenize texts (simple whitespace tokenization)
        gt_tokens = ' '.join(gt_lines).split()
        pred_tokens = ' '.join(transcription_lines).split()
        
        # Calculate Bag of Words distributions
        gt_bow = Counter(gt_tokens)
        pred_bow = Counter(pred_tokens)
        
        # Calculate Bag of Words Error Rate (Jaccard distance)
        bow_intersection = sum((gt_bow & pred_bow).values())
        bow_union = sum((gt_bow | pred_bow).values())
        bow_error_rate = 1.0 - (bow_intersection / bow_union if bow_union > 0 else 0.0)
        
        # Add Bag of Words metrics
        results['document_metrics']['bow_error_rate'] = bow_error_rate

        # Save results
        save_results(results, output_dir, doc_id, method=provider)

        # Get document metrics
        doc_metrics = results['document_metrics']
        doc_metrics['document_id'] = doc_id

        result["status"] = "success"
        result["metrics"] = doc_metrics
        result["message"] = f"CER: {doc_metrics['cer']:.4f}, WER: {doc_metrics['wer']:.4f}, BWER: {doc_metrics['bwer']:.4f}, BOW: {bow_error_rate:.4f}, Seg: {segmentation_metrics['segmentation_accuracy']:.2f}"

    except Exception as e:
        result["message"] = str(e)
        print(f"Exception during processing: {str(e)}")
        import traceback
        traceback.print_exc()

    return result


def get_transkribus_text(doc_id: str, transkribus_dir: str) -> tuple[str, list]:
    """
    Get Transkribus transcription text for a document

    Args:
        doc_id: Document ID
        transkribus_dir: Directory containing Transkribus XML files

    Returns:
        Tuple of (full_text, lines_list) or ("", []) if not found
    """
    # Find Transkribus transcription file
    transkribus_path = find_file_for_id(doc_id, transkribus_dir, ['.xml'])

    if not transkribus_path or not os.path.exists(transkribus_path):
        return "", []

    # Extract text from Transkribus file
    transkribus_lines = extract_text_from_xml(transkribus_path)
    transkribus_text = "\n".join(transkribus_lines) if transkribus_lines else ""

    return transkribus_text, transkribus_lines


def _process_provider_evaluation(
    provider: str,
    model_name: str,
    gt_files: Dict[str, str],
    image_dir: str,
    base_output_dir: str,
    create_messages: callable,
    eval_type: str,
    limit: Optional[int] = None,
    print_callback: callable = print,
    use_structured_output: bool = False
):
    """
    Process evaluation for a single provider (used for parallel processing)

    Args:
        provider: Provider name
        model_name: Model name
        gt_files: Dictionary mapping doc_ids to ground truth file paths
        image_dir: Directory containing image files
        base_output_dir: Base output directory
        create_messages: Function to create messages for each document
        eval_type: Evaluation type
        limit: Maximum number of documents to process
        print_callback: Function to use for printing (useful for UI updates)
        use_structured_output: Whether to use structured output with Pydantic models

    Returns:
        Tuple of (all_results_df, comparison_data)
    """
    # Create output directory
    output_dir = f'{base_output_dir}/{eval_type}/{provider}'
    os.makedirs(output_dir, exist_ok=True)

    # Process all documents for this provider
    all_results = []

    # Create document list and apply limit if specified
    documents = list(gt_files.items())
    if limit is not None:
        documents = documents[:limit]

    # Process each document
    for doc_idx, (doc_id, gt_path) in enumerate(documents):
        # Status update
        print_callback(f"Processing {provider} document {doc_idx+1}/{len(documents)}: {doc_id}")

        # Get image path
        image_path = os.path.join(image_dir, f"{doc_id}.jpg")

        try:
            # Get messages for this document
            messages = create_messages(doc_id, image_path)

            # Process the document
            result = process_document(
                provider=provider,
                model_name=model_name,
                doc_id=doc_id,
                gt_path=gt_path,
                image_path=image_path,
                output_dir=output_dir,
                messages=messages,
                use_structured_output=use_structured_output
            )

            # Display result
            if result['status'] == 'success':
                print_callback(f"✅ {result['doc_id']}: {result['message']}")
                all_results.append(result['metrics'])
            else:
                print_callback(f"⚠️ {result['doc_id']}: {result['message']}")
        except Exception as e:
            print_callback(f"⚠️ Error processing {doc_id}: {str(e)}")
            continue


        time.sleep(1)

    # Compile all results for this provider
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(output_dir, f"{provider}_all_results.csv"), index=False)

        # Calculate aggregate metrics
        calculate_aggregate_metrics(output_dir, provider)

        # Create comparison data
        comparison_data = {
            "model": model_name,
            "avg_cer": all_results_df['cer'].mean(),
            "avg_wer": all_results_df['wer'].mean(),
            "avg_bwer": all_results_df['bwer'].mean(),
            "doc_count": len(all_results_df)
        }

        return all_results_df, comparison_data

    return None, None


def run_evaluation(
    provider_models: Dict[str, str],
    gt_dir: str,
    image_dir: str,
    base_output_dir: str,
    create_messages: callable,
    eval_type: str,
    limit: Optional[int] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    use_structured_output: bool = False
):
    """
    Run evaluation for multiple providers and models, optionally in parallel

    Args:
        provider_models: Dictionary mapping provider names to model names
        gt_dir: Directory containing ground truth files
        image_dir: Directory containing image files
        base_output_dir: Base output directory
        create_messages: Function that creates messages for each document (will be called for each doc)
        eval_type: Evaluation type (zero_shot, one_shot, etc.)
        limit: Maximum number of documents to process (None for all)
        parallel: Whether to run providers in parallel
        max_workers: Maximum number of parallel workers (None for auto-detection)
        use_structured_output: Whether to use structured output with Pydantic models

    Returns:
        Dictionary containing results for all providers
    """
    # Find available images first
    available_images = []
    for f in os.listdir(image_dir):
        if f.endswith('.jpg'):
            image_id = os.path.splitext(f)[0]
            available_images.append(image_id)

    print(f"Found {len(available_images)} images to process")

    # Get matching ground truth files
    gt_files = {}
    for image_id in available_images:
        gt_path = os.path.join(gt_dir, f"{image_id}.xml")
        if os.path.exists(gt_path):
            gt_files[image_id] = gt_path

    print(f"Found {len(gt_files)} matching ground truth files")

    # Store results
    all_provider_results = {}
    comparison_data = {}

    if parallel and len(provider_models) > 1:
        # Process providers in parallel
        print(f"Starting parallel evaluation with {len(provider_models)} providers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each provider
            future_to_provider = {
                executor.submit(
                    _process_provider_evaluation,
                    provider=provider,
                    model_name=model_name,
                    gt_files=gt_files,
                    image_dir=image_dir,
                    base_output_dir=base_output_dir,
                    create_messages=create_messages,
                    eval_type=eval_type,
                    limit=limit,
                    print_callback=print,
                    use_structured_output=use_structured_output
                ): provider
                for provider, model_name in provider_models.items()
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    provider_df, provider_comparison = future.result()
                    if provider_df is not None:
                        all_provider_results[provider] = provider_df
                        comparison_data[provider] = provider_comparison
                        print(f"✅ Completed evaluation for {provider}")
                except Exception as e:
                    print(f"⚠️ Error processing {provider}: {str(e)}")
    else:
        # Process sequentially (original behavior)
        for provider, model_name in provider_models.items():
            print(f"Evaluating {provider.upper()} with {model_name}")

            provider_df, provider_comparison = _process_provider_evaluation(
                provider=provider,
                model_name=model_name,
                gt_files=gt_files,
                image_dir=image_dir,
                base_output_dir=base_output_dir,
                create_messages=create_messages,
                eval_type=eval_type,
                limit=limit,
                use_structured_output=use_structured_output
            )

            if provider_df is not None:
                all_provider_results[provider] = provider_df
                comparison_data[provider] = provider_comparison

    # Save comparison data
    with open(os.path.join(base_output_dir, f"{eval_type}_comparison.json"), 'w') as f:
        json.dump(comparison_data, f, indent=2)

    return {
        "provider_results": all_provider_results,
        "comparison_data": comparison_data
    }


def process_document_by_lines(
    provider: str,
    model_name: str,
    doc_id: str,
    gt_path: str,
    image_path: str,
    transkribus_path: str,
    output_dir: str,
    messages_creator: callable,
    batch_size: int = 1,
    use_structured_output: bool = False
):
    """
    Process a document by individual lines using a vision language model.
    
    Args:
        provider: Provider name (e.g., 'openai', 'gemini', 'mistral')
        model_name: Model name or ID
        doc_id: Document ID
        gt_path: Path to ground truth XML file
        image_path: Path to document image
        transkribus_path: Path to Transkribus XML file (for line segmentation only)
        output_dir: Directory to save results
        messages_creator: Function to create messages for each line
        batch_size: Number of lines to process in each batch
        use_structured_output: Whether to use structured output
        
    Returns:
        Dictionary with processing results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize model
        model = OpenAICompatibleModel(provider, model_name)
        
        # STEP 1: Use Transkribus XML for LINE SEGMENTATION ONLY
        # This gives us coordinates and images for each line, but we ignore the text
        page_data = process_page_by_lines(image_path, transkribus_path)
        
        if not page_data['lines']:
            return {
                'status': 'error',
                'doc_id': doc_id,
                'message': f"No lines extracted from {image_path}",
                'metrics': {}
            }
        
        # STEP 2: Get ground truth lines separately from ground truth XML
        gt_lines = extract_text_from_xml(gt_path)
        if not gt_lines:
            return {
                'status': 'error',
                'doc_id': doc_id,
                'message': f"No ground truth text found in {gt_path}",
                'metrics': {}
            }
        
        # STEP 3: Also get Transkribus lines for reference and comparison
        transkribus_lines = extract_text_from_xml(transkribus_path)
        
        # STEP 4: Process lines individually
        all_pred_lines = []
        
        # Create batches of lines
        line_batches = [page_data['lines'][i:i + batch_size] for i in range(0, len(page_data['lines']), batch_size)]
        
        for batch_idx, line_batch in enumerate(line_batches):
            # Process each line in the batch
            for line_idx_in_batch, line_data in enumerate(line_batch):
                global_line_idx = batch_idx * batch_size + line_idx_in_batch
                
                try:
                    # Create messages for this line
                    line_messages = messages_creator(
                        doc_id=doc_id,
                        line_id=line_data['id'],
                        line_image=line_data['image'],
                        line_idx=global_line_idx
                    )
                    
                    # Add retry logic with exponential backoff
                    max_retries = 5
                    base_wait_time = 1  # Start with 1 second
                    
                    for retry_count in range(max_retries):
                        try:
                            # Use structured output if requested
                            if use_structured_output:
                                response = model.client.beta.chat.completions.parse(
                                    model=model.model_name,
                                    messages=line_messages,
                                    response_format=Transcription,
                                    temperature=0
                                )
                                line_text = response.choices[0].message.parsed.correct_transcription.strip()
                            else:
                                # Call the model normally
                                response = model.client.chat.completions.create(
                                    model=model.model_name,
                                    messages=line_messages,
                                    temperature=0
                                )
                                line_text = response.choices[0].message.content.strip()
                            
                            # If we got here, the call succeeded
                            break
                            
                        except Exception as api_error:
                            error_str = str(api_error)
                            
                            # Check if it's a rate limit error
                            if "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                                if retry_count < max_retries - 1:  # Don't sleep on the last retry
                                    # Calculate wait time with exponential backoff
                                    wait_time = base_wait_time * (2 ** retry_count)
                                    # Add some randomness to avoid all workers retrying at the same time
                                    wait_time = wait_time * (0.75 + 0.5 * random.random())
                                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds (attempt {retry_count+1}/{max_retries})...")
                                    time.sleep(wait_time)
                                else:
                                    # If we've exhausted all retries, re-raise the exception
                                    raise
                            else:
                                # If it's not a rate limit error, re-raise immediately
                                raise
                    
                    # Store results
                    all_pred_lines.append(line_text)
                    
                except Exception as e:
                    print(f"Error processing line {global_line_idx}: {str(e)}")
                    # If a line fails, use empty text
                    all_pred_lines.append("")
        
        # STEP 5: Calculate segmentation metrics
        # Filter out empty predictions for line counting
        non_empty_pred_lines = [line for line in all_pred_lines if line.strip()]
        
        # Compare the number of lines in ground truth vs. prediction (excluding empty lines)
        segmentation_metrics = {
            'line_count_match': len(gt_lines) == len(non_empty_pred_lines),
            'gt_line_count': len(gt_lines),
            'pred_line_count': len(non_empty_pred_lines),  # Count only non-empty lines
            'transkribus_line_count': len(transkribus_lines),
            'segmentation_accuracy': min(len(non_empty_pred_lines), len(gt_lines)) / max(len(non_empty_pred_lines), len(gt_lines)) if max(len(non_empty_pred_lines), len(gt_lines)) > 0 else 1.0
        }
        
        # STEP 6: Align prediction lines with ground truth for better comparison
        all_pred_lines_aligned = align_lines(gt_lines, all_pred_lines)
        
        # STEP 7: Create detailed comparison CSV
        with open(os.path.join(output_dir, f"{doc_id}_comparison.csv"), 'w', encoding='utf-8') as f:
            f.write("line_number,ground_truth,prediction,transkribus,cer,wer,bwer\n")
            
            # Process all lines, handling mismatches properly
            max_lines = max(len(gt_lines), len(all_pred_lines_aligned), len(transkribus_lines))
            
            for i in range(max_lines):
                # Get text for each source, using empty string if index is out of bounds
                gt_text = gt_lines[i] if i < len(gt_lines) else ""
                pred_text = all_pred_lines_aligned[i] if i < len(all_pred_lines_aligned) else ""
                trans_text = transkribus_lines[i] if i < len(transkribus_lines) else ""
                
                # Clean commas for CSV
                gt_clean = gt_text.replace(",", " ") if gt_text else ""
                pred_clean = pred_text.replace(",", " ") if pred_text else ""
                trans_clean = trans_text.replace(",", " ") if trans_text else ""
                
                # Ensure "None" string values are converted to empty strings, only for model predictions
                if pred_clean == "None":
                    pred_clean = ""
                
                # Calculate metrics based on text content
                if not gt_text and not pred_text:
                    # Both are empty - this is a perfect match
                    cer = 0.0
                    wer = 0.0
                    bwer = 0.0
                elif gt_text and pred_text:
                    # Both have content - calculate actual metrics
                    line_result = evaluate_transcription([gt_text], [pred_text])
                    line_metrics = line_result['line_metrics'][0]
                    cer = line_metrics['cer']
                    wer = line_metrics['wer']
                    bwer = line_metrics['bwer']
                else:
                    # One is empty and one has content - worst case
                    cer = 1.0
                    wer = 1.0
                    bwer = 1.0
                
                # Add line to CSV
                line_status = ""
                if i >= len(gt_lines) and pred_clean:  # Only mark as extra if prediction has content
                    line_status = "[extra]"
                elif i >= len(all_pred_lines_aligned):
                    line_status = "[missed]"
                
                f.write(f"{i+1},{gt_clean},{pred_clean}{line_status},{trans_clean},{cer},{wer},{bwer}\n")
        
        # STEP 8: Save full transcriptions
        with open(os.path.join(output_dir, f"{doc_id}_model_transcription.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_pred_lines))
            
        with open(os.path.join(output_dir, f"{doc_id}_ground_truth.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(gt_lines))
            
        with open(os.path.join(output_dir, f"{doc_id}_transkribus.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(transkribus_lines))
        
        # STEP 9: Use aligned predictions for metrics calculation
        min_lines = min(len(gt_lines), len(all_pred_lines_aligned))
        results = evaluate_transcription(gt_lines[:min_lines], all_pred_lines_aligned[:min_lines])
        
        # Add segmentation metrics to results
        results['document_metrics'].update(segmentation_metrics)
        
        # Calculate Bag of Words metrics as specified in OCR-D
        from collections import Counter
        
        # Tokenize texts (simple whitespace tokenization)
        gt_tokens = ' '.join(gt_lines).split()
        pred_tokens = ' '.join(all_pred_lines_aligned).split()
        
        # Calculate Bag of Words distributions
        gt_bow = Counter(gt_tokens)
        pred_bow = Counter(pred_tokens)
        
        # Calculate Bag of Words Error Rate (Jaccard distance)
        bow_intersection = sum((gt_bow & pred_bow).values())
        bow_union = sum((gt_bow | pred_bow).values())
        bow_error_rate = 1.0 - (bow_intersection / bow_union if bow_union > 0 else 0.0)
        
        # Add Bag of Words metrics
        results['document_metrics']['bow_error_rate'] = bow_error_rate
        
        # Save results
        save_results(results, output_dir, doc_id, method=provider)
        
        return {
            'status': 'success',
            'doc_id': doc_id,
            'message': f"CER: {results['document_metrics']['cer']:.4f}, WER: {results['document_metrics']['wer']:.4f}, BWER: {results['document_metrics']['bwer']:.4f}, BOW: {bow_error_rate:.4f}, Seg: {segmentation_metrics['segmentation_accuracy']:.2f}",
            'metrics': results['document_metrics']
        }
        
    except Exception as e:
        print(f"Exception during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'doc_id': doc_id,
            'message': str(e),
            'metrics': {}
        }


def _process_provider_line_evaluation(
    provider: str,
    model_name: str,
    documents: List[Tuple[str, str]],
    image_dir: str,
    transkribus_dir: str,
    base_output_dir: str,
    create_line_messages: callable,
    eval_type: str,
    batch_size: int = 1,
    print_callback: callable = print,
    use_structured_output: bool = False,
    rate_limit_delay: float = 0.0
):
    """
    Process line-based evaluation for a single provider (used for parallel processing)

    Args:
        provider: Provider name
        model_name: Model name
        documents: List of (doc_id, gt_path) tuples
        image_dir: Directory containing image files
        transkribus_dir: Directory containing Transkribus XML files
        base_output_dir: Base output directory
        create_line_messages: Function to create messages for each line
        eval_type: Evaluation type
        batch_size: Number of lines to process in a single API call
        print_callback: Function to use for printing (useful for UI updates)
        use_structured_output: Whether to use structured output with Pydantic models
        rate_limit_delay: Time to wait between API calls (in seconds) to avoid rate limits

    Returns:
        Tuple of (provider_results_df, provider_comparison_data)
    """
    # Create output directory
    output_dir = f'{base_output_dir}/{eval_type}/{provider}'
    os.makedirs(output_dir, exist_ok=True)

    print_callback(f"Starting evaluation for {provider.upper()} with {model_name}")

    # Process all documents for this provider
    all_results = []

    # Process each document
    for doc_idx, (doc_id, gt_path) in enumerate(documents):
        # Status update
        print_callback(f"Processing {provider} document {doc_idx+1}/{len(documents)}: {doc_id}")

        # Find corresponding transkribus file
        transkribus_path = find_file_for_id(doc_id, transkribus_dir, ['.xml'])

        if not transkribus_path:
            print_callback(f"⚠️ No Transkribus file found for {doc_id}")
            continue

        # Get image path
        image_path = os.path.join(image_dir, f"{doc_id}.jpg")

        try:
            # Process the document line by line using Transkribus XML for segmentation
            result = process_document_by_lines(
                provider=provider,
                model_name=model_name,
                doc_id=doc_id,
                gt_path=gt_path,
                image_path=image_path,
                transkribus_path=transkribus_path,
                output_dir=output_dir,
                messages_creator=create_line_messages,
                batch_size=batch_size,
                use_structured_output=use_structured_output
            )

            # Display result
            if result['status'] == 'success':
                print_callback(f"✅ {result['doc_id']}: {result['message']}")
                all_results.append(result['metrics'])
            else:
                print_callback(f"⚠️ {result['doc_id']}: {result['message']}")
        except Exception as e:
            print_callback(f"⚠️ Error processing {doc_id}: {str(e)}")
            continue


        time.sleep(rate_limit_delay)


    # Compile all results for this provider
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(output_dir, f"{provider}_all_results.csv"), index=False)

        # Calculate aggregate metrics
        calculate_aggregate_metrics(output_dir, provider)

        # Create comparison data
        comparison_data = {
            "model": model_name,
            "avg_cer": all_results_df['cer'].mean(),
            "avg_wer": all_results_df['wer'].mean(),
            "avg_bwer": all_results_df['bwer'].mean(),
            "doc_count": len(all_results_df)
        }

        return provider, all_results_df, comparison_data

    return provider, None, None


def run_line_evaluation(
    provider_models: Dict[str, str],
    gt_dir: str,
    image_dir: str,
    transkribus_dir: str,
    base_output_dir: str,
    create_line_messages: callable,
    eval_type: str,
    limit: Optional[int] = None,
    batch_size: int = 1,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    use_structured_output: bool = False,
    rate_limit_delay: float = 0.0
):
    """
    Run line-based evaluation for multiple providers and models, optionally in parallel

    Args:
        provider_models: Dictionary mapping provider names to model names
        gt_dir: Directory containing ground truth files
        image_dir: Directory containing image files
        transkribus_dir: Directory containing Transkribus XML files
        base_output_dir: Base output directory
        create_line_messages: Function to create messages for each line
        eval_type: Evaluation type (zero_shot_lines, one_shot_lines, etc.)
        limit: Maximum number of documents to process (None for all)
        batch_size: Number of lines to process in a single API call
        parallel: Whether to run providers in parallel
        max_workers: Maximum number of parallel workers (None for auto-detection)
        use_structured_output: Whether to use structured output with Pydantic models
        rate_limit_delay: Time to wait between API calls (in seconds) to avoid rate limits

    Returns:
        Dictionary containing results for all providers
    """
    # Find available images
    available_images = []
    for f in os.listdir(image_dir):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            image_id = os.path.splitext(f)[0]
            available_images.append(image_id)

    print(f"Found {len(available_images)} images to process")

    # Get matching ground truth files
    gt_files = {}
    for image_id in available_images:
        gt_path = os.path.join(gt_dir, f"{image_id}.xml")
        if os.path.exists(gt_path):
            gt_files[image_id] = gt_path

    print(f"Found {len(gt_files)} matching ground truth files")

    # Apply limit if specified
    documents = list(gt_files.items())
    if limit is not None:
        documents = documents[:limit]

    # Store results
    all_provider_results = {}
    comparison_data = {}

    if parallel and len(provider_models) > 1:
        # Process providers in parallel
        print(f"Starting parallel line-based evaluation with {len(provider_models)} providers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each provider
            future_to_provider = {
                executor.submit(
                    _process_provider_line_evaluation,
                    provider=provider,
                    model_name=model_name,
                    documents=documents,
                    image_dir=image_dir,
                    transkribus_dir=transkribus_dir,
                    base_output_dir=base_output_dir,
                    create_line_messages=create_line_messages,
                    eval_type=eval_type,
                    batch_size=batch_size,
                    print_callback=print,
                    use_structured_output=use_structured_output,
                    rate_limit_delay=rate_limit_delay
                ): provider
                for provider, model_name in provider_models.items()
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    provider_name, provider_df, provider_comparison = future.result()
                    if provider_df is not None:
                        all_provider_results[provider_name] = provider_df
                        comparison_data[provider_name] = provider_comparison
                        print(f"✅ Completed evaluation for {provider_name}")
                except Exception as e:
                    print(f"⚠️ Error processing {provider}: {str(e)}")
    else:
        # Process sequentially (original behavior)
        for provider, model_name in provider_models.items():
            provider_name, provider_df, provider_comparison = _process_provider_line_evaluation(
                provider=provider,
                model_name=model_name,
                documents=documents,
                image_dir=image_dir,
                transkribus_dir=transkribus_dir,
                base_output_dir=base_output_dir,
                create_line_messages=create_line_messages,
                eval_type=eval_type,
                batch_size=batch_size,
                use_structured_output=use_structured_output,
                rate_limit_delay=rate_limit_delay
            )
            
            if provider_df is not None:
                all_provider_results[provider_name] = provider_df
                comparison_data[provider_name] = provider_comparison

    # Save comparison data
    with open(os.path.join(base_output_dir, f"{eval_type}_comparison.json"), 'w') as f:
        json.dump(comparison_data, f, indent=2)

    return {
        "provider_results": all_provider_results,
        "comparison_data": comparison_data
    }


def parse_coords_points(points_str: str) -> Tuple[int, int, int, int]:
    """
    Parse coordinate points string from PAGE XML format.
    """
    try:
        # Split the points string and convert to integer pairs
        point_pairs = [p.split(',') for p in points_str.split()]
        points = [(int(p[0]), int(p[1])) for p in point_pairs]

        # Calculate bounding box
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]

        # Ensure proper ordering (x_min, y_min, x_max, y_max)
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # Validate coordinates
        if x_min >= x_max or y_min >= y_max:
            print(f"Warning: Invalid coordinates: {x_min},{y_min},{x_max},{y_max}")
            # Add 1 pixel to ensure valid box if needed
            if x_min == x_max:
                x_max += 1
            if y_min == y_max:
                y_max += 1

        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        # Return a minimal valid bounding box
        return (0, 0, 1, 1)


def run_single_file_evaluation(
    provider_models: Dict[str, str],
    doc_id: str,
    gt_dir: str,
    image_dir: str,
    transkribus_dir: str,
    base_output_dir: str,
    create_line_messages: callable,
    eval_type: str = 'single_file',
    use_structured_output: bool = False,
    rate_limit_delay: float = 0.0
):
    """
    Run evaluation for a single document ID with multiple providers.
    
    Args:
        provider_models: Dictionary mapping provider names to model names
        doc_id: Document ID to evaluate
        gt_dir: Directory containing ground truth files
        image_dir: Directory containing image files
        transkribus_dir: Directory containing Transkribus XML files
        base_output_dir: Base output directory for results
        create_line_messages: Function to create messages for each line
        eval_type: Evaluation type (default: 'single_file')
        use_structured_output: Whether to use structured output with Pydantic
        rate_limit_delay: Time to wait between API calls (in seconds) to avoid rate limits
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Running single file evaluation for document ID: {doc_id}")
    
    # Check if required files exist
    gt_path = os.path.join(gt_dir, f"{doc_id}.xml")
    if not os.path.exists(gt_path):
        gt_path = find_file_for_id(doc_id, gt_dir, ['.xml'])
        if not gt_path:
            print(f"⚠️ Ground truth file not found for document ID: {doc_id}")
            return None
    
    image_path = os.path.join(image_dir, f"{doc_id}.jpg")
    if not os.path.exists(image_path):
        image_path = find_file_for_id(doc_id, image_dir, ['.jpg', '.jpeg', '.png'])
        if not image_path:
            print(f"⚠️ Image file not found for document ID: {doc_id}")
            return None
    
    transkribus_path = find_file_for_id(doc_id, transkribus_dir, ['.xml'])
    if not transkribus_path:
        print(f"⚠️ Transkribus file not found for document ID: {doc_id}")
        return None
    
    # Store results for each provider
    all_provider_results = {}
    comparison_data = {}
    
    # Process each provider
    for provider, model_name in provider_models.items():
        print(f"Evaluating {provider.upper()} with {model_name}")
        
        # Create output directory
        output_dir = f'{base_output_dir}/{eval_type}/{provider}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the document line by line
        result = process_document_by_lines(
            provider=provider,
            model_name=model_name,
            doc_id=doc_id,
            gt_path=gt_path,
            image_path=image_path,
            transkribus_path=transkribus_path,
            output_dir=output_dir,
            messages_creator=create_line_messages,
            use_structured_output=use_structured_output
        )
        
        # Display result
        if result['status'] == 'success':
            print(f"✅ {doc_id}: {result['message']}")
            
            # Create a single-row DataFrame for this result
            provider_df = pd.DataFrame([result['metrics']])
            all_provider_results[provider] = provider_df
            
            # Add to comparison data
            comparison_data[provider] = {
                "model": model_name,
                "cer": result['metrics']['cer'],
                "wer": result['metrics']['wer'],
                "bwer": result['metrics']['bwer'],
                "bow_error_rate": result['metrics'].get('bow_error_rate', 0.0),
                "segmentation_accuracy": result['metrics']['segmentation_accuracy'],
                "doc_count": 1
            }
        else:
            print(f"⚠️ {doc_id}: {result['message']}")
            
        # Add delay to avoid rate limits between providers
        if rate_limit_delay > 0:
            print(f"Waiting {rate_limit_delay} seconds before next provider to avoid rate limits...")
            time.sleep(rate_limit_delay)
    
    # Save comparison data
    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, f"{eval_type}_{doc_id}_comparison.json"), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    return {
        "provider_results": all_provider_results,
        "comparison_data": comparison_data,
        "file_paths": {
            "gt_path": gt_path,
            "image_path": image_path,
            "transkribus_path": transkribus_path
        }
    }


def calculate_metrics_for_text(
    gt_text: Union[str, List[str]],
    pred_text: Union[str, List[str]],
    doc_id: str = "manual_evaluation"
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for a provided text against ground truth.
    
    Args:
        gt_text: Ground truth text (string or list of lines)
        pred_text: Predicted text (string or list of lines)
        doc_id: Optional document ID for reference
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to lists if strings are provided
    if isinstance(gt_text, str):
        gt_lines = gt_text.split('\n')
    else:
        gt_lines = gt_text
    
    if isinstance(pred_text, str):
        pred_lines = pred_text.split('\n')
    else:
        pred_lines = pred_text
    
    # Calculate segmentation metrics
    segmentation_metrics = {
        'line_count_match': len(gt_lines) == len(pred_lines),
        'gt_line_count': len(gt_lines),
        'pred_line_count': len(pred_lines),
        'segmentation_accuracy': min(len(pred_lines), len(gt_lines)) / max(len(pred_lines), len(gt_lines)) if max(len(pred_lines), len(gt_lines)) > 0 else 1.0
    }
    
    # Calculate text metrics on overlapping lines
    min_lines = min(len(gt_lines), len(pred_lines))
    results = evaluate_transcription(gt_lines[:min_lines], pred_lines[:min_lines])
    
    # Add segmentation metrics to results
    results['document_metrics'].update(segmentation_metrics)
    
    # Calculate Bag of Words metrics
    from collections import Counter
    
    # Tokenize texts (simple whitespace tokenization)
    gt_tokens = ' '.join(gt_lines).split()
    pred_tokens = ' '.join(pred_lines).split()
    
    # Calculate Bag of Words distributions
    gt_bow = Counter(gt_tokens)
    pred_bow = Counter(pred_tokens)
    
    # Calculate Bag of Words Error Rate (Jaccard distance)
    bow_intersection = sum((gt_bow & pred_bow).values())
    bow_union = sum((gt_bow | pred_bow).values())
    bow_error_rate = 1.0 - (bow_intersection / bow_union if bow_union > 0 else 0.0)
    
    # Add Bag of Words metrics
    results['document_metrics']['bow_error_rate'] = bow_error_rate
    
    # Create line-by-line comparison
    line_comparison = []
    max_lines = max(len(gt_lines), len(pred_lines))
    
    for i in range(max_lines):
        line_data = {
            'line_number': i + 1,
            'ground_truth': gt_lines[i] if i < len(gt_lines) else "",
            'prediction': pred_lines[i] if i < len(pred_lines) else ""
        }
        
        # Calculate line metrics if both texts exist
        if i < min_lines:
            line_metrics = results['line_metrics'][i]
            line_data.update({
                'cer': line_metrics['cer'],
                'wer': line_metrics['wer'],
                'bwer': line_metrics['bwer']
            })
        else:
            line_data.update({
                'cer': 1.0,
                'wer': 1.0,
                'bwer': 1.0,
                'status': 'missed' if i >= len(pred_lines) else 'extra'
            })
        
        line_comparison.append(line_data)
    
    # Add document ID to metrics
    results['document_metrics']['document_id'] = doc_id
    
    # Add summary string
    doc_metrics = results['document_metrics']
    summary = f"CER: {doc_metrics['cer']:.4f}, WER: {doc_metrics['wer']:.4f}, BWER: {doc_metrics['bwer']:.4f}, BOW: {bow_error_rate:.4f}, Seg: {segmentation_metrics['segmentation_accuracy']:.2f}"
    
    return {
        'metrics': results['document_metrics'],
        'line_metrics': results['line_metrics'],
        'line_comparison': line_comparison,
        'summary': summary
    }
