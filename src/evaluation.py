import os
import time
import pandas as pd
import json
import concurrent.futures
from typing import List, Dict, Any, Optional

from .models.openai_compatible import OpenAICompatibleModel
from .file_utils import encode_image, extract_text_from_xml, find_file_for_id
from .metrics import evaluate_transcription, save_results, calculate_aggregate_metrics


def process_document(
    provider: str,
    model_name: str,
    doc_id: str,
    gt_path: str,
    image_path: str,
    output_dir: str,
    messages: List[Dict]
):
    """
    Process a single document with the specified model

    Args:
        provider: Model provider name
        model_name: Model name
        doc_id: Document ID
        gt_path: Path to ground truth file
        image_path: Path to the image file
        output_dir: Output directory for results
        messages: List of message dictionaries for the API call

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

        # Extract ground truth
        gt_lines = extract_text_from_xml(gt_path)
        if not gt_lines:
            result["message"] = f"No ground truth text found in {gt_path}"
            return result

        # Call the model
        response = model.client.chat.completions.create(
            model=model.model_name,
            messages=messages,
            temperature=0
        )

        # Process response
        transcription_text = response.choices[0].message.content.strip()
        transcription_lines = transcription_text.split('\n')

        # Save raw transcription
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{doc_id}_transcription.txt"), 'w', encoding='utf-8') as f:
            f.write(transcription_text)

        # Evaluate
        results = evaluate_transcription(gt_lines, transcription_lines)

        # Save results
        save_results(results, output_dir, doc_id, method=provider)

        # Get document metrics
        doc_metrics = results['document_metrics']
        doc_metrics['document_id'] = doc_id

        result["status"] = "success"
        result["metrics"] = doc_metrics
        result["message"] = f"CER: {doc_metrics['cer']:.4f}, WER: {doc_metrics['wer']:.4f}, BWER: {doc_metrics['bwer']:.4f}"

    except Exception as e:
        result["message"] = str(e)

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
    print_callback: callable = print
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
                messages=messages
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

        # Add delay to avoid rate limits
        time.sleep(0.5)

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
    max_workers: Optional[int] = None
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
                    print_callback=print
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
                limit=limit
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
