import marimo

__generated_with = "0.11.30"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    import sys
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return mo, os, project_root, sys


@app.cell
def _(mo):
    mo.md(
        r"""
        # Phase 1: Quantitative Transkription - [Bentham Papers](https://regionalia.blb-karlsruhe.de/frontdoor/index/index/docId/22636)

        Handwritten English from the 18th Century
        ---
        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Traditional OCR Model - [Transkribus Text Titan 1](https://app.transkribus.org/models/text/51170)""")
    return


@app.cell
def _(mo):
    from src.transkribus import evaluate_transkribus

    # Evaluate Transkribus transcriptions
    transkribus_df = evaluate_transkribus(
        ground_truth_dir='data/reichenau_10_test/ground_truth',
        transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test/',
        output_dir='bentham_temp',
        save_transcriptions=False
    )

    # Calculate average metrics for Transkribus
    if not transkribus_df.empty:
        transkribus_avg_cer = transkribus_df['cer'].mean()
        transkribus_avg_wer = transkribus_df['wer'].mean()
        transkribus_avg_bwer = transkribus_df['bwer'].mean()
        transkribus_doc_count = len(transkribus_df)

        # Create stat display for Transkribus baseline
        mo.output.append(mo.md("## Traditional OCR Model - Transkribus Text Titan 1 Results"))
        mo.output.append(mo.vstack([
            mo.hstack([
                mo.stat(f"{transkribus_avg_cer:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{transkribus_avg_wer:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{transkribus_avg_bwer:.4f}", label="Average BWER", bordered=True),
                mo.stat(transkribus_doc_count, label="Documents", bordered=True)
            ])
        ]))

        mo.output.append(mo.md("### Detailed Results by Document"))
        mo.output.append(mo.callout(mo.plain(transkribus_df)))
    else:
        mo.md("## ⚠️ No Transkribus results available")
    return (
        evaluate_transkribus,
        transkribus_avg_bwer,
        transkribus_avg_cer,
        transkribus_avg_wer,
        transkribus_df,
        transkribus_doc_count,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## MM-LLM Zero-Shot Evaluation""")
    return


@app.cell
def _(mo, os):
    from src.models.openai_compatible import OpenAICompatibleModel
    from src.file_utils import encode_image, extract_text_from_xml
    from src.metrics import evaluate_transcription, save_results, calculate_aggregate_metrics
    import time
    import pandas as pd

    # System prompt for transcription
    zero_shot_system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=8, value="""
    You are a specialized transcription model for medieval german printed text.
    Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
    preserving the original text without modernizing or correcting spelling.
    Important instructions:
    1. Use original medieval characters and spelling (ſ, æ, etc.)
    2. Preserve abbreviations and special characters
    3. Separate each line with a newline character (\\n)
    4. Do not add any explanations or commentary
    5. Do not include line numbers
    6. Transcribe text only, ignore images or decorative elements
    Your transcription should match the original manuscript as closely as possible.

    CRITICAL LINE BREAK INSTRUCTIONS:
    - You MUST maintain the EXACT same number of lines as in the original manuscript
    - Each physical line in the manuscript should be ONE line in your transcription
    - DO NOT merge short lines together
    - DO NOT split long lines into multiple lines
    - Preserve the exact same line structure as the manuscript""")

    # Create a run button
    run_button = mo.ui.run_button(
        label="Run Evaluation",
        kind="success",
        tooltip="Start the evaluation process"
    )

    # Define models to evaluate
    provider_models = {
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "mistral": "pixtral-large-latest"
    }
    def process_document(provider, model_name, doc_id, gt_path, output_dir, system_prompt, custom_messages=None):
        """Process a single document with the specified model"""
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

            # Find corresponding image
            image_path = os.path.join('data/reichenau_inkunabeln/images', f"{doc_id}.jpg")
            if not os.path.exists(image_path):
                result["message"] = "No image found"
                return result

            # Extract ground truth
            gt_lines = extract_text_from_xml(gt_path)
            if not gt_lines:
                result["message"] = "No ground truth text found"
                return result

            # Encode image
            image_base64 = encode_image(image_path)

            # Use custom messages if provided, otherwise create default messages
            if custom_messages:
                messages = custom_messages
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please transcribe this historical manuscript image accurately, preserving the line breaks exactly as they appear."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ]

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

    # Display configuration UI
    mo.vstack([
        mo.md("## Zero-Shot Evaluation for OpenAI-Compatible Models"),
        mo.md("Configure the evaluation parameters and click the button to start."),
        mo.vstack([
            zero_shot_system_prompt,
            run_button
        ])
    ])
    return (
        OpenAICompatibleModel,
        calculate_aggregate_metrics,
        encode_image,
        evaluate_transcription,
        extract_text_from_xml,
        pd,
        process_document,
        provider_models,
        run_button,
        save_results,
        time,
        zero_shot_system_prompt,
    )


@app.cell(hide_code=True)
def _(
    calculate_aggregate_metrics,
    mo,
    os,
    pd,
    process_document,
    provider_models,
    run_button,
    time,
    zero_shot_system_prompt,
):
    if run_button.value:
        # Display evaluation in progress
        print("## Evaluation in Progress")

        # Find available images first
        image_dir = 'data/reichenau_10_test/images'
        available_images = []
        for f in os.listdir(image_dir):
            if f.endswith('.jpg'):
                image_id = os.path.splitext(f)[0]
                available_images.append(image_id)

        print(f"Found {len(available_images)} images to process")

        # Get matching ground truth files
        gt_dir = 'data/reichenau_10_test/ground_truth'
        gt_files = {}
        for image_id in available_images:
            gt_path = os.path.join(gt_dir, f"{image_id}.xml")
            if os.path.exists(gt_path):
                gt_files[image_id] = gt_path

        print(f"Found {len(gt_files)} matching ground truth files")

        # Store results
        all_provider_results = {}
        comparison_data = {}

        # Process each provider
        for provider, model_name in provider_models.items():
            print(f"### Evaluating {provider.upper()} with {model_name}")

            # Create output directory
            output_dir = f'temp/zero_shot/{provider}'
            os.makedirs(output_dir, exist_ok=True)

            # Process all documents for this provider
            all_results = []

            # Create document list for progress bar
            documents = list(gt_files.items())

            # Process each document with progress bar
            for doc_idx in mo.status.progress_bar(
                range(len(documents)),
                title=f"Processing {provider} documents",
                subtitle=f"Model: {model_name}"
            ):
                doc_id, gt_path = documents[doc_idx]

                # Process the document
                result = process_document(
                    provider=provider,
                    model_name=model_name,
                    doc_id=doc_id,
                    gt_path=gt_path,
                    output_dir=output_dir,
                    system_prompt=zero_shot_system_prompt.value
                )

                # Display result
                if result['status'] == 'success':
                    print(f"✅ **{result['doc_id']}**: {result['message']}")
                    all_results.append(result['metrics'])
                else:
                    print(f"⚠️ **{result['doc_id']}**: {result['message']}")

                # Add delay to avoid rate limits
                time.sleep(0.5)

            # Compile all results for this provider
            if all_results:
                all_results_df = pd.DataFrame(all_results)
                all_results_df.to_csv(os.path.join(output_dir, f"{provider}_all_results.csv"), index=False)

                # Calculate aggregate metrics
                calculate_aggregate_metrics(output_dir, provider)

                # Store for comparison
                all_provider_results[provider] = all_results_df

                # Add to comparison data
                comparison_data[provider] = {
                    "model": provider_models[provider],
                    "avg_cer": all_results_df['cer'].mean(),
                    "avg_wer": all_results_df['wer'].mean(),
                    "avg_bwer": all_results_df['bwer'].mean(),
                    "doc_count": len(all_results_df)
                }


        # Create comparison table if we have results from multiple providers
        if len(comparison_data) > 0:
            mo.md("## Comparison Across Providers")
            comparison_df = pd.DataFrame(comparison_data).T
            mo.ui.table(comparison_df)

            # Create stat components for each provider
            provider_stats = []
            for provider, data in comparison_data.items():
                model_name = data["model"]

                # Create stat components row for this provider
                provider_stats.append(mo.md(f"### {provider.upper()} with {model_name}"))
                provider_stats.append(
                    mo.hstack([
                        mo.stat(f"{data['avg_cer']:.4f}", label="Average CER", bordered=True),
                        mo.stat(f"{data['avg_wer']:.4f}", label="Average WER", bordered=True),
                        mo.stat(f"{data['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                        mo.stat(data["doc_count"], label="Documents", bordered=True)
                    ])
                )

            # Combine everything into a vstack layout
            mo.vstack([
                mo.md("## Model Performance Metrics"),
                *provider_stats
            ])

        all_provider_results
    return (
        all_provider_results,
        all_results,
        all_results_df,
        available_images,
        comparison_data,
        comparison_df,
        data,
        doc_id,
        doc_idx,
        documents,
        f,
        gt_dir,
        gt_files,
        gt_path,
        image_dir,
        image_id,
        model_name,
        output_dir,
        provider,
        provider_stats,
        result,
    )


@app.cell
def _(all_provider_results):
    all_provider_results
    return


@app.cell
def _(all_provider_results, mo, pd, provider_models, transkribus_df):
    transkribus_metrics = {
        "avg_cer": transkribus_df['cer'].mean() if not transkribus_df.empty else None,
        "avg_wer": transkribus_df['wer'].mean() if not transkribus_df.empty else None,
        "avg_bwer": transkribus_df['bwer'].mean() if not transkribus_df.empty else None
    }

    # Create fresh comparison data in this cell
    final_comparison_data = {}
    for provider_name, df in all_provider_results.items():
        if not df.empty:
            final_comparison_data[provider_name] = {
                "model": provider_models[provider_name],
                "avg_cer": df['cer'].mean(),
                "avg_wer": df['wer'].mean(),
                "avg_bwer": df['bwer'].mean(),
                "doc_count": len(df)
            }

    final_comparison_df = pd.DataFrame(final_comparison_data).T

    # Add Transkribus for comparison
    if not transkribus_df.empty:
        final_comparison_df.loc['transkribus'] = {
            "model": "Text Titan 1",
            "avg_cer": transkribus_metrics["avg_cer"],
            "avg_wer": transkribus_metrics["avg_wer"],
            "avg_bwer": transkribus_metrics["avg_bwer"],
            "doc_count": len(transkribus_df)
        }

    # Create fresh stat components for each provider with Transkribus comparison
    final_provider_stats = []
    for provider_name, metrics in final_comparison_data.items():
        _model_name = metrics["model"]

        # Skip Transkribus in this section (we'll show it separately)
        if provider_name == 'transkribus':
            continue

        # Compare with Transkribus metrics and determine direction
        if transkribus_metrics["avg_cer"] is not None:
            # Calculate absolute differences
            cer_diff = transkribus_metrics["avg_cer"] - metrics["avg_cer"]
            wer_diff = transkribus_metrics["avg_wer"] - metrics["avg_wer"]
            bwer_diff = transkribus_metrics["avg_bwer"] - metrics["avg_bwer"]

            # Calculate percentage changes (relative to Transkribus baseline)
            cer_pct = (cer_diff / transkribus_metrics["avg_cer"]) * 100 if transkribus_metrics["avg_cer"] > 0 else 0
            wer_pct = (wer_diff / transkribus_metrics["avg_wer"]) * 100 if transkribus_metrics["avg_wer"] > 0 else 0
            bwer_pct = (bwer_diff / transkribus_metrics["avg_bwer"]) * 100 if transkribus_metrics["avg_bwer"] > 0 else 0

            # For error metrics like CER/WER/BWER, lower is better
            # So if our model has lower error (cer_diff > 0), that's an improvement
            cer_direction = "increase" if cer_diff > 0 else "decrease"
            wer_direction = "increase" if wer_diff > 0 else "decrease"
            bwer_direction = "increase" if bwer_diff > 0 else "decrease"

            # Format percentage with sign (positive means improvement)
            cer_caption = f"{cer_pct:+.1f}% vs Transkribus"
            wer_caption = f"{wer_pct:+.1f}% vs Transkribus"
            bwer_caption = f"{bwer_pct:+.1f}% vs Transkribus"

            # Create stat components row for this provider
            final_provider_stats.append(mo.md(f"### {provider_name.upper()} with {_model_name}"))
            final_provider_stats.append(
                mo.hstack([
                    mo.stat(
                        f"{metrics['avg_cer']:.4f}",
                        label="Average CER",
                        caption=cer_caption,
                        direction=cer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{metrics['avg_wer']:.4f}",
                        label="Average WER",
                        caption=wer_caption,
                        direction=wer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{metrics['avg_bwer']:.4f}",
                        label="Average BWER",
                        caption=bwer_caption,
                        direction=bwer_direction,
                        bordered=True
                    ),
                    mo.stat(metrics["doc_count"], label="Documents", bordered=True)
                ])
            )
        else:
            # If Transkribus metrics aren't available, show regular stats
            final_provider_stats.append(mo.md(f"### {provider_name.upper()} with {_model_name}"))
            final_provider_stats.append(
                mo.hstack([
                    mo.stat(f"{metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                    mo.stat(metrics["doc_count"], label="Documents", bordered=True)
                ])
            )

    # Add Transkribus baseline stats at the top
    if not transkribus_df.empty:
        transkribus_stats = [
            mo.md("### TRANSKRIBUS with Text Titan 1 (Baseline)"),
            mo.hstack([
                mo.stat(f"{transkribus_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{transkribus_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{transkribus_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                mo.stat(len(transkribus_df), label="Documents", bordered=True)
            ])
        ]
    else:
        transkribus_stats = [mo.md("### Transkribus baseline not available")]

    # Combine everything into a vstack layout
    mo.vstack([
        mo.md("## Model Performance Metrics"),
        *transkribus_stats,
        mo.md("### MM-LLM Models"),
        *final_provider_stats,
        mo.md("## Comparison Table"),
        mo.ui.table(final_comparison_df)
    ])
    return (
        bwer_caption,
        bwer_diff,
        bwer_direction,
        bwer_pct,
        cer_caption,
        cer_diff,
        cer_direction,
        cer_pct,
        df,
        final_comparison_data,
        final_comparison_df,
        final_provider_stats,
        metrics,
        provider_name,
        transkribus_metrics,
        transkribus_stats,
        wer_caption,
        wer_diff,
        wer_direction,
        wer_pct,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Zero Shot Evaluation: Line-wise transcription
        TBD
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Hybrid Evaluation: MM-LLM Zero Shot + Transkribus""")
    return


@app.cell
def _(mo):
    zero_shot_hybrid_system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=8, value="""
    You are a specialized transcription model for medieval german printed text.
    Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
    preserving the original text without modernizing or correcting spelling.
    Important instructions:
    1. Use original medieval characters and spelling (ſ, æ, etc.)
    2. Preserve abbreviations and special characters
    3. Separate each line with a newline character (\\n)
    4. Do not add any explanations or commentary
    5. Do not include line numbers
    6. Transcribe text only, ignore images or decorative elements
    Your transcription should match the original manuscript as closely as possible.

    CRITICAL LINE BREAK INSTRUCTIONS:
    - You MUST maintain the EXACT same number of lines as in the original manuscript
    - Each physical line in the manuscript should be ONE line in your transcription
    - DO NOT merge short lines together
    - DO NOT split long lines into multiple lines
    - Preserve the exact same line structure as the manuscript""")

    # Create a run button
    hybrid_run_button = mo.ui.run_button(
        label="Run Evaluation",
        kind="success",
        tooltip="Start the evaluation process"
    )

    # Display configuration UI
    mo.vstack([
        mo.md("### Configure the evaluation parameters and click the button to start."),
        mo.vstack([
            zero_shot_hybrid_system_prompt,
            hybrid_run_button
        ])
    ])
    return hybrid_run_button, zero_shot_hybrid_system_prompt


@app.cell
def _(
    calculate_aggregate_metrics,
    encode_image,
    extract_text_from_xml,
    hybrid_run_button,
    mo,
    os,
    pd,
    process_document,
    provider_models,
    time,
    zero_shot_hybrid_system_prompt,
):
    from src.file_utils import find_file_for_id
    if hybrid_run_button.value:
        # Display evaluation in progress
        print("## Hybrid Evaluation in Progress (MM-LLM + Transkribus)")

        # Find available images first
        hybrid_image_dir = 'data/reichenau_10_test/images'
        hybrid_available_images = []
        for hybrid_f in os.listdir(hybrid_image_dir):
            if hybrid_f.endswith('.jpg'):
                hybrid_image_id = os.path.splitext(hybrid_f)[0]
                hybrid_available_images.append(hybrid_image_id)

        print(f"Found {len(hybrid_available_images)} images to process")

        # Get matching ground truth files
        hybrid_gt_dir = 'data/reichenau_10_test/ground_truth'
        hybrid_transkribus_dir = 'results/linear_transcription/reichenau_inkunabeln/transkribus_10_test'
        hybrid_gt_files = {}
        for hybrid_image_id in hybrid_available_images:
            hybrid_gt_path = os.path.join(hybrid_gt_dir, f"{hybrid_image_id}.xml")
            if os.path.exists(hybrid_gt_path):
                hybrid_gt_files[hybrid_image_id] = hybrid_gt_path

        print(f"Found {len(hybrid_gt_files)} matching ground truth files")

        # Store results
        hybrid_results = {}
        hybrid_comparison_data = {}

        # Process each provider
        for hybrid_provider, hybrid_model_name in provider_models.items():
            print(f"### Evaluating Hybrid {hybrid_provider.upper()} + Transkribus with {hybrid_model_name}")

            # Create output directory
            hybrid_output_dir = f'bentham_temp/hybrid_zero_shot/{hybrid_provider}'
            os.makedirs(hybrid_output_dir, exist_ok=True)

            # Process all documents for this provider
            hybrid_all_results = []

            # Create document list for progress bar
            hybrid_documents = list(hybrid_gt_files.items())

            # Process each document with progress bar
            for hybrid_doc_idx in mo.status.progress_bar(
                range(len(hybrid_documents)),
                title=f"Processing {hybrid_provider} documents",
                subtitle=f"Model: {hybrid_model_name}"
            ):
                hybrid_doc_id, hybrid_gt_path = hybrid_documents[hybrid_doc_idx]

                # Find Transkribus transcription file
                hybrid_transkribus_path = find_file_for_id(hybrid_doc_id, hybrid_transkribus_dir, ['.xml'])

                if hybrid_transkribus_path:
                    # Extract text from Transkribus file
                    hybrid_transkribus_lines = extract_text_from_xml(hybrid_transkribus_path)
                    hybrid_transkribus_text = "\n".join(hybrid_transkribus_lines) if hybrid_transkribus_lines else ""

                    # Encode image
                    hybrid_image_path = os.path.join('data/reichenau_10_test/images', f"{hybrid_doc_id}.jpg")
                    hybrid_image_base64 = encode_image(hybrid_image_path)

                    # Create custom messages with Transkribus transcription
                    hybrid_custom_messages = [
                        {"role": "system", "content": zero_shot_hybrid_system_prompt.value},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Please transcribe this historical manuscript image accurately, preserving the line breaks exactly as they appear. The following is the output of a traditional OCR model from Transkribus. It is fine-tuned on medieval texts. It can help you transcribe the page, but may also contain errors:\n\n{hybrid_transkribus_text}"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{hybrid_image_base64}"}
                                }
                            ]
                        }
                    ]

                    # Process the document with custom messages
                    hybrid_result = process_document(
                        provider=f"{hybrid_provider}",
                        model_name=hybrid_model_name,
                        doc_id=hybrid_doc_id,
                        gt_path=hybrid_gt_path,
                        output_dir=hybrid_output_dir,
                        system_prompt=zero_shot_hybrid_system_prompt.value,
                        custom_messages=hybrid_custom_messages
                    )
                else:
                    # No Transkribus file found
                    hybrid_result = {
                        "provider": f"{hybrid_provider}",
                        "doc_id": hybrid_doc_id,
                        "status": "error",
                        "message": "No Transkribus transcription found"
                    }

                # Display result
                if hybrid_result['status'] == 'success':
                    print(f"✅ **{hybrid_result['doc_id']}**: {hybrid_result['message']}")
                    hybrid_all_results.append(hybrid_result['metrics'])
                else:
                    print(f"⚠️ **{hybrid_result['doc_id']}**: {hybrid_result['message']}")

                # Add delay to avoid rate limits
                time.sleep(0.5)

            # Compile all results for this provider
            if hybrid_all_results:
                hybrid_results_df = pd.DataFrame(hybrid_all_results)
                hybrid_results_df.to_csv(os.path.join(hybrid_output_dir, f"{hybrid_provider}_hybrid_all_results.csv"), index=False)

                # Calculate aggregate metrics
                calculate_aggregate_metrics(hybrid_output_dir, f"{hybrid_provider}")

                # Store for comparison
                hybrid_results[f"{hybrid_provider}"] = hybrid_results_df

                # Add to comparison data
                hybrid_comparison_data[f"{hybrid_provider}"] = {
                    "model": f"{provider_models[hybrid_provider]} + Transkribus",
                    "avg_cer": hybrid_results_df['cer'].mean(),
                    "avg_wer": hybrid_results_df['wer'].mean(),
                    "avg_bwer": hybrid_results_df['bwer'].mean(),
                    "doc_count": len(hybrid_results_df)
                }

                # Show summary
                print(f"**Summary for {hybrid_provider} Hybrid ({hybrid_model_name} + Transkribus):**")
                mo.output.append(mo.vstack([
                    mo.hstack([
                        mo.stat(f"{hybrid_results_df['cer'].mean():.4f}", label="Average CER", bordered=True),
                        mo.stat(f"{hybrid_results_df['wer'].mean():.4f}", label="Average WER", bordered=True),
                        mo.stat(f"{hybrid_results_df['bwer'].mean():.4f}", label="Average BWER", bordered=True),
                        mo.stat(len(hybrid_results_df), label="Documents", bordered=True)
                    ])
                ]))
    return (
        find_file_for_id,
        hybrid_all_results,
        hybrid_available_images,
        hybrid_comparison_data,
        hybrid_custom_messages,
        hybrid_doc_id,
        hybrid_doc_idx,
        hybrid_documents,
        hybrid_f,
        hybrid_gt_dir,
        hybrid_gt_files,
        hybrid_gt_path,
        hybrid_image_base64,
        hybrid_image_dir,
        hybrid_image_id,
        hybrid_image_path,
        hybrid_model_name,
        hybrid_output_dir,
        hybrid_provider,
        hybrid_result,
        hybrid_results,
        hybrid_results_df,
        hybrid_transkribus_dir,
        hybrid_transkribus_lines,
        hybrid_transkribus_path,
        hybrid_transkribus_text,
    )


@app.cell
def _(hybrid_results):
    hybrid_results
    return


@app.cell
def _(hybrid_results, mo, pd, provider_models, transkribus_df):
    hybrid_transkribus_metrics = {
        "avg_cer": transkribus_df['cer'].mean() if not transkribus_df.empty else None,
        "avg_wer": transkribus_df['wer'].mean() if not transkribus_df.empty else None,
        "avg_bwer": transkribus_df['bwer'].mean() if not transkribus_df.empty else None
    }

    # Create fresh comparison data in this cell
    hybrid_final_comparison_data = {}
    for h_provider_name, h_df in hybrid_results.items():
        if not h_df.empty:
            hybrid_final_comparison_data[h_provider_name] = {
                "model": f"{provider_models[h_provider_name.replace('_hybrid', '')]} + Transkribus",
                "avg_cer": h_df['cer'].mean(),
                "avg_wer": h_df['wer'].mean(),
                "avg_bwer": h_df['bwer'].mean(),
                "doc_count": len(h_df)
            }

    hybrid_final_comparison_df = pd.DataFrame(hybrid_final_comparison_data).T

    # Add Transkribus for comparison
    if not transkribus_df.empty:
        hybrid_final_comparison_df.loc['transkribus'] = {
            "model": "Text Titan 1",
            "avg_cer": hybrid_transkribus_metrics["avg_cer"],
            "avg_wer": hybrid_transkribus_metrics["avg_wer"],
            "avg_bwer": hybrid_transkribus_metrics["avg_bwer"],
            "doc_count": len(transkribus_df)
        }

    # Create fresh stat components for each provider with Transkribus comparison
    hybrid_final_provider_stats = []
    for h_provider_name, h_metrics in hybrid_final_comparison_data.items():
        h_model_name = h_metrics["model"]

        # Skip Transkribus in this section
        if h_provider_name == 'transkribus':
            continue

        # Compare with Transkribus metrics and determine direction
        if hybrid_transkribus_metrics["avg_cer"] is not None:
            # Calculate absolute differences
            hybrid_cer_diff = hybrid_transkribus_metrics["avg_cer"] - h_metrics["avg_cer"]
            hybrid_wer_diff = hybrid_transkribus_metrics["avg_wer"] - h_metrics["avg_wer"]
            hybrid_bwer_diff = hybrid_transkribus_metrics["avg_bwer"] - h_metrics["avg_bwer"]

            # Calculate percentage changes (relative to Transkribus baseline)
            hybrid_cer_pct = (hybrid_cer_diff / hybrid_transkribus_metrics["avg_cer"]) * 100 if hybrid_transkribus_metrics["avg_cer"] > 0 else 0
            hybrid_wer_pct = (hybrid_wer_diff / hybrid_transkribus_metrics["avg_wer"]) * 100 if hybrid_transkribus_metrics["avg_wer"] > 0 else 0
            hybrid_bwer_pct = (hybrid_bwer_diff / hybrid_transkribus_metrics["avg_bwer"]) * 100 if hybrid_transkribus_metrics["avg_bwer"] > 0 else 0

            # For error metrics like CER/WER/BWER, lower is better
            # So if our model has lower error (hybrid_cer_diff > 0), that's an improvement
            hybrid_cer_direction = "increase" if hybrid_cer_diff > 0 else "decrease"
            hybrid_wer_direction = "increase" if hybrid_wer_diff > 0 else "decrease"
            hybrid_bwer_direction = "increase" if hybrid_bwer_diff > 0 else "decrease"

            # Format percentage with sign (positive means improvement)
            hybrid_cer_caption = f"{hybrid_cer_pct:+.1f}% vs Transkribus"
            hybrid_wer_caption = f"{hybrid_wer_pct:+.1f}% vs Transkribus"
            hybrid_bwer_caption = f"{hybrid_bwer_pct:+.1f}% vs Transkribus"

            # Create stat components row for this provider
            hybrid_final_provider_stats.append(mo.md(f"### {h_provider_name.upper()}"))
            hybrid_final_provider_stats.append(
                mo.hstack([
                    mo.stat(
                        f"{h_metrics['avg_cer']:.4f}",
                        label="Average CER",
                        caption=hybrid_cer_caption,
                        direction=hybrid_cer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{h_metrics['avg_wer']:.4f}",
                        label="Average WER",
                        caption=hybrid_wer_caption,
                        direction=hybrid_wer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{h_metrics['avg_bwer']:.4f}",
                        label="Average BWER",
                        caption=hybrid_bwer_caption,
                        direction=hybrid_bwer_direction,
                        bordered=True
                    ),
                    mo.stat(h_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )
        else:
            # If Transkribus metrics aren't available, show regular stats
            hybrid_final_provider_stats.append(mo.md(f"### {h_provider_name.upper()}"))
            hybrid_final_provider_stats.append(
                mo.hstack([
                    mo.stat(f"{h_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{h_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{h_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                    mo.stat(h_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )

    # Add Transkribus baseline stats at the top
    if not transkribus_df.empty:
        hybrid_transkribus_stats = [
            mo.md("### TRANSKRIBUS with Text Titan 1 (Baseline)"),
            mo.hstack([
                mo.stat(f"{hybrid_transkribus_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{hybrid_transkribus_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{hybrid_transkribus_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                mo.stat(len(transkribus_df), label="Documents", bordered=True)
            ])
        ]
    else:
        hybrid_transkribus_stats = [mo.md("### Transkribus baseline not available")]

    # Combine everything into a vstack layout
    mo.vstack([
        mo.md("## Hybrid Model Performance Metrics"),
        *hybrid_transkribus_stats,
        mo.md("### MM-LLM + Transkribus Hybrid Models"),
        *hybrid_final_provider_stats,
        mo.md("## Comparison Table"),
        mo.ui.table(hybrid_final_comparison_df)
    ])
    return (
        h_df,
        h_metrics,
        h_model_name,
        h_provider_name,
        hybrid_bwer_caption,
        hybrid_bwer_diff,
        hybrid_bwer_direction,
        hybrid_bwer_pct,
        hybrid_cer_caption,
        hybrid_cer_diff,
        hybrid_cer_direction,
        hybrid_cer_pct,
        hybrid_final_comparison_data,
        hybrid_final_comparison_df,
        hybrid_final_provider_stats,
        hybrid_transkribus_metrics,
        hybrid_transkribus_stats,
        hybrid_wer_caption,
        hybrid_wer_diff,
        hybrid_wer_direction,
        hybrid_wer_pct,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## One-Shot Evaluation for MM-LLMs""")
    return


@app.cell(hide_code=True)
def _(mo):
    one_shot_llm_system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=8, value="""
    You are a specialized transcription model for medieval german printed text.
    Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
    preserving the original text without modernizing or correcting spelling.
    Important instructions:
    1. Use original medieval characters and spelling (ſ, æ, etc.)
    2. Preserve abbreviations and special characters
    3. Separate each line with a newline character (\\n)
    4. Do not add any explanations or commentary
    5. Do not include line numbers
    6. Transcribe text only, ignore images or decorative elements
    Your transcription should match the original manuscript as closely as possible.

    CRITICAL LINE BREAK INSTRUCTIONS:
    - You MUST maintain the EXACT same number of lines as in the original manuscript
    - Each physical line in the manuscript should be ONE line in your transcription
    - DO NOT merge short lines together
    - DO NOT split long lines into multiple lines
    - Preserve the exact same line structure as the manuscript""")

    # Create a run button for one-shot LLM
    one_shot_llm_run_button = mo.ui.run_button(
        label="Run One-Shot LLM Evaluation",
        kind="success",
        tooltip="Start the one-shot LLM evaluation process"
    )

    # Display configuration UI
    mo.vstack([
        mo.md("### Configure the One-Shot LLM evaluation parameters and click the button to start."),
        mo.vstack([
            one_shot_llm_system_prompt,
            one_shot_llm_run_button
        ])
    ])
    return one_shot_llm_run_button, one_shot_llm_system_prompt


@app.cell
def _(
    calculate_aggregate_metrics,
    encode_image,
    extract_text_from_xml,
    mo,
    one_shot_llm_run_button,
    one_shot_llm_system_prompt,
    os,
    pd,
    process_document,
    provider_models,
    time,
):
    def get_one_shot_llm_example_content():
        """Get content of example page for one-shot learning"""
        # Direct path to example file
        example_gt_path = 'data/reichenau_10_test/few-shot-samples/7474192.xml'  # Path to example ground truth
        example_img_path = 'data/reichenau_10_test/few-shot-samples/7474192.jpg'  # Path to example image

        if os.path.exists(example_gt_path):
            example_lines = extract_text_from_xml(example_gt_path)
            example_text = "\n".join(example_lines) if example_lines else ""

            return {
                "text": example_text,
                "image_path": example_img_path if os.path.exists(example_img_path) else None,
                "success": True
            }

        return {
            "text": "",
            "image_path": None,
            "success": False
        }

    if one_shot_llm_run_button.value:
        # Display evaluation in progress
        mo.md("## One-Shot LLM Evaluation in Progress (MM-LLM Only)")

        # Get example content for one-shot learning
        one_shot_llm_example = get_one_shot_llm_example_content()
        if one_shot_llm_example["success"]:
            mo.md(f"✅ Using example from dedicated example folder for one-shot learning")
        else:
            mo.md("⚠️ Using fallback example for one-shot learning - results may be suboptimal")

        # Find available images first
        one_shot_llm_image_dir = 'data/reichenau_10_test/images'
        one_shot_llm_available_images = []
        for one_shot_llm_f in os.listdir(one_shot_llm_image_dir):
            if one_shot_llm_f.endswith('.jpg'):
                one_shot_llm_image_id = os.path.splitext(one_shot_llm_f)[0]
                one_shot_llm_available_images.append(one_shot_llm_image_id)

        mo.md(f"Found {len(one_shot_llm_available_images)} images to process")

        # Get matching ground truth files
        one_shot_llm_gt_dir = 'data/reichenau_10_test/ground_truth'
        one_shot_llm_gt_files = {}
        for one_shot_llm_image_id in one_shot_llm_available_images:
            one_shot_llm_gt_path = os.path.join(one_shot_llm_gt_dir, f"{one_shot_llm_image_id}.xml")
            if os.path.exists(one_shot_llm_gt_path):
                one_shot_llm_gt_files[one_shot_llm_image_id] = one_shot_llm_gt_path

        mo.md(f"Found {len(one_shot_llm_gt_files)} matching ground truth files")

        # Store results
        one_shot_llm_results = {}
        one_shot_llm_comparison_data = {}

        # Process each provider
        for one_shot_llm_provider, one_shot_llm_model_name in provider_models.items():
            mo.md(f"### Evaluating One-Shot LLM {one_shot_llm_provider.upper()} with {one_shot_llm_model_name}")

            # Create output directory
            one_shot_llm_output_dir = f'bentham_temp/one_shot_llm/{one_shot_llm_provider}'
            os.makedirs(one_shot_llm_output_dir, exist_ok=True)

            # Process all documents for this provider
            one_shot_llm_all_results = []

            # Create document list for progress bar
            one_shot_llm_documents = list(one_shot_llm_gt_files.items())

            # Process each document with progress bar
            for one_shot_llm_doc_idx in mo.status.progress_bar(
                range(len(one_shot_llm_documents)),
                title=f"Processing {one_shot_llm_provider} documents",
                subtitle=f"Model: {one_shot_llm_model_name}"
            ):
                one_shot_llm_doc_id, one_shot_llm_gt_path = one_shot_llm_documents[one_shot_llm_doc_idx]

                # Encode image
                one_shot_llm_image_path = os.path.join('data/reichenau_10_test/images', f"{one_shot_llm_doc_id}.jpg")
                one_shot_llm_image_base64 = encode_image(one_shot_llm_image_path)

                # Create custom messages with example transcription (but NO Transkribus transcription)
                one_shot_llm_custom_messages = [
                    {"role": "system", "content": one_shot_llm_system_prompt.value},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is an example of a historical manuscript page and its correct transcription:"},
                        ]
                    }
                ]

                # Add example image if available
                if one_shot_llm_example["image_path"] and os.path.exists(one_shot_llm_example["image_path"]):
                    example_image_base64 = encode_image(one_shot_llm_example["image_path"])
                    one_shot_llm_custom_messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{example_image_base64}"}
                    })

                # Add example transcription text
                one_shot_llm_custom_messages[1]["content"].append(
                    {"type": "text", "text": f"Example transcription:\n{one_shot_llm_example['text']}"}
                )

                # Complete the custom messages
                one_shot_llm_custom_messages.extend([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please transcribe this new historical manuscript image accurately, preserving the line breaks exactly as they appear."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{one_shot_llm_image_base64}"}
                            }
                        ]
                    }
                ])

                # Process the document with custom messages
                one_shot_llm_result = process_document(
                    provider=f"{one_shot_llm_provider}",
                    model_name=one_shot_llm_model_name,
                    doc_id=one_shot_llm_doc_id,
                    gt_path=one_shot_llm_gt_path,
                    output_dir=one_shot_llm_output_dir,
                    system_prompt=one_shot_llm_system_prompt.value,
                    custom_messages=one_shot_llm_custom_messages
                )

                # Display result
                if one_shot_llm_result['status'] == 'success':
                    mo.md(f"✅ **{one_shot_llm_result['doc_id']}**: {one_shot_llm_result['message']}")
                    one_shot_llm_all_results.append(one_shot_llm_result['metrics'])
                else:
                    mo.md(f"⚠️ **{one_shot_llm_result['doc_id']}**: {one_shot_llm_result['message']}")

                # Add delay to avoid rate limits
                time.sleep(0.5)

            # Compile all results for this provider
            if one_shot_llm_all_results:
                one_shot_llm_results_df = pd.DataFrame(one_shot_llm_all_results)
                one_shot_llm_results_df.to_csv(os.path.join(one_shot_llm_output_dir, f"{one_shot_llm_provider}_one_shot_llm_results.csv"), index=False)

                # Calculate aggregate metrics
                calculate_aggregate_metrics(one_shot_llm_output_dir, f"{one_shot_llm_provider}")

                # Store for comparison
                one_shot_llm_results[f"{one_shot_llm_provider}"] = one_shot_llm_results_df

                # Add to comparison data
                one_shot_llm_comparison_data[f"{one_shot_llm_provider}"] = {
                    "model": f"{provider_models[one_shot_llm_provider]} (One-Shot)",
                    "avg_cer": one_shot_llm_results_df['cer'].mean(),
                    "avg_wer": one_shot_llm_results_df['wer'].mean(),
                    "avg_bwer": one_shot_llm_results_df['bwer'].mean(),
                    "doc_count": len(one_shot_llm_results_df)
                }

                # Show summary
                mo.md(f"**Summary for {one_shot_llm_provider} One-Shot ({one_shot_llm_model_name}):**")
                mo.vstack([
                    mo.hstack([
                        mo.stat(f"{one_shot_llm_results_df['cer'].mean():.4f}", label="Average CER", bordered=True),
                        mo.stat(f"{one_shot_llm_results_df['wer'].mean():.4f}", label="Average WER", bordered=True),
                        mo.stat(f"{one_shot_llm_results_df['bwer'].mean():.4f}", label="Average BWER", bordered=True),
                        mo.stat(len(one_shot_llm_results_df), label="Documents", bordered=True)
                    ])
                ])
    return (
        example_image_base64,
        get_one_shot_llm_example_content,
        one_shot_llm_all_results,
        one_shot_llm_available_images,
        one_shot_llm_comparison_data,
        one_shot_llm_custom_messages,
        one_shot_llm_doc_id,
        one_shot_llm_doc_idx,
        one_shot_llm_documents,
        one_shot_llm_example,
        one_shot_llm_f,
        one_shot_llm_gt_dir,
        one_shot_llm_gt_files,
        one_shot_llm_gt_path,
        one_shot_llm_image_base64,
        one_shot_llm_image_dir,
        one_shot_llm_image_id,
        one_shot_llm_image_path,
        one_shot_llm_model_name,
        one_shot_llm_output_dir,
        one_shot_llm_provider,
        one_shot_llm_result,
        one_shot_llm_results,
        one_shot_llm_results_df,
    )


@app.cell
def _(one_shot_llm_results):
    one_shot_llm_results
    return


@app.cell
def _(mo, one_shot_llm_results, pd, provider_models, transkribus_df):
    one_shot_llm_transkribus_metrics = {
        "avg_cer": transkribus_df['cer'].mean() if not transkribus_df.empty else None,
        "avg_wer": transkribus_df['wer'].mean() if not transkribus_df.empty else None,
        "avg_bwer": transkribus_df['bwer'].mean() if not transkribus_df.empty else None
    }

    # Create fresh comparison data in this cell
    one_shot_llm_final_comparison_data = {}
    for os_llm_provider_name, os_llm_df in one_shot_llm_results.items():
        if not os_llm_df.empty:
            one_shot_llm_final_comparison_data[os_llm_provider_name] = {
                "model": f"{provider_models[os_llm_provider_name.replace('_one_shot_llm', '')]} (One-Shot)",
                "avg_cer": os_llm_df['cer'].mean(),
                "avg_wer": os_llm_df['wer'].mean(),
                "avg_bwer": os_llm_df['bwer'].mean(),
                "doc_count": len(os_llm_df)
            }

    one_shot_llm_final_comparison_df = pd.DataFrame(one_shot_llm_final_comparison_data).T

    # Add Transkribus for comparison
    if not transkribus_df.empty:
        one_shot_llm_final_comparison_df.loc['transkribus'] = {
            "model": "Text Titan 1",
            "avg_cer": one_shot_llm_transkribus_metrics["avg_cer"],
            "avg_wer": one_shot_llm_transkribus_metrics["avg_wer"],
            "avg_bwer": one_shot_llm_transkribus_metrics["avg_bwer"],
            "doc_count": len(transkribus_df)
        }

    # Create fresh stat components for each provider with Transkribus comparison
    one_shot_llm_final_provider_stats = []
    for os_llm_provider_name, os_llm_metrics in one_shot_llm_final_comparison_data.items():
        os_llm_model_name = os_llm_metrics["model"]

        # Skip Transkribus in this section
        if os_llm_provider_name == 'transkribus':
            continue

        # Compare with Transkribus metrics and determine direction
        if one_shot_llm_transkribus_metrics["avg_cer"] is not None:
            # Calculate absolute differences
            os_llm_cer_diff = one_shot_llm_transkribus_metrics["avg_cer"] - os_llm_metrics["avg_cer"]
            os_llm_wer_diff = one_shot_llm_transkribus_metrics["avg_wer"] - os_llm_metrics["avg_wer"]
            os_llm_bwer_diff = one_shot_llm_transkribus_metrics["avg_bwer"] - os_llm_metrics["avg_bwer"]

            # Calculate percentage changes (relative to Transkribus baseline)
            os_llm_cer_pct = (os_llm_cer_diff / one_shot_llm_transkribus_metrics["avg_cer"]) * 100 if one_shot_llm_transkribus_metrics["avg_cer"] > 0 else 0
            os_llm_wer_pct = (os_llm_wer_diff / one_shot_llm_transkribus_metrics["avg_wer"]) * 100 if one_shot_llm_transkribus_metrics["avg_wer"] > 0 else 0
            os_llm_bwer_pct = (os_llm_bwer_diff / one_shot_llm_transkribus_metrics["avg_bwer"]) * 100 if one_shot_llm_transkribus_metrics["avg_bwer"] > 0 else 0

            # For error metrics like CER/WER/BWER, lower is better
            # So if our model has lower error (os_llm_cer_diff > 0), that's an improvement
            os_llm_cer_direction = "increase" if os_llm_cer_diff > 0 else "decrease"
            os_llm_wer_direction = "increase" if os_llm_wer_diff > 0 else "decrease"
            os_llm_bwer_direction = "increase" if os_llm_bwer_diff > 0 else "decrease"

            # Format percentage with sign (positive means improvement)
            os_llm_cer_caption = f"{os_llm_cer_pct:+.1f}% vs Transkribus"
            os_llm_wer_caption = f"{os_llm_wer_pct:+.1f}% vs Transkribus"
            os_llm_bwer_caption = f"{os_llm_bwer_pct:+.1f}% vs Transkribus"

            # Create stat components row for this provider
            one_shot_llm_final_provider_stats.append(mo.md(f"### {os_llm_provider_name.upper()}"))
            one_shot_llm_final_provider_stats.append(
                mo.hstack([
                    mo.stat(
                        f"{os_llm_metrics['avg_cer']:.4f}",
                        label="Average CER",
                        caption=os_llm_cer_caption,
                        direction=os_llm_cer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{os_llm_metrics['avg_wer']:.4f}",
                        label="Average WER",
                        caption=os_llm_wer_caption,
                        direction=os_llm_wer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{os_llm_metrics['avg_bwer']:.4f}",
                        label="Average BWER",
                        caption=os_llm_bwer_caption,
                        direction=os_llm_bwer_direction,
                        bordered=True
                    ),
                    mo.stat(os_llm_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )
        else:
            # If Transkribus metrics aren't available, show regular stats
            one_shot_llm_final_provider_stats.append(mo.md(f"### {os_llm_provider_name.upper()}"))
            one_shot_llm_final_provider_stats.append(
                mo.hstack([
                    mo.stat(f"{os_llm_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{os_llm_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{os_llm_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                    mo.stat(os_llm_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )

    # Add Transkribus baseline stats at the top
    if not transkribus_df.empty:
        one_shot_llm_transkribus_stats = [
            mo.md("### TRANSKRIBUS with Text Titan 1 (Baseline)"),
            mo.hstack([
                mo.stat(f"{one_shot_llm_transkribus_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{one_shot_llm_transkribus_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{one_shot_llm_transkribus_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                mo.stat(len(transkribus_df), label="Documents", bordered=True)
            ])
        ]
    else:
        one_shot_llm_transkribus_stats = [mo.md("### Transkribus baseline not available")]

    # Combine everything into a vstack layout
    mo.vstack([
        mo.md("## One-Shot LLM Model Performance Metrics"),
        *one_shot_llm_transkribus_stats,
        mo.md("### MM-LLM One-Shot Models"),
        *one_shot_llm_final_provider_stats,
        mo.md("## Comparison Table"),
        mo.ui.table(one_shot_llm_final_comparison_df)
    ])
    return (
        one_shot_llm_final_comparison_data,
        one_shot_llm_final_comparison_df,
        one_shot_llm_final_provider_stats,
        one_shot_llm_transkribus_metrics,
        one_shot_llm_transkribus_stats,
        os_llm_bwer_caption,
        os_llm_bwer_diff,
        os_llm_bwer_direction,
        os_llm_bwer_pct,
        os_llm_cer_caption,
        os_llm_cer_diff,
        os_llm_cer_direction,
        os_llm_cer_pct,
        os_llm_df,
        os_llm_metrics,
        os_llm_model_name,
        os_llm_provider_name,
        os_llm_wer_caption,
        os_llm_wer_diff,
        os_llm_wer_direction,
        os_llm_wer_pct,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Hybrid Evaluation: MM-LLM One-Shot + Transkribus""")
    return


@app.cell(hide_code=True)
def _(mo):
    one_shot_hybrid_system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=8, value="""
    You are a specialized transcription model for medieval german printed text.
    Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
    preserving the original text without modernizing or correcting spelling.
    Important instructions:
    1. Use original medieval characters and spelling (ſ, æ, etc.)
    2. Preserve abbreviations and special characters
    3. Separate each line with a newline character (\\n)
    4. Do not add any explanations or commentary
    5. Do not include line numbers
    6. Transcribe text only, ignore images or decorative elements
    Your transcription should match the original manuscript as closely as possible.

    CRITICAL LINE BREAK INSTRUCTIONS:
    - You MUST maintain the EXACT same number of lines as in the original manuscript
    - Each physical line in the manuscript should be ONE line in your transcription
    - DO NOT merge short lines together
    - DO NOT split long lines into multiple lines
    - Preserve the exact same line structure as the manuscript""")

    # Create a run button for one-shot hybrid
    one_shot_hybrid_run_button = mo.ui.run_button(
        label="Run One-Shot Hybrid Evaluation",
        kind="success",
        tooltip="Start the one-shot hybrid evaluation process"
    )

    # Display configuration UI
    mo.vstack([
        mo.md("### Configure the One-Shot Hybrid evaluation parameters and click the button to start."),
        mo.vstack([
            one_shot_hybrid_system_prompt,
            one_shot_hybrid_run_button
        ])
    ])
    return one_shot_hybrid_run_button, one_shot_hybrid_system_prompt


@app.cell
def _(
    calculate_aggregate_metrics,
    encode_image,
    extract_text_from_xml,
    find_file_for_id,
    mo,
    one_shot_hybrid_run_button,
    one_shot_hybrid_system_prompt,
    os,
    pd,
    process_document,
    provider_models,
    time,
):
    def get_example_content():
        """Get content of example page for one-shot learning"""
        example_gt_path = 'data/reichenau_10_test/few-shot-samples/7474192.xml'  # Path to example ground truth
        example_img_path = 'data/reichenau_10_test/few-shot-samples/7474192.jpg'  # Path to example image

        if os.path.exists(example_gt_path):
            example_lines = extract_text_from_xml(example_gt_path)
            example_text = "\n".join(example_lines) if example_lines else ""

            return {
                "text": example_text,
                "image_path": example_img_path if os.path.exists(example_img_path) else None,
                "success": True
            }

    if one_shot_hybrid_run_button.value:
        # Display evaluation in progress
        print("## One-Shot Hybrid Evaluation in Progress (MM-LLM + Transkribus)")

        # Get example content for one-shot learning
        one_shot_example = get_example_content()
        if one_shot_example["success"]:
            print(f"✅ Using example from dedicated example folder for one-shot learning")
        else:
            print("⚠️ Using fallback example for one-shot learning - results may be suboptimal")

        # Find available images first
        one_shot_image_dir = 'data/reichenau_10_test/images'
        one_shot_available_images = []
        for one_shot_f in os.listdir(one_shot_image_dir):
            if one_shot_f.endswith('.jpg'):
                one_shot_image_id = os.path.splitext(one_shot_f)[0]
                one_shot_available_images.append(one_shot_image_id)

        print(f"Found {len(one_shot_available_images)} images to process")

        # Get matching ground truth files
        one_shot_gt_dir = 'data/reichenau_10_test/ground_truth'
        one_shot_transkribus_dir = 'results/linear_transcription/reichenau_inkunabeln/transkribus_10_test'
        one_shot_gt_files = {}
        for one_shot_image_id in one_shot_available_images:
            one_shot_gt_path = os.path.join(one_shot_gt_dir, f"{one_shot_image_id}.xml")
            if os.path.exists(one_shot_gt_path):
                one_shot_gt_files[one_shot_image_id] = one_shot_gt_path

        print(f"Found {len(one_shot_gt_files)} matching ground truth files")

        # Store results
        one_shot_results = {}
        one_shot_comparison_data = {}

        # Process each provider
        for one_shot_provider, one_shot_model_name in provider_models.items():
            print(f"### Evaluating One-Shot Hybrid {one_shot_provider.upper()} + Transkribus with {one_shot_model_name}")

            # Create output directory
            one_shot_output_dir = f'bentham_temp/one_shot_hybrid/{one_shot_provider}'
            os.makedirs(one_shot_output_dir, exist_ok=True)

            # Process all documents for this provider
            one_shot_all_results = []

            # Create document list for progress bar
            one_shot_documents = list(one_shot_gt_files.items())

            # Process each document with progress bar
            for one_shot_doc_idx in mo.status.progress_bar(
                range(len(one_shot_documents)),
                title=f"Processing {one_shot_provider} documents",
                subtitle=f"Model: {one_shot_model_name}"
            ):
                one_shot_doc_id, one_shot_gt_path = one_shot_documents[one_shot_doc_idx]

                # Find Transkribus transcription file
                one_shot_transkribus_path = find_file_for_id(one_shot_doc_id, one_shot_transkribus_dir, ['.xml'])

                if one_shot_transkribus_path:
                    # Extract text from Transkribus file
                    one_shot_transkribus_lines = extract_text_from_xml(one_shot_transkribus_path)
                    one_shot_transkribus_text = "\n".join(one_shot_transkribus_lines) if one_shot_transkribus_lines else ""

                    # Encode image
                    one_shot_image_path = os.path.join('data/reichenau_10_test/images', f"{one_shot_doc_id}.jpg")
                    one_shot_image_base64 = encode_image(one_shot_image_path)

                    # Create custom messages with example transcription AND Transkribus transcription
                    one_shot_custom_messages = [
                        {"role": "system", "content": one_shot_hybrid_system_prompt.value},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Here is an example of a historical manuscript page and its correct transcription:"},
                                {"type": "text", "text": f"Example transcription:\n{one_shot_example['text']}"}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Please transcribe this new historical manuscript image accurately, preserving the line breaks exactly as they appear. The following is the output of a traditional OCR model from Transkribus. It is fine-tuned on medieval texts. It can help you transcribe the page, but may also contain errors:\n\n{one_shot_transkribus_text}"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{one_shot_image_base64}"}
                                }
                            ]
                        }
                    ]

                    # Process the document with custom messages
                    one_shot_result = process_document(
                        provider=f"{one_shot_provider}",
                        model_name=one_shot_model_name,
                        doc_id=one_shot_doc_id,
                        gt_path=one_shot_gt_path,
                        output_dir=one_shot_output_dir,
                        system_prompt=one_shot_hybrid_system_prompt.value,
                        custom_messages=one_shot_custom_messages
                    )
                else:
                    # No Transkribus file found
                    one_shot_result = {
                        "provider": f"{one_shot_provider}",
                        "doc_id": one_shot_doc_id,
                        "status": "error",
                        "message": "No Transkribus transcription found"
                    }

                # Display result
                if one_shot_result['status'] == 'success':
                    print(f"✅ **{one_shot_result['doc_id']}**: {one_shot_result['message']}")
                    one_shot_all_results.append(one_shot_result['metrics'])
                else:
                    print(f"⚠️ **{one_shot_result['doc_id']}**: {one_shot_result['message']}")

                # Add delay to avoid rate limits
                time.sleep(0.5)

            # Compile all results for this provider
            if one_shot_all_results:
                one_shot_results_df = pd.DataFrame(one_shot_all_results)
                one_shot_results_df.to_csv(os.path.join(one_shot_output_dir, f"{one_shot_provider}_one_shot_results.csv"), index=False)

                # Calculate aggregate metrics
                calculate_aggregate_metrics(one_shot_output_dir, f"{one_shot_provider}")

                # Store for comparison
                one_shot_results[f"{one_shot_provider}"] = one_shot_results_df

                # Add to comparison data
                one_shot_comparison_data[f"{one_shot_provider}"] = {
                    "model": f"{provider_models[one_shot_provider]} + Transkribus (One-Shot)",
                    "avg_cer": one_shot_results_df['cer'].mean(),
                    "avg_wer": one_shot_results_df['wer'].mean(),
                    "avg_bwer": one_shot_results_df['bwer'].mean(),
                    "doc_count": len(one_shot_results_df)
                }
    return (
        get_example_content,
        one_shot_all_results,
        one_shot_available_images,
        one_shot_comparison_data,
        one_shot_custom_messages,
        one_shot_doc_id,
        one_shot_doc_idx,
        one_shot_documents,
        one_shot_example,
        one_shot_f,
        one_shot_gt_dir,
        one_shot_gt_files,
        one_shot_gt_path,
        one_shot_image_base64,
        one_shot_image_dir,
        one_shot_image_id,
        one_shot_image_path,
        one_shot_model_name,
        one_shot_output_dir,
        one_shot_provider,
        one_shot_result,
        one_shot_results,
        one_shot_results_df,
        one_shot_transkribus_dir,
        one_shot_transkribus_lines,
        one_shot_transkribus_path,
        one_shot_transkribus_text,
    )


@app.cell
def _(one_shot_results):
    one_shot_results
    return


@app.cell(hide_code=True)
def _(mo, one_shot_results, pd, provider_models, transkribus_df):
    one_shot_transkribus_metrics = {
        "avg_cer": transkribus_df['cer'].mean() if not transkribus_df.empty else None,
        "avg_wer": transkribus_df['wer'].mean() if not transkribus_df.empty else None,
        "avg_bwer": transkribus_df['bwer'].mean() if not transkribus_df.empty else None
    }

    # Create fresh comparison data in this cell
    one_shot_final_comparison_data = {}
    for os_provider_name, os_df in one_shot_results.items():
        if not os_df.empty:
            one_shot_final_comparison_data[os_provider_name] = {
                "model": f"{provider_models[os_provider_name.replace('_one_shot', '')]} + Transkribus (One-Shot)",
                "avg_cer": os_df['cer'].mean(),
                "avg_wer": os_df['wer'].mean(),
                "avg_bwer": os_df['bwer'].mean(),
                "doc_count": len(os_df)
            }

    one_shot_final_comparison_df = pd.DataFrame(one_shot_final_comparison_data).T

    # Add Transkribus for comparison
    if not transkribus_df.empty:
        one_shot_final_comparison_df.loc['transkribus'] = {
            "model": "Text Titan 1",
            "avg_cer": one_shot_transkribus_metrics["avg_cer"],
            "avg_wer": one_shot_transkribus_metrics["avg_wer"],
            "avg_bwer": one_shot_transkribus_metrics["avg_bwer"],
            "doc_count": len(transkribus_df)
        }

    # Create fresh stat components for each provider with Transkribus comparison
    one_shot_final_provider_stats = []
    for os_provider_name, os_metrics in one_shot_final_comparison_data.items():
        os_model_name = os_metrics["model"]

        # Skip Transkribus in this section
        if os_provider_name == 'transkribus':
            continue

        # Compare with Transkribus metrics and determine direction
        if one_shot_transkribus_metrics["avg_cer"] is not None:
            # Calculate absolute differences
            os_cer_diff = one_shot_transkribus_metrics["avg_cer"] - os_metrics["avg_cer"]
            os_wer_diff = one_shot_transkribus_metrics["avg_wer"] - os_metrics["avg_wer"]
            os_bwer_diff = one_shot_transkribus_metrics["avg_bwer"] - os_metrics["avg_bwer"]

            # Calculate percentage changes (relative to Transkribus baseline)
            os_cer_pct = (os_cer_diff / one_shot_transkribus_metrics["avg_cer"]) * 100 if one_shot_transkribus_metrics["avg_cer"] > 0 else 0
            os_wer_pct = (os_wer_diff / one_shot_transkribus_metrics["avg_wer"]) * 100 if one_shot_transkribus_metrics["avg_wer"] > 0 else 0
            os_bwer_pct = (os_bwer_diff / one_shot_transkribus_metrics["avg_bwer"]) * 100 if one_shot_transkribus_metrics["avg_bwer"] > 0 else 0

            # For error metrics like CER/WER/BWER, lower is better
            # So if our model has lower error (os_cer_diff > 0), that's an improvement
            os_cer_direction = "increase" if os_cer_diff > 0 else "decrease"
            os_wer_direction = "increase" if os_wer_diff > 0 else "decrease"
            os_bwer_direction = "increase" if os_bwer_diff > 0 else "decrease"

            # Format percentage with sign (negative means reduction in error, which is good)
            os_cer_caption = f"{os_cer_pct:+.1f}% vs Transkribus"
            os_wer_caption = f"{os_wer_pct:+.1f}% vs Transkribus"
            os_bwer_caption = f"{os_bwer_pct:+.1f}% vs Transkribus"

            # Create stat components row for this provider
            one_shot_final_provider_stats.append(mo.md(f"### {os_provider_name.upper()}"))
            one_shot_final_provider_stats.append(
                mo.hstack([
                    mo.stat(
                        f"{os_metrics['avg_cer']:.4f}",
                        label="Average CER",
                        caption=os_cer_caption,
                        direction=os_cer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{os_metrics['avg_wer']:.4f}",
                        label="Average WER",
                        caption=os_wer_caption,
                        direction=os_wer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{os_metrics['avg_bwer']:.4f}",
                        label="Average BWER",
                        caption=os_bwer_caption,
                        direction=os_bwer_direction,
                        bordered=True
                    ),
                    mo.stat(os_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )
        else:
            # If Transkribus metrics aren't available, show regular stats
            one_shot_final_provider_stats.append(mo.md(f"### {os_provider_name.upper()}"))
            one_shot_final_provider_stats.append(
                mo.hstack([
                    mo.stat(f"{os_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{os_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{os_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                    mo.stat(os_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )

    # Add Transkribus baseline stats at the top
    if not transkribus_df.empty:
        one_shot_transkribus_stats = [
            mo.md("### TRANSKRIBUS with Text Titan 1 (Baseline)"),
            mo.hstack([
                mo.stat(f"{one_shot_transkribus_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{one_shot_transkribus_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{one_shot_transkribus_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                mo.stat(len(transkribus_df), label="Documents", bordered=True)
            ])
        ]
    else:
        one_shot_transkribus_stats = [mo.md("### Transkribus baseline not available")]

    # Combine everything into a vstack layout
    mo.vstack([
        mo.md("## One-Shot Hybrid Model Performance Metrics"),
        *one_shot_transkribus_stats,
        mo.md("### MM-LLM + Transkribus One-Shot Hybrid Models"),
        *one_shot_final_provider_stats,
        mo.md("## Comparison Table"),
        mo.ui.table(one_shot_final_comparison_df)
    ])
    return (
        one_shot_final_comparison_data,
        one_shot_final_comparison_df,
        one_shot_final_provider_stats,
        one_shot_transkribus_metrics,
        one_shot_transkribus_stats,
        os_bwer_caption,
        os_bwer_diff,
        os_bwer_direction,
        os_bwer_pct,
        os_cer_caption,
        os_cer_diff,
        os_cer_direction,
        os_cer_pct,
        os_df,
        os_metrics,
        os_model_name,
        os_provider_name,
        os_wer_caption,
        os_wer_diff,
        os_wer_direction,
        os_wer_pct,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Two-Shot Evaluation for MM-LLMs""")
    return


@app.cell(hide_code=True)
def _(mo):
    two_shot_llm_system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=8, value="""
    You are a specialized transcription model for medieval german printed text.
    Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
    preserving the original text without modernizing or correcting spelling.
    Important instructions:
    1. Use original medieval characters and spelling (ſ, æ, etc.)
    2. Preserve abbreviations and special characters
    3. Separate each line with a newline character (\\n)
    4. Do not add any explanations or commentary
    5. Do not include line numbers
    6. Transcribe text only, ignore images or decorative elements
    Your transcription should match the original manuscript as closely as possible.

    CRITICAL LINE BREAK INSTRUCTIONS:
    - You MUST maintain the EXACT same number of lines as in the original manuscript
    - Each physical line in the manuscript should be ONE line in your transcription
    - DO NOT merge short lines together
    - DO NOT split long lines into multiple lines
    - Preserve the exact same line structure as the manuscript""")

    # Create a run button for two-shot LLM
    two_shot_llm_run_button = mo.ui.run_button(
        label="Run Two-Shot LLM Evaluation",
        kind="success",
        tooltip="Start the two-shot LLM evaluation process"
    )

    # Display configuration UI
    mo.vstack([
        mo.md("### Configure the Two-Shot LLM evaluation parameters and click the button to start."),
        mo.vstack([
            two_shot_llm_system_prompt,
            two_shot_llm_run_button
        ])
    ])
    return two_shot_llm_run_button, two_shot_llm_system_prompt


@app.cell
def _(
    calculate_aggregate_metrics,
    encode_image,
    extract_text_from_xml,
    mo,
    os,
    pd,
    process_document,
    provider_models,
    time,
    two_shot_llm_run_button,
    two_shot_llm_system_prompt,
):
    def get_two_shot_llm_examples():
        """Get content of two example pages for two-shot learning"""
        # Example 1
        example1_gt_path = 'data/reichenau_10_test/few-shot-samples/7474192.xml'  # Path to first example ground truth
        example1_img_path = 'data/reichenau_10_test/few-shot-samples/7474192.jpg'  # Path to first example image
        example1_text = ""
        if os.path.exists(example1_gt_path):
            example1_lines = extract_text_from_xml(example1_gt_path)
            example1_text = "\n".join(example1_lines) if example1_lines else ""

        # Example 2
        example2_gt_path = 'data/reichenau_10_test/few-shot-samples/7764258.xml'  # Path to second example ground truth
        example2_img_path = 'data/reichenau_10_test/few-shot-samples/7764258.jpg'  # Path to second example image
        example2_text = ""
        if os.path.exists(example2_gt_path):
            example2_lines = extract_text_from_xml(example2_gt_path)
            example2_text = "\n".join(example2_lines) if example2_lines else ""

        # Check if both examples were successfully loaded
        success = bool(example1_text and example2_text)

        return {
            "example1_text": example1_text,
            "example1_img_path": example1_img_path if os.path.exists(example1_img_path) else None,
            "example2_text": example2_text,
            "example2_img_path": example2_img_path if os.path.exists(example2_img_path) else None,
            "success": success
        }

    if two_shot_llm_run_button.value:
        # Display evaluation in progress
        mo.md("## Two-Shot LLM Evaluation in Progress (MM-LLM Only)")

        # Get example content for two-shot learning
        two_shot_llm_examples = get_two_shot_llm_examples()
        if two_shot_llm_examples["success"]:
            mo.md(f"✅ Using two examples from dedicated example folder for two-shot learning")
        else:
            mo.md("⚠️ Warning: One or both examples for two-shot learning couldn't be loaded")

        # Find available images first
        two_shot_llm_image_dir = 'data/reichenau_10_test/images'
        two_shot_llm_available_images = []
        for two_shot_llm_f in os.listdir(two_shot_llm_image_dir):
            if two_shot_llm_f.endswith('.jpg'):
                two_shot_llm_image_id = os.path.splitext(two_shot_llm_f)[0]
                two_shot_llm_available_images.append(two_shot_llm_image_id)

        mo.md(f"Found {len(two_shot_llm_available_images)} images to process")

        # Get matching ground truth files
        two_shot_llm_gt_dir = 'data/reichenau_10_test/ground_truth'
        two_shot_llm_gt_files = {}
        for two_shot_llm_image_id in two_shot_llm_available_images:
            two_shot_llm_gt_path = os.path.join(two_shot_llm_gt_dir, f"{two_shot_llm_image_id}.xml")
            if os.path.exists(two_shot_llm_gt_path):
                two_shot_llm_gt_files[two_shot_llm_image_id] = two_shot_llm_gt_path

        mo.md(f"Found {len(two_shot_llm_gt_files)} matching ground truth files")

        # Store results
        two_shot_llm_results = {}
        two_shot_llm_comparison_data = {}

        # Process each provider
        for two_shot_llm_provider, two_shot_llm_model_name in provider_models.items():
            mo.md(f"### Evaluating Two-Shot LLM {two_shot_llm_provider.upper()} with {two_shot_llm_model_name}")

            # Create output directory
            two_shot_llm_output_dir = f'bentham_temp/two_shot_llm/{two_shot_llm_provider}'
            os.makedirs(two_shot_llm_output_dir, exist_ok=True)

            # Process all documents for this provider
            two_shot_llm_all_results = []

            # Create document list for progress bar
            two_shot_llm_documents = list(two_shot_llm_gt_files.items())

            # Process each document with progress bar
            for two_shot_llm_doc_idx in mo.status.progress_bar(
                range(len(two_shot_llm_documents)),
                title=f"Processing {two_shot_llm_provider} documents",
                subtitle=f"Model: {two_shot_llm_model_name}"
            ):
                two_shot_llm_doc_id, two_shot_llm_gt_path = two_shot_llm_documents[two_shot_llm_doc_idx]

                # Encode image
                two_shot_llm_image_path = os.path.join('data/reichenau_10_test/images', f"{two_shot_llm_doc_id}.jpg")
                two_shot_llm_image_base64 = encode_image(two_shot_llm_image_path)

                # Create custom messages with TWO example transcriptions (but NO Transkribus transcription)
                two_shot_llm_custom_messages = [
                    {"role": "system", "content": two_shot_llm_system_prompt.value},
                    # First example
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is an example of a historical manuscript page and its correct transcription:"},
                        ]
                    }
                ]

                # Add first example image if available
                if two_shot_llm_examples["example1_img_path"] and os.path.exists(two_shot_llm_examples["example1_img_path"]):
                    example1_image_base64 = encode_image(two_shot_llm_examples["example1_img_path"])
                    two_shot_llm_custom_messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{example1_image_base64}"}
                    })

                # Add first example transcription text
                two_shot_llm_custom_messages[1]["content"].append(
                    {"type": "text", "text": f"Example 1 transcription:\n{two_shot_llm_examples['example1_text']}"}
                )



                # Add second example
                second_example_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is another example of a historical manuscript page and its correct transcription:"},
                    ]
                }

                # Add second example image if available
                if two_shot_llm_examples["example2_img_path"] and os.path.exists(two_shot_llm_examples["example2_img_path"]):
                    example2_image_base64 = encode_image(two_shot_llm_examples["example2_img_path"])
                    second_example_message["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{example2_image_base64}"}
                    })

                # Add second example transcription text
                second_example_message["content"].append(
                    {"type": "text", "text": f"Example 2 transcription:\n{two_shot_llm_examples['example2_text']}"}
                )

                # Add second example to messages
                two_shot_llm_custom_messages.append(second_example_message)


                # Add actual document to transcribe
                two_shot_llm_custom_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please transcribe this new historical manuscript image accurately, preserving the line breaks exactly as they appear."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{two_shot_llm_image_base64}"}
                        }
                    ]
                })

                # Process the document with custom messages
                two_shot_llm_result = process_document(
                    provider=f"{two_shot_llm_provider}",
                    model_name=two_shot_llm_model_name,
                    doc_id=two_shot_llm_doc_id,
                    gt_path=two_shot_llm_gt_path,
                    output_dir=two_shot_llm_output_dir,
                    system_prompt=two_shot_llm_system_prompt.value,
                    custom_messages=two_shot_llm_custom_messages
                )

                # Display result
                if two_shot_llm_result['status'] == 'success':
                    mo.md(f"✅ **{two_shot_llm_result['doc_id']}**: {two_shot_llm_result['message']}")
                    two_shot_llm_all_results.append(two_shot_llm_result['metrics'])
                else:
                    mo.md(f"⚠️ **{two_shot_llm_result['doc_id']}**: {two_shot_llm_result['message']}")

                # Add delay to avoid rate limits
                time.sleep(0.5)

            # Compile all results for this provider
            if two_shot_llm_all_results:
                two_shot_llm_results_df = pd.DataFrame(two_shot_llm_all_results)
                two_shot_llm_results_df.to_csv(os.path.join(two_shot_llm_output_dir, f"{two_shot_llm_provider}_two_shot_llm_results.csv"), index=False)

                # Calculate aggregate metrics
                calculate_aggregate_metrics(two_shot_llm_output_dir, f"{two_shot_llm_provider}")

                # Store for comparison
                two_shot_llm_results[f"{two_shot_llm_provider}"] = two_shot_llm_results_df

                # Add to comparison data
                two_shot_llm_comparison_data[f"{two_shot_llm_provider}"] = {
                    "model": f"{provider_models[two_shot_llm_provider]} (Two-Shot)",
                    "avg_cer": two_shot_llm_results_df['cer'].mean(),
                    "avg_wer": two_shot_llm_results_df['wer'].mean(),
                    "avg_bwer": two_shot_llm_results_df['bwer'].mean(),
                    "doc_count": len(two_shot_llm_results_df)
                }
    return (
        example1_image_base64,
        example2_image_base64,
        get_two_shot_llm_examples,
        second_example_message,
        two_shot_llm_all_results,
        two_shot_llm_available_images,
        two_shot_llm_comparison_data,
        two_shot_llm_custom_messages,
        two_shot_llm_doc_id,
        two_shot_llm_doc_idx,
        two_shot_llm_documents,
        two_shot_llm_examples,
        two_shot_llm_f,
        two_shot_llm_gt_dir,
        two_shot_llm_gt_files,
        two_shot_llm_gt_path,
        two_shot_llm_image_base64,
        two_shot_llm_image_dir,
        two_shot_llm_image_id,
        two_shot_llm_image_path,
        two_shot_llm_model_name,
        two_shot_llm_output_dir,
        two_shot_llm_provider,
        two_shot_llm_result,
        two_shot_llm_results,
        two_shot_llm_results_df,
    )


@app.cell
def _(two_shot_llm_results):
    two_shot_llm_results
    return


@app.cell
def _(mo, pd, provider_models, transkribus_df, two_shot_llm_results):
    two_shot_llm_transkribus_metrics = {
        "avg_cer": transkribus_df['cer'].mean() if not transkribus_df.empty else None,
        "avg_wer": transkribus_df['wer'].mean() if not transkribus_df.empty else None,
        "avg_bwer": transkribus_df['bwer'].mean() if not transkribus_df.empty else None
    }

    # Create fresh comparison data in this cell
    two_shot_llm_final_comparison_data = {}
    for ts_llm_provider_name, ts_llm_df in two_shot_llm_results.items():
        if not ts_llm_df.empty:
            two_shot_llm_final_comparison_data[ts_llm_provider_name] = {
                "model": f"{provider_models[ts_llm_provider_name.replace('_two_shot_llm', '')]} (Two-Shot)",
                "avg_cer": ts_llm_df['cer'].mean(),
                "avg_wer": ts_llm_df['wer'].mean(),
                "avg_bwer": ts_llm_df['bwer'].mean(),
                "doc_count": len(ts_llm_df)
            }

    two_shot_llm_final_comparison_df = pd.DataFrame(two_shot_llm_final_comparison_data).T

    # Add Transkribus for comparison
    if not transkribus_df.empty:
        two_shot_llm_final_comparison_df.loc['transkribus'] = {
            "model": "Text Titan 1",
            "avg_cer": two_shot_llm_transkribus_metrics["avg_cer"],
            "avg_wer": two_shot_llm_transkribus_metrics["avg_wer"],
            "avg_bwer": two_shot_llm_transkribus_metrics["avg_bwer"],
            "doc_count": len(transkribus_df)
        }

    # Create fresh stat components for each provider with Transkribus comparison
    two_shot_llm_final_provider_stats = []
    for ts_llm_provider_name, ts_llm_metrics in two_shot_llm_final_comparison_data.items():
        ts_llm_model_name = ts_llm_metrics["model"]

        # Skip Transkribus in this section
        if ts_llm_provider_name == 'transkribus':
            continue

        # Compare with Transkribus metrics and determine direction
        if two_shot_llm_transkribus_metrics["avg_cer"] is not None:
            # Calculate absolute differences
            ts_llm_cer_diff = two_shot_llm_transkribus_metrics["avg_cer"] - ts_llm_metrics["avg_cer"]
            ts_llm_wer_diff = two_shot_llm_transkribus_metrics["avg_wer"] - ts_llm_metrics["avg_wer"]
            ts_llm_bwer_diff = two_shot_llm_transkribus_metrics["avg_bwer"] - ts_llm_metrics["avg_bwer"]

            # Calculate percentage changes (relative to Transkribus baseline)
            ts_llm_cer_pct = (ts_llm_cer_diff / two_shot_llm_transkribus_metrics["avg_cer"]) * 100 if two_shot_llm_transkribus_metrics["avg_cer"] > 0 else 0
            ts_llm_wer_pct = (ts_llm_wer_diff / two_shot_llm_transkribus_metrics["avg_wer"]) * 100 if two_shot_llm_transkribus_metrics["avg_wer"] > 0 else 0
            ts_llm_bwer_pct = (ts_llm_bwer_diff / two_shot_llm_transkribus_metrics["avg_bwer"]) * 100 if two_shot_llm_transkribus_metrics["avg_bwer"] > 0 else 0

            # For error metrics like CER/WER/BWER, lower is better
            # So if our model has lower error (ts_llm_cer_diff > 0), that's an improvement
            ts_llm_cer_direction = "increase" if ts_llm_cer_diff > 0 else "decrease"
            ts_llm_wer_direction = "increase" if ts_llm_wer_diff > 0 else "decrease"
            ts_llm_bwer_direction = "increase" if ts_llm_bwer_diff > 0 else "decrease"

            # Format percentage with sign (positive means improvement)
            ts_llm_cer_caption = f"{ts_llm_cer_pct:+.1f}% vs Transkribus"
            ts_llm_wer_caption = f"{ts_llm_wer_pct:+.1f}% vs Transkribus"
            ts_llm_bwer_caption = f"{ts_llm_bwer_pct:+.1f}% vs Transkribus"

            # Create stat components row for this provider
            two_shot_llm_final_provider_stats.append(mo.md(f"### {ts_llm_provider_name.upper()}"))
            two_shot_llm_final_provider_stats.append(
                mo.hstack([
                    mo.stat(
                        f"{ts_llm_metrics['avg_cer']:.4f}",
                        label="Average CER",
                        caption=ts_llm_cer_caption,
                        direction=ts_llm_cer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{ts_llm_metrics['avg_wer']:.4f}",
                        label="Average WER",
                        caption=ts_llm_wer_caption,
                        direction=ts_llm_wer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{ts_llm_metrics['avg_bwer']:.4f}",
                        label="Average BWER",
                        caption=ts_llm_bwer_caption,
                        direction=ts_llm_bwer_direction,
                        bordered=True
                    ),
                    mo.stat(ts_llm_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )
        else:
            # If Transkribus metrics aren't available, show regular stats
            two_shot_llm_final_provider_stats.append(mo.md(f"### {ts_llm_provider_name.upper()}"))
            two_shot_llm_final_provider_stats.append(
                mo.hstack([
                    mo.stat(f"{ts_llm_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{ts_llm_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{ts_llm_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                    mo.stat(ts_llm_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )

    # Add Transkribus baseline stats at the top
    if not transkribus_df.empty:
        two_shot_llm_transkribus_stats = [
            mo.md("### TRANSKRIBUS with Text Titan 1 (Baseline)"),
            mo.hstack([
                mo.stat(f"{two_shot_llm_transkribus_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{two_shot_llm_transkribus_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{two_shot_llm_transkribus_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                mo.stat(len(transkribus_df), label="Documents", bordered=True)
            ])
        ]
    else:
        two_shot_llm_transkribus_stats = [mo.md("### Transkribus baseline not available")]

    # Combine everything into a vstack layout
    mo.vstack([
        mo.md("## Two-Shot LLM Model Performance Metrics"),
        *two_shot_llm_transkribus_stats,
        mo.md("### MM-LLM Two-Shot Models"),
        *two_shot_llm_final_provider_stats,
        mo.md("## Comparison Table"),
        mo.ui.table(two_shot_llm_final_comparison_df)
    ])
    return (
        ts_llm_bwer_caption,
        ts_llm_bwer_diff,
        ts_llm_bwer_direction,
        ts_llm_bwer_pct,
        ts_llm_cer_caption,
        ts_llm_cer_diff,
        ts_llm_cer_direction,
        ts_llm_cer_pct,
        ts_llm_df,
        ts_llm_metrics,
        ts_llm_model_name,
        ts_llm_provider_name,
        ts_llm_wer_caption,
        ts_llm_wer_diff,
        ts_llm_wer_direction,
        ts_llm_wer_pct,
        two_shot_llm_final_comparison_data,
        two_shot_llm_final_comparison_df,
        two_shot_llm_final_provider_stats,
        two_shot_llm_transkribus_metrics,
        two_shot_llm_transkribus_stats,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Hybrid Evaluation: MM-LLM Two-Shot + Transkribus""")
    return


@app.cell
def _(mo):
    two_shot_hybrid_system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=8, value="""
    You are a specialized transcription model for medieval german printed text.
    Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
    preserving the original text without modernizing or correcting spelling.
    Important instructions:
    1. Use original medieval characters and spelling (ſ, æ, etc.)
    2. Preserve abbreviations and special characters
    3. Separate each line with a newline character (\\n)
    4. Do not add any explanations or commentary
    5. Do not include line numbers
    6. Transcribe text only, ignore images or decorative elements
    Your transcription should match the original manuscript as closely as possible.

    CRITICAL LINE BREAK INSTRUCTIONS:
    - You MUST maintain the EXACT same number of lines as in the original manuscript
    - Each physical line in the manuscript should be ONE line in your transcription
    - DO NOT merge short lines together
    - DO NOT split long lines into multiple lines
    - Preserve the exact same line structure as the manuscript""")

    # Create a run button for two-shot hybrid
    two_shot_hybrid_run_button = mo.ui.run_button(
        label="Run Two-Shot Hybrid Evaluation",
        kind="success",
        tooltip="Start the two-shot hybrid evaluation process"
    )

    # Display configuration UI
    mo.vstack([
        mo.md("### Configure the Two-Shot Hybrid evaluation parameters and click the button to start."),
        mo.vstack([
            two_shot_hybrid_system_prompt,
            two_shot_hybrid_run_button
        ])
    ])
    return two_shot_hybrid_run_button, two_shot_hybrid_system_prompt


app._unparsable_cell(
    r"""
    def get_two_example_content():
        \"\"\"Get content of two example pages for two-shot learning\"\"\"
        # Example 1
        example1_gt_path = 'data/reichenau_10_test/few-shot-samples/7474192.xml'  # Path to first example ground truth
        example1_img_path = 'data/reichenau_10_test/few-shot-samples/7474192.jpg'  # Path to first example image
        example1_text = \"\"

        if os.path.exists(example1_gt_path):
            example1_lines = extract_text_from_xml(example1_gt_path)
            example1_text = \"\n\".join(example1_lines) if example1_lines else \"\"

        # Example 2
        example2_gt_path = 'data/reichenau_10_test/few-shot-samples/7764258.xml'  # Path to second example ground truth
        example2_img_path = 'data/reichenau_10_test/few-shot-samples/7764258.jpg'  # Path to second example image
        example2_text = \"\"

        if os.path.exists(example2_gt_path):
            example2_lines = extract_text_from_xml(example2_gt_path)
            example2_text = \"\n\".join(example2_lines) if example2_lines else \"\"

        # Check if both examples were successfully loaded
        success = bool(example1_text and example2_text)

        return {
            \"example1_text\": example1_text,
            \"example1_image_path\": example1_img_path if os.path.exists(example1_img_path) else None,
            \"example2_text\": example2_text,
            \"example2_image_path\": example2_img_path if os.path.exists(example2_img_path) else None,
            \"success\": success
        }

    if two_shot_hybrid_run_button.value:
        # Display evaluation in progress
        print(\"## Two-Shot Hybrid Evaluation in Progress (MM-LLM + Transkribus)\")

        # Get example content for two-shot learning
        two_shot_examples = get_two_example_content()
        if two_shot_examples[\"success\"]:
            print(f\"✅ Using two examples from dedicated example folder for two-shot learning\")
        else:
            print(\"⚠️ Warning: One or both examples for two-shot learning couldn't be loaded\")

        # Encode example images if available
        example1_image_base64 = None
        if two_shot_examples.get(\"example1_image_path\"):
            try:
                example1_image_base64 = encode_image(two_shot_examples[\"example1_image_path\"])
                print(\"✅ Successfully encoded example 1 image\")
            except Exception as e:
                print(f\"⚠️ Failed to encode example 1 image: {str(e)}\")

        example2_image_base64 = None
        if two_shot_examples.get(\"example2_image_path\"):
            try:
                example2_image_base64 = encode_image(two_shot_examples[\"example2_image_path\"])
                print(\"✅ Successfully encoded example 2 image\")
            except Exception as e:
               print(f\"⚠️ Failed to encode example 2 image: {str(e)}\")

        # Find available images first
        two_shot_image_dir = 'data/reichenau_10_test/images'
        two_shot_available_images = []
        for two_shot_f in os.listdir(two_shot_image_dir):
            if two_shot_f.endswith('.jpg'):
                two_shot_image_id = os.path.splitext(two_shot_f)[0]
                two_shot_available_images.append(two_shot_image_id)

        print(f\"Found {len(two_shot_available_images)} images to process\")

        # Get matching ground truth files
        two_shot_gt_dir = 'data/reichenau_10_test/ground_truth'
        two_shot_transkribus_dir = 'results/linear_transcription/reichenau_inkunabeln/transkribus_10_test'
        two_shot_gt_files = {}
        for two_shot_image_id in two_shot_available_images:
            two_shot_gt_path = os.path.join(two_shot_gt_dir, f\"{two_shot_image_id}.xml\")
            if os.path.exists(two_shot_gt_path):
                two_shot_gt_files[two_shot_image_id] = two_shot_gt_path

       print(f\"Found {len(two_shot_gt_files)} matching ground truth files\")

        # Store results
        two_shot_results = {}
        two_shot_comparison_data = {}

        # Process each provider
        for two_shot_provider, two_shot_model_name in provider_models.items():
            print(f\"### Evaluating Two-Shot Hybrid {two_shot_provider.upper()} + Transkribus with {two_shot_model_name}\")

            # Create output directory
            two_shot_output_dir = f'bentham_temp/two_shot_hybrid/{two_shot_provider}'
            os.makedirs(two_shot_output_dir, exist_ok=True)

            # Process all documents for this provider
            two_shot_all_results = []

            # Create document list for progress bar
            two_shot_documents = list(two_shot_gt_files.items())

            # Process each document with progress bar
            for two_shot_doc_idx in mo.status.progress_bar(
                range(len(two_shot_documents)),
                title=f\"Processing {two_shot_provider} documents\",
                subtitle=f\"Model: {two_shot_model_name}\"
            ):
                two_shot_doc_id, two_shot_gt_path = two_shot_documents[two_shot_doc_idx]

                # Find Transkribus transcription file
                two_shot_transkribus_path = find_file_for_id(two_shot_doc_id, two_shot_transkribus_dir, ['.xml'])

                if two_shot_transkribus_path:
                    # Extract text from Transkribus file
                    two_shot_transkribus_lines = extract_text_from_xml(two_shot_transkribus_path)
                    two_shot_transkribus_text = \"\n\".join(two_shot_transkribus_lines) if two_shot_transkribus_lines else \"\"

                    # Encode image
                    two_shot_image_path = os.path.join('data/reichenau_10_test/images', f\"{two_shot_doc_id}.jpg\")
                    two_shot_image_base64 = encode_image(two_shot_image_path)

                    # Prepare content for first example with image if available
                    example1_content = [
                        {\"type\": \"text\", \"text\": \"Here is an example of a historical manuscript page and its correct transcription:\"}
                    ]
                    if example1_image_base64:
                        example1_content.append({\"type\": \"image_url\", \"image_url\": {\"url\": f\"data:image/jpeg;base64,{example1_image_base64}\"}})
                    example1_content.append({\"type\": \"text\", \"text\": f\"Example 1 transcription:\n{two_shot_examples['example1_text']}\"})

                    # Prepare content for second example with image if available
                    example2_content = [
                        {\"type\": \"text\", \"text\": \"Here is another example of a historical manuscript page and its correct transcription:\"}
                    ]
                    if example2_image_base64:
                        example2_content.append({\"type\": \"image_url\", \"image_url\": {\"url\": f\"data:image/jpeg;base64,{example2_image_base64}\"}})
                    example2_content.append({\"type\": \"text\", \"text\": f\"Example 2 transcription:\n{two_shot_examples['example2_text']}\"})

                    # Create custom messages with TWO example transcriptions AND Transkribus transcription
                    two_shot_custom_messages = [
                        {\"role\": \"system\", \"content\": two_shot_hybrid_system_prompt.value},
                        # First example with image
                        {\"role\": \"user\", \"content\": example1_content},
                        # Second example with image
                        {\"role\": \"user\", \"content\": example2_content},
                        # Actual document to transcribe
                        {
                            \"role\": \"user\",
                            \"content\": [
                                {\"type\": \"text\", \"text\": f\"Please transcribe this new historical manuscript image accurately, preserving the line breaks exactly as they appear. The following is the output of a traditional OCR model from Transkribus. It is fine-tuned on medieval texts. It can help you transcribe the page, but may also contain errors:\n\n{two_shot_transkribus_text}\"},
                                {
                                    \"type\": \"image_url\",
                                    \"image_url\": {\"url\": f\"data:image/jpeg;base64,{two_shot_image_base64}\"}
                                }
                            ]
                        }
                    ]

                    # Process the document with custom messages
                    two_shot_result = process_document(
                        provider=f\"{two_shot_provider}\",
                        model_name=two_shot_model_name,
                        doc_id=two_shot_doc_id,
                        gt_path=two_shot_gt_path,
                        output_dir=two_shot_output_dir,
                        system_prompt=two_shot_hybrid_system_prompt.value,
                        custom_messages=two_shot_custom_messages
                    )
                else:
                    # No Transkribus file found
                    two_shot_result = {
                        \"provider\": f\"{two_shot_provider}\",
                        \"doc_id\": two_shot_doc_id,
                        \"status\": \"error\",
                        \"message\": \"No Transkribus transcription found\"
                    }

                # Display result
                if two_shot_result['status'] == 'success':
                   print(f\"✅ **{two_shot_result['doc_id']}**: {two_shot_result['message']}\")
                    two_shot_all_results.append(two_shot_result['metrics'])
                else:
                    print(f\"⚠️ **{two_shot_result['doc_id']}**: {two_shot_result['message']}\")

                # Add delay to avoid rate limits
                time.sleep(0.5)

            # Compile all results for this provider
            if two_shot_all_results:
                two_shot_results_df = pd.DataFrame(two_shot_all_results)
                two_shot_results_df.to_csv(os.path.join(two_shot_output_dir, f\"{two_shot_provider}_two_shot_results.csv\"), index=False)

                # Calculate aggregate metrics
                calculate_aggregate_metrics(two_shot_output_dir, f\"{two_shot_provider}\")

                # Store for comparison
                two_shot_results[f\"{two_shot_provider}\"] = two_shot_results_df

                # Add to comparison data
                two_shot_comparison_data[f\"{two_shot_provider}\"] = {
                    \"model\": f\"{provider_models[two_shot_provider]} + Transkribus (Two-Shot)\",
                    \"avg_cer\": two_shot_results_df['cer'].mean(),
                    \"avg_wer\": two_shot_results_df['wer'].mean(),
                    \"avg_bwer\": two_shot_results_df['bwer'].mean(),
                    \"doc_count\": len(two_shot_results_df)
                }



    """,
    name="_"
)


@app.cell
def _(two_shot_results):
    two_shot_results
    return


@app.cell
def _(mo, pd, provider_models, transkribus_df, two_shot_results):
    two_shot_transkribus_metrics = {
        "avg_cer": transkribus_df['cer'].mean() if not transkribus_df.empty else None,
        "avg_wer": transkribus_df['wer'].mean() if not transkribus_df.empty else None,
        "avg_bwer": transkribus_df['bwer'].mean() if not transkribus_df.empty else None
    }

    two_shot_final_comparison_data = {}
    for ts_provider_name, ts_df in two_shot_results.items():
        if not ts_df.empty:
            two_shot_final_comparison_data[ts_provider_name] = {
                "model": f"{provider_models[ts_provider_name.replace('_two_shot', '')]} + Transkribus (Two-Shot)",
                "avg_cer": ts_df['cer'].mean(),
                "avg_wer": ts_df['wer'].mean(),
                "avg_bwer": ts_df['bwer'].mean(),
                "doc_count": len(ts_df)
            }

    two_shot_final_comparison_df = pd.DataFrame(two_shot_final_comparison_data).T

    # Add Transkribus for comparison
    if not transkribus_df.empty:
        two_shot_final_comparison_df.loc['transkribus'] = {
            "model": "Text Titan 1",
            "avg_cer": two_shot_transkribus_metrics["avg_cer"],
            "avg_wer": two_shot_transkribus_metrics["avg_wer"],
            "avg_bwer": two_shot_transkribus_metrics["avg_bwer"],
            "doc_count": len(transkribus_df)
        }

    # Create fresh stat components for each provider with Transkribus comparison
    two_shot_final_provider_stats = []
    for ts_provider_name, ts_metrics in two_shot_final_comparison_data.items():
        ts_model_name = ts_metrics["model"]

        # Skip Transkribus in this section
        if ts_provider_name == 'transkribus':
            continue

        # Compare with Transkribus metrics and determine direction
        if two_shot_transkribus_metrics["avg_cer"] is not None:
            # Calculate absolute differences
            ts_cer_diff = two_shot_transkribus_metrics["avg_cer"] - ts_metrics["avg_cer"]
            ts_wer_diff = two_shot_transkribus_metrics["avg_wer"] - ts_metrics["avg_wer"]
            ts_bwer_diff = two_shot_transkribus_metrics["avg_bwer"] - ts_metrics["avg_bwer"]

            # Calculate percentage changes (relative to Transkribus baseline)
            ts_cer_pct = (ts_cer_diff / two_shot_transkribus_metrics["avg_cer"]) * 100 if two_shot_transkribus_metrics["avg_cer"] > 0 else 0
            ts_wer_pct = (ts_wer_diff / two_shot_transkribus_metrics["avg_wer"]) * 100 if two_shot_transkribus_metrics["avg_wer"] > 0 else 0
            ts_bwer_pct = (ts_bwer_diff / two_shot_transkribus_metrics["avg_bwer"]) * 100 if two_shot_transkribus_metrics["avg_bwer"] > 0 else 0

            # For error metrics like CER/WER/BWER, lower is better
            # So if our model has lower error (ts_cer_diff > 0), that's an improvement
            ts_cer_direction = "increase" if ts_cer_diff > 0 else "decrease"
            ts_wer_direction = "increase" if ts_wer_diff > 0 else "decrease"
            ts_bwer_direction = "increase" if ts_bwer_diff > 0 else "decrease"

            # Format percentage with sign (positive means improvement)
            ts_cer_caption = f"{ts_cer_pct:+.1f}% vs Transkribus"
            ts_wer_caption = f"{ts_wer_pct:+.1f}% vs Transkribus"
            ts_bwer_caption = f"{ts_bwer_pct:+.1f}% vs Transkribus"

            # Create stat components row for this provider
            two_shot_final_provider_stats.append(mo.md(f"### {ts_provider_name.upper()}"))
            two_shot_final_provider_stats.append(
                mo.hstack([
                    mo.stat(
                        f"{ts_metrics['avg_cer']:.4f}",
                        label="Average CER",
                        caption=ts_cer_caption,
                        direction=ts_cer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{ts_metrics['avg_wer']:.4f}",
                        label="Average WER",
                        caption=ts_wer_caption,
                        direction=ts_wer_direction,
                        bordered=True
                    ),
                    mo.stat(
                        f"{ts_metrics['avg_bwer']:.4f}",
                        label="Average BWER",
                        caption=ts_bwer_caption,
                        direction=ts_bwer_direction,
                        bordered=True
                    ),
                    mo.stat(ts_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )
        else:
            # If Transkribus metrics aren't available, show regular stats
            two_shot_final_provider_stats.append(mo.md(f"### {ts_provider_name.upper()}"))
            two_shot_final_provider_stats.append(
                mo.hstack([
                    mo.stat(f"{ts_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{ts_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{ts_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                    mo.stat(ts_metrics["doc_count"], label="Documents", bordered=True)
                ])
            )

    # Add Transkribus baseline stats at the top
    if not transkribus_df.empty:
        two_shot_transkribus_stats = [
            mo.md("### TRANSKRIBUS with Text Titan 1 (Baseline)"),
            mo.hstack([
                mo.stat(f"{two_shot_transkribus_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{two_shot_transkribus_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{two_shot_transkribus_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                mo.stat(len(transkribus_df), label="Documents", bordered=True)
            ])
        ]
    else:
        two_shot_transkribus_stats = [mo.md("### Transkribus baseline not available")]

    # Combine everything into a vstack layout
    mo.vstack([
        mo.md("## Two-Shot Hybrid Model Performance Metrics"),
        *two_shot_transkribus_stats,
        mo.md("### MM-LLM + Transkribus Two-Shot Hybrid Models"),
        *two_shot_final_provider_stats,
        mo.md("## Comparison Table"),
        mo.ui.table(two_shot_final_comparison_df)
    ])
    return (
        ts_bwer_caption,
        ts_bwer_diff,
        ts_bwer_direction,
        ts_bwer_pct,
        ts_cer_caption,
        ts_cer_diff,
        ts_cer_direction,
        ts_cer_pct,
        ts_df,
        ts_metrics,
        ts_model_name,
        ts_provider_name,
        ts_wer_caption,
        ts_wer_diff,
        ts_wer_direction,
        ts_wer_pct,
        two_shot_final_comparison_data,
        two_shot_final_comparison_df,
        two_shot_final_provider_stats,
        two_shot_transkribus_metrics,
        two_shot_transkribus_stats,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
