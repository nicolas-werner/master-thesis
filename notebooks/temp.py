import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    import sys
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return mo, os, project_root, sys


@app.cell
def _():
    from src.file_utils import extract_text_from_xml
    from src.file_utils import find_file_for_id, extract_line_coords_from_xml
    from src.evaluation import run_evaluation, get_transkribus_text, run_line_evaluation, process_page_by_lines
    from src.file_utils import encode_image, encode_image_object

    transkribus_text = extract_text_from_xml("data/reichenau_10_test/ground_truth/7474185.xml")
    print("\n".join(transkribus_text))
    return (
        encode_image,
        encode_image_object,
        extract_line_coords_from_xml,
        extract_text_from_xml,
        find_file_for_id,
        get_transkribus_text,
        process_page_by_lines,
        run_evaluation,
        run_line_evaluation,
        transkribus_text,
    )


@app.cell
def _():
    from src.models.openai_compatible import OpenAICompatibleModel
    from pydantic import BaseModel, Field

    model = OpenAICompatibleModel("mistral", "mistral-small-latest")

    class Transcription(BaseModel):
        correct_transcription: str = Field(..., description="The correct transcription.")

    completion = model.client.beta.chat.completions.parse(
        model = model.model_name,
        messages=[
            {"role": "system", "content": "You are a transcription expert. You will be given a text and you will transcribe it."},
            {"role": "user", "content": "Xegion ſo wol ein welt genempt mag werden"}
        ],
        response_format=Transcription,
    )

    output = completion.choices[0].message.parsed.correct_transcription.strip()



    print(output)
    return (
        BaseModel,
        Field,
        OpenAICompatibleModel,
        Transcription,
        completion,
        model,
        output,
    )


@app.cell
def _(mo):
    mo.md(r"""# Prompt Improvement""")
    return


@app.cell
def _():
    # other system prompts:

    ## Reichenau Pagewise:
        # You are a expert for medieval incunabula — early printed books from the 15th century — written in Early New High German.
        # Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
        # preserving the original text without modernizing or correcting spelling.
        # Important instructions:
        # 1. Pay particular attention to:
        #    - Early printed ligatures (connected letter combinations)
        #    - Abbreviation marks and symbols
        #    - Special characters from medieval Latin or German, such as: ¬, uͤ, n̄, ē, ſ, ꝛ, ꝯ, ꝫ, ꝓ, ꝟ, ẜ, etc.
        # 2. Preserve abbreviations and special characters
        # 3. Separate each line with a newline character (\\n)
        # 4. Do not add any explanations or commentary
        # 5. Do not include line numbers

        # Your transcription should match the original manuscript as closely as possible. If you're uncertain about a character, provide your best interpretation.

        # CRITICAL LINE BREAK INSTRUCTIONS:
        # - You MUST maintain the EXACT same number of lines as in the original manuscript
        # - Each physical line in the manuscript should be ONE line in your transcription
        # - DO NOT merge short lines together
        # - DO NOT split long lines into multiple lines
        # - Preserve the exact same line structure as the manuscript


    ## Reichenau Linewise:
        # You are an expert in historical typography and early printed books (incunabula) from the 15th century. Please transcribe the provided line image exactly as it appears.

        # Important instructions:
        # 1. If the image shows part of an illustration, decoration, or non-textual element, return an empty string.
        # 2. Focus ONLY on the primary text line in the center of the image, ignoring partial text from lines above or below.
        # 3. Preserve original historical spellings, abbreviations, and special characters.
        # 4. Pay particular attention to:
        #    - Early printed ligatures (connected letter combinations)
        #    - Abbreviation marks and symbols
        #    - Special characters from medieval Latin or German, such as: ¬, uͤ, n̄, ē, ſ, ꝛ, ꝯ, ꝫ, ꝓ, ꝟ, ẜ, etc.
        # 5. Do not expand abbreviations.
        # 6. Do not add any explanations or commentary. Only return the transcribed text.
        # 7. Transcribe exactly what you see, maintaining historical orthography.
        # 8. If the image contains no text, return an empty string.

        # If you're uncertain about a character, provide your best interpretation.



    ## Bentham Pagewise
        # You are a expert for historical english handwritten manuscripts — especially those authored by Jeremy Bentham (1748–1832).
        # Please transcribe the provided manuscript image line by line. Transcribe exactly what you see in the image,
        # preserving the original text without modernizing or correcting spelling.

        # Important instructions:
        # 1. Use original historical characters and spelling.
        # 2. Preserve abbreviations, marginalia, and special characters.
        # 3. Separate each line with a newline character (\n)
        # 4. Do not add any explanations or commentary
        # 5. Do not include line numbers
        # 6. Transcribe text only, ignore decorative elements or stamps unless they contain readable text
        # 7. Ignore text that is clearly struck through in the manuscript

        # Your transcription should match the original manuscript as closely as possible. If you're uncertain about a character, provide your best interpretation.

        # CRITICAL LINE BREAK INSTRUCTIONS:
        # - You MUST maintain the EXACT same number of lines as in the original manuscript
        # - Each physical line in the manuscript should be ONE line in your transcription
        # - DO NOT merge short lines together
        # - DO NOT split long lines into multiple lines
        # - Preserve the exact same line structure as the manuscript
    return


@app.cell
def _(mo):
    mo.image(src="data/reichenau_10_test/images/7474187.jpg")
    return


@app.cell
def _(mo):
    system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=20, value="""
    You are an expert in historical typography and early printed books (incunabula) from the 15th century. You will be presented with images from single text lines. Please transcribe the provided line image exactly as it appears.
    Important instructions:
    1. If the image shows part of an illustration, decoration, or non-textual element, return an empty string ("").
    2. Preserve original historical spellings, abbreviations, and special characters.
    3. Pay particular attention to:
       - Early printed ligatures (connected letter combinations)
       - Abbreviation marks and symbols
       - Special characters from medieval Latin or German, such as: ¬, uͤ, n̄, ē, ſ, ꝛ, ꝯ, ꝫ, ꝓ, ꝟ, ẜ, etc.
    4. Do not expand abbreviations.
    5. Do not add any explanations or commentary. Only return the transcribed text.
    6. Transcribe exactly what you see, maintaining historical orthography.
    7. If the image contains no text, return an empty string.

    The Line Segmentation for the page could be faulty. It could happen that the line images contain line artifacts from other lines. Focus ONLY on the primary text line in the the image, ignoring partial text from lines above or below.
    If you're uncertain about a character, provide your best interpretation.
    """)

    provider_models = {
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "mistral": "mistral-small-latest",

    }

    system_prompt
    return provider_models, system_prompt


@app.cell
def _(encode_image, extract_text_from_xml):
    help_text = extract_text_from_xml("results/linear_transcription/reichenau_inkunabeln/transkribus_10_test/0004_7474187.xml")

    transcription_image = encode_image(image_path="data/reichenau_10_test/images/7474187.jpg")
    return help_text, transcription_image


@app.cell
def _():
    # from openai import OpenAI

    # client = OpenAI(
    #   base_url="https://openrouter.ai/api/v1",
    #   api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    # )

    # t_completion = client.chat.completions.create(
    #   model="openai/gpt-4.1",
    #         messages = [
    #             {"role": "system", "content": system_prompt.value},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {"url": f"data:image/jpeg;base64,{transcription_image}"}
    #                     },
    #                     {"type": "text", "text": f"The following is the output of a traditional OCR model (Transkribus) for this page. It can help with your transcription, but may contain errors:\n\n{transkribus_text}"},
    #                 ]
    #             }
    #         ],
    #     temperature=0,
    #     seed=42
    # )
    # print(t_completion.choices[0].message.content)
    return


@app.cell
def _():
    # from src.metrics import evaluate_transcription
    # gt_text = extract_text_from_xml("data/reichenau_10_test/ground_truth/7474187.xml")

    # result = evaluate_transcription(gt_text, t_completion.choices[0].message.content.split('\n'))
    # result
    return


@app.cell
def _(
    encode_image_object,
    extract_text_from_xml,
    find_file_for_id,
    os,
    process_page_by_lines,
    system_prompt,
):
    doc_id = "7474184"

    # Define paths
    image_path = f"data/reichenau_10_test/images/{doc_id}.jpg"
    transkribus_path = find_file_for_id(
        doc_id,
        'results/linear_transcription/reichenau_inkunabeln/transkribus_10_test',
        ['.xml']
    )
    gt_path = f"data/reichenau_10_test/ground_truth/{doc_id}.xml"

    # Process page to get line images
    from PIL import Image

    page_data = process_page_by_lines(image_path, transkribus_path)

    # Get ground truth for later evaluation
    gt_lines = extract_text_from_xml(gt_path)

    # Initialize model
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )

    # Store results for each line
    all_pred_lines = []

    # Process each line individually
    for line_idx, line_data in enumerate(page_data['lines']):
        # Encode the line image
        line_image_base64 = encode_image_object(line_data['image'])

        # Get Transkribus text for this line (if available)
        transkribus_line_text = line_data.get('text', '')

        # Call the model for this line
        try:
            line_response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": system_prompt.value},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{line_image_base64}"}
                            },
                            {
                                "type": "text", 
                                "text": f"The following is the OCR output for this line. It may contain errors:\n\n{transkribus_line_text}"
                            }
                        ]
                    }
                ],
                temperature=0,
                seed=42
            )

            # Extract the transcription for this line
            line_text = line_response.choices[0].message.content.strip()
            all_pred_lines.append(line_text)
            print(f"Line {line_idx+1}: {line_text}")

        except Exception as e:
            print(f"Error processing line {line_idx}: {str(e)}")
            all_pred_lines.append("")

    # Calculate metrics
    from src.metrics import evaluate_transcription
    results = evaluate_transcription(gt_lines, all_pred_lines)
    metrics = results['document_metrics']

    print(f"\nMetrics for linewise approach:")
    print(f"CER: {metrics['cer']:.4f}, WER: {metrics['wer']:.4f}, BWER: {metrics['bwer']:.4f}")

    # Combine all lines for complete transcription
    full_transcription = '\n'.join(all_pred_lines)
    return (
        Image,
        OpenAI,
        all_pred_lines,
        client,
        doc_id,
        evaluate_transcription,
        full_transcription,
        gt_lines,
        gt_path,
        image_path,
        line_data,
        line_idx,
        line_image_base64,
        line_response,
        line_text,
        metrics,
        page_data,
        results,
        transkribus_line_text,
        transkribus_path,
    )


@app.cell
def _(all_pred_lines, gt_lines, mo, page_data):
    import io
    import base64

    rows = []

    for i, line_d in enumerate(page_data['lines'][:10]): 
        img_byte_arr = io.BytesIO()
        line_d['image'].save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
        d_transkribus_text = line_d.get('text', '')
        model_text = all_pred_lines[i] if all_pred_lines and i < len(all_pred_lines) else ""
        gt_text = gt_lines[i] if gt_lines and i < len(gt_lines) else ""
    
        rows.append({
            "Line": f"{i+1}",
            "Image": mo.image(src=f"data:image/png;base64,{img_base64}", width=200),
            "Transkribus": d_transkribus_text,
            "Model Output": model_text,
            "Ground Truth": gt_text
        })

    table = mo.ui.table(rows, selection=None)
    table   
    return (
        base64,
        d_transkribus_text,
        gt_text,
        i,
        img_base64,
        img_byte_arr,
        io,
        line_d,
        model_text,
        rows,
        table,
    )


if __name__ == "__main__":
    app.run()
