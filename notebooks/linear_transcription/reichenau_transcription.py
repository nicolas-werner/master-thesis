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
def _():
    from src.models import OpenAICompatibleModel
    from src.file_utils import encode_image, load_prompt_from_file

    model = OpenAICompatibleModel("openai", "gpt-4o")

    system_prompt = """
    Transcribe the following image accurately.
    """

    # Encode the image
    image_base64 = encode_image("data/reichenau_inkunabeln/images/7474184.jpg")

    # Create the message structure directly in the notebook
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please transcribe this manuscript image accurately."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ]

    # Make the API call directly
    response = model.client.chat.completions.create(
        model=model.model_name,
        messages=messages,
        temperature=0
    )

    # Process the response
    text = response.choices[0].message.content.strip()
    lines = text.split('\n')

    print(f"Transcribed {len(lines)} lines:")
    for line in lines:
        print(line)
    return (
        OpenAICompatibleModel,
        encode_image,
        image_base64,
        line,
        lines,
        load_prompt_from_file,
        messages,
        model,
        response,
        system_prompt,
        text,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
