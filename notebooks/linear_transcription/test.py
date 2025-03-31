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

    model = OpenAICompatibleModel("openai", "gpt-4o")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please say hello!"},
            ]
        }
    ]

    response = model.client.chat.completions.create(
        model=model.model_name,
        messages=messages,
        temperature=0
    )

    text = response.choices[0].message.content
    text
    return OpenAICompatibleModel, messages, model, response, text


@app.cell
def _(mo):
    # Dropdown to select which model to use
    model_selection = mo.ui.dropdown(
        options={
            "OpenAI GPT-4o": "openai_gpt4o",
            "Gemini 2.0 Flash": "gemini_flash",
            "Mistral Pixtral Large": "mistral_large"
        },
        value="OpenAI GPT-4o",
        label="Select Model"
    )

    # Display the dropdown
    model_selection
    return (model_selection,)


@app.cell
def _(OpenAICompatibleModel, model_selection):


    # Function that creates a model instance based on the selection
    def get_model(model_type):
        if model_type == "openai_gpt4o":
            return OpenAICompatibleModel("openai", "gpt-4o")
        elif model_type == "gemini_flash":
            return OpenAICompatibleModel("gemini", "gemini-2.0-flash")
        elif model_type == "mistral_large":
            return OpenAICompatibleModel("mistral", "pixtral-large")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # Create a custom model function for the chat
    def transcription_assistant(messages, config=None):
        # Extract the last user message
        if not messages or messages[-1].role != "user":
            return "Please ask a question about transcription."

        user_message = messages[-1].content

        # Get image attachment if any
        image_attachment = None
        if hasattr(messages[-1], 'attachments') and messages[-1].attachments:
            for attachment in messages[-1].attachments:
                if attachment.content_type.startswith('image/'):
                    image_attachment = attachment.url

        # Format the prompt
        system_prompt = """
        You are a specialized transcription assistant for historical documents.
        Provide accurate insights about transcription techniques, OCR systems,
        and best practices for working with historical manuscripts.
        If an image is attached, transcribe the text accurately, preserving line breaks.
        """

        # Create API messages structure
        api_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages[:-1]:
            api_messages.append({"role": msg.role, "content": msg.content})

        # Handle the last message with possible image
        if image_attachment:
            api_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_attachment}
                    }
                ]
            })
        else:
            api_messages.append({"role": "user", "content": user_message})

        try:
            # Get current model selection value and create the model
            current_model = get_model(model_selection.value)

            # Make the API call
            response = current_model.client.chat.completions.create(
                model=current_model.model_name,
                messages=api_messages,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}\n\nMake sure you have set the appropriate API key in your environment variables."


    return get_model, transcription_assistant


@app.cell
def _(mo, transcription_assistant):
    # Create the chat UI
    chatbot = mo.ui.chat(
        transcription_assistant,
        prompts=[
            "What are the best practices for transcribing medieval manuscripts?",
            "Can you compare Transkribus with other OCR systems?",
            "Please transcribe this image for me.",
            "What CER (Character Error Rate) should I expect from a {{model_name}} on {{document_type}}?"
        ],
        allow_attachments=["image/png", "image/jpeg"],
    )

    # Display the chat interface
    chatbot
    return (chatbot,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
