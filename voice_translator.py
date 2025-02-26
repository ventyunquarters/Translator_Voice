import gradio as gr
import assemblyai as aai  # For transcription
from deep_translator import GoogleTranslator  # Using Deep Translator
from elevenlabs import generate, set_api_key #Using Eleven Labs
import os
import io
from concurrent.futures import ThreadPoolExecutor
import uuid

# Set API keys
ELEVEN_LABS_API_KEY = "enterelabsapikeyhere"
ASSEMBLY_AI_API_KEY = "enteraaiapikeyhere"

# Ensure API keys are not None
if not ELEVEN_LABS_API_KEY or not ASSEMBLY_AI_API_KEY:
    raise ValueError("Missing API keys! Make sure to set them correctly.")

set_api_key(ELEVEN_LABS_API_KEY)
aai.settings.api_key = ASSEMBLY_AI_API_KEY

# Function to transcribe audio
def transcribe_audio(audio):
    try:
        config = aai.TranscriptionConfig()
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio).wait_for_completion()
        return transcript.text
    except Exception as e:
        return f"Error in transcription: {str(e)}"

# Function to translate text
def translate_text(text):
    try:
        targets = {"Spanish": "es", "German": "de", "Japanese": "ja"}
        with ThreadPoolExecutor() as executor:
            translations = {
                lang: GoogleTranslator(source="auto", target=code).translate(text)
                for lang, code in targets.items()
            }
        return translations
    except Exception as e:
        return {"Error": f"Translation failed: {str(e)}"}

# Function to generate TTS audio
def generate_tts(text, lang):
    try:
        tts_audio = generate(text=text, voice="2WvAXMgrakBkapSmnlv7")
        file_path = f"{uuid.uuid4()}.mp3"
        with open(file_path, "wb") as f:
            f.write(tts_audio)
        return file_path
    except Exception as e:
        return f"Error generating TTS for {lang}: {str(e)}"

# Function to process audio input
def process_audio(audio):
    transcribed_text = transcribe_audio(audio)
    if transcribed_text.startswith("Error"):
        return transcribed_text, None, None, None

    translations = translate_text(transcribed_text)
    if "Error" in translations:
        return transcribed_text, None, None, None

    audio_outputs = {
        lang: generate_tts(text, lang)
        for lang, text in translations.items()
    }

    return transcribed_text, audio_outputs["Spanish"], audio_outputs["German"], audio_outputs["Japanese"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Audio Transcription and Translation")
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
        submit_button = gr.Button("Submit")
    
    transcribed_output = gr.Textbox(label="Transcribed Text")
    audio_outputs = [
        gr.Audio(label="Spanish"),
        gr.Audio(label="German"),
        gr.Audio(label="Japanese")
    ]

    submit_button.click(process_audio, inputs=audio_input, outputs=[transcribed_output] + audio_outputs)

demo.launch()
