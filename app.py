"""
Gradio UI for Text-to-Speech using HiggsAudioServeEngine
"""

import argparse
import base64
import os
import uuid
import json
from typing import Optional
import gradio as gr
from loguru import logger
import numpy as np
import time
from functools import lru_cache
import re
import spaces
import torch

# Import HiggsAudio components
from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
from higgs_audio.data_types import ChatMLSample, AudioContent, Message

# Global engine instance
engine = None

# Default model configuration
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-staging"
DEFAULT_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer-staging"
SAMPLE_RATE = 24000

DEFAULT_SYSTEM_PROMPT = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\n"
    "Audio is recorded from a quiet room.\n"
    "<|scene_desc_end|>"
)

DEFAULT_STOP_STRINGS = ["<|end_of_text|>", "<|eot_id|>"]

# Predefined examples for system and input messages
PREDEFINED_EXAMPLES = {
    "voice-clone": {
        "system_prompt": "",
        "input_text": "Hey there! I'm your friendly voice twin in the making. Pick a voice preset below or upload your own audio - let's clone some vocals and bring your voice to life! ",
        "description": "Voice clone template",
    },
    "zero-shot": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "input_text": "Hey hey! Welcome to Higgs Audio, your voice's new best friend. Drop your text below, and I'll turn it into something that sounds awesome! Let's make some audio magic!",
        "description": "Zero-shot template",
    },
    "multispeaker-interleave": {
        "system_prompt": "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: vocal fry;feminism;slightly fast\n"
        "SPEAKER1: masculine;moderate;moderate pitch;monotone;mature\n"
        "In this scene, a group of adventurers is debating whether to investigate a potentially dangerous situation.\n"
        "<|scene_desc_end|>",
        "input_text": "<|generation_instruction_start|>\nGenerate interleaved transcript and audio that lasts for around 10 seconds.\n<|generation_instruction_end|>",
        "description": "Multispeaker interleave example",
    },
    "single-speaker-accent": {
        "system_prompt": "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: british accent\n"
        "<|scene_desc_end|>",
        "input_text": "Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
        "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
        "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.\n"
        "\n"
        "So here's the big question: Do you want to understand how deep learning works?\n",
        "description": "Single speaker example",
    },
    "single-speaker-zh": {
        "system_prompt": "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "\nAudio is recorded from a quiet room.\n"
        "\nSPEAKER0: feminine\n"
        "<|scene_desc_end|>",
        "input_text": "大家好, 欢迎收听本期的跟李沐学AI. 今天沐哥在忙着洗数据, 所以由我, 希格斯主播代替他讲这期视频.\n"
        "今天我们要聊的是一个你绝对不能忽视的话题: 多模态学习.\n"
        "那么, 问题来了, 你真的了解多模态吗? 你知道如何自己动手构建多模态大模型吗.\n"
        "或者说, 你能察觉到我其实是个机器人吗?",
        "description": "Single speaker with Chinese text",
    },
}


@lru_cache(maxsize=20)
def encode_audio_file(file_path):
    """Encode an audio file to base64."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def get_current_device():
    """Get the current device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_voice_presets():
    """Load the voice presets from the voice_examples directory."""
    try:
        with open(
            os.path.join(os.path.dirname(__file__), "voice_examples", "config.json"),
            "r",
        ) as f:
            voice_dict = json.load(f)
        voice_presets = {k: v["transcript"] for k, v in voice_dict.items()}
        voice_presets["EMPTY"] = "No reference voice"
        logger.info(f"Loaded voice presets: {list(voice_presets.keys())}")
        return voice_presets
    except FileNotFoundError:
        logger.warning("Voice examples config file not found. Using empty voice presets.")
        return {"EMPTY": "No reference voice"}
    except Exception as e:
        logger.error(f"Error loading voice presets: {e}")
        return {"EMPTY": "No reference voice"}


def get_voice_present(voice_preset):
    """Get the voice path and text for a given voice preset."""
    voice_path = os.path.join(os.path.dirname(__file__), "voice_examples", f"{voice_preset}.wav")
    if not os.path.exists(voice_path):
        logger.warning(f"Voice preset file not found: {voice_path}")
        return None, "Voice preset not found"

    text = VOICE_PRESETS.get(voice_preset, "No transcript available")
    return voice_path, text


@spaces.GPU
def initialize_engine(model_path, audio_tokenizer_path) -> bool:
    """Initialize the HiggsAudioServeEngine."""
    global engine
    try:
        logger.info(f"Initializing engine with model: {model_path} and audio tokenizer: {audio_tokenizer_path}")
        engine = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=audio_tokenizer_path,
            device=get_current_device(),
        )
        logger.info(f"Successfully initialized HiggsAudioServeEngine with model: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        return False


def check_return_audio(audio_wv: np.ndarray):
    # check if the audio returned is all silent
    if np.all(audio_wv == 0):
        logger.warning("Audio is silent, returning None")


def process_text_output(text_output: str):
    # remove all the continuous <|AUDIO_OUT|> tokens with a single <|AUDIO_OUT|>
    text_output = re.sub(r"(<\|AUDIO_OUT\|>)+", r"<|AUDIO_OUT|>", text_output)
    return text_output


def prepare_chatml_sample(
    voice_present: str,
    text: str,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """Prepare a ChatMLSample for the HiggsAudioServeEngine."""
    messages = []

    # Add system message if provided
    if len(system_prompt) > 0:
        messages.append(Message(role="system", content=system_prompt))

    # Add reference audio if provided
    audio_base64 = None
    ref_text = ""

    if reference_audio:
        # Custom reference audio
        audio_base64 = encode_audio_file(reference_audio)
        ref_text = reference_text or ""
    elif voice_present != "EMPTY":
        # Voice preset
        voice_path, ref_text = get_voice_present(voice_present)
        if voice_path is None:
            logger.warning(f"Voice preset {voice_present} not found, skipping reference audio")
        else:
            audio_base64 = encode_audio_file(voice_path)

    # Only add reference audio if we have it
    if audio_base64 is not None:
        # Add user message with reference text
        messages.append(Message(role="user", content=ref_text))

        # Add assistant message with audio content
        audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
        messages.append(Message(role="assistant", content=[audio_content]))

    # Add the main user message
    messages.append(Message(role="user", content=text))

    return ChatMLSample(messages=messages)


@spaces.GPU(duration=500)
def text_to_speech(
    text,
    voice_preset,
    reference_audio=None,
    reference_text=None,
    max_completion_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    stop_strings=None,
):
    """Convert text to speech using HiggsAudioServeEngine."""
    global engine

    if engine is None:
        initialize_engine(DEFAULT_MODEL_PATH, DEFAULT_AUDIO_TOKENIZER_PATH)

    try:
        # Prepare ChatML sample
        chatml_sample = prepare_chatml_sample(voice_preset, text, reference_audio, reference_text, system_prompt)

        # Convert stop strings format
        if stop_strings is None:
            stop_list = DEFAULT_STOP_STRINGS
        else:
            stop_list = [s for s in stop_strings["stops"] if s.strip()]

        request_id = f"tts-playground-{str(uuid.uuid4())}"
        logger.info(
            f"{request_id}: Generating speech for text: {text[:100]}..., \n"
            f"with parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, stop_list={stop_list}"
        )
        start_time = time.time()

        # Generate using the engine
        response = engine.generate(
            chat_ml_sample=chatml_sample,
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            stop_strings=stop_list,
        )

        generation_time = time.time() - start_time
        logger.info(f"{request_id}: Generated audio in {generation_time:.3f} seconds")
        gr.Info(f"Generated audio in {generation_time:.3f} seconds")

        # Process the response
        text_output = process_text_output(response.generated_text)

        if response.audio is not None:
            # Convert to int16 for Gradio
            audio_data = (response.audio * 32767).astype(np.int16)
            check_return_audio(audio_data)
            return text_output, (response.sampling_rate, audio_data)
        else:
            logger.warning("No audio generated")
            return text_output, None

    except Exception as e:
        error_msg = f"Error generating speech: {e}"
        logger.error(error_msg)
        gr.Error(error_msg)
        return f"❌ {error_msg}", None


def create_ui():
    my_theme = "JohnSmith9982/small_and_pretty"

    # Add custom CSS to disable focus highlighting on textboxes
    custom_css = """
    .gradio-container input:focus, 
    .gradio-container textarea:focus, 
    .gradio-container select:focus,
    .gradio-container .gr-input:focus,
    .gradio-container .gr-textarea:focus,
    .gradio-container .gr-textbox:focus,
    .gradio-container .gr-textbox:focus-within,
    .gradio-container .gr-form:focus-within,
    .gradio-container *:focus {
        box-shadow: none !important;
        border-color: var(--border-color-primary) !important;
        outline: none !important;
        background-color: var(--input-background-fill) !important;
    }

    /* Override any hover effects as well */
    .gradio-container input:hover, 
    .gradio-container textarea:hover, 
    .gradio-container select:hover,
    .gradio-container .gr-input:hover,
    .gradio-container .gr-textarea:hover,
    .gradio-container .gr-textbox:hover {
        border-color: var(--border-color-primary) !important;
        background-color: var(--input-background-fill) !important;
    }

    /* Style for checked checkbox */
    .gradio-container input[type="checkbox"]:checked {
        background-color: var(--primary-500) !important;
        border-color: var(--primary-500) !important;
    }
    """

    """Create the Gradio UI."""
    with gr.Blocks(theme=my_theme, css=custom_css) as demo:
        gr.Markdown("# Higgs Audio Text-to-Speech Playground")

        # Main UI section
        with gr.Row():
            with gr.Column(scale=2):
                # Template selection dropdown
                template_dropdown = gr.Dropdown(
                    label="TTS Template",
                    choices=list(PREDEFINED_EXAMPLES.keys()),
                    value="zero-shot",
                    info="Select a predefined example for system and input messages. Voice preset will be set to EMPTY when a example is selected.",
                )

                system_prompt = gr.TextArea(
                    label="System Prompt",
                    placeholder="Enter system prompt to guide the model...",
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=2,
                )

                input_text = gr.TextArea(
                    label="Input Text",
                    placeholder="Type the text you want to convert to speech...",
                    lines=5,
                )

                voice_preset = gr.Dropdown(
                    label="Voice Preset",
                    choices=list(VOICE_PRESETS.keys()),
                    value="EMPTY",
                    interactive=False,  # Disabled by default since default template is not voice-clone
                    visible=False,
                )

                with gr.Accordion(
                    "Custom Reference (Optional)", open=False, visible=False
                ) as custom_reference_accordion:
                    reference_audio = gr.Audio(label="Reference Audio", type="filepath")
                    reference_text = gr.TextArea(
                        label="Reference Text (transcript of the reference audio)",
                        placeholder="Enter the transcript of your reference audio...",
                        lines=3,
                    )

                with gr.Accordion("Advanced Parameters", open=False):
                    max_completion_tokens = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        value=1024,
                        step=10,
                        label="Max Completion Tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top P")
                    top_k = gr.Slider(minimum=-1, maximum=100, value=50, step=1, label="Top K")
                    # Add stop strings component
                    stop_strings = gr.Dataframe(
                        label="Stop Strings",
                        headers=["stops"],
                        datatype=["str"],
                        value=[[s] for s in DEFAULT_STOP_STRINGS],
                        interactive=True,
                        col_count=(1, "fixed"),
                    )

                submit_btn = gr.Button("Generate Speech", variant="primary", scale=1)

            with gr.Column(scale=2):
                output_text = gr.TextArea(label="Model Response", lines=2)

                # Audio output
                output_audio = gr.Audio(label="Generated Audio", interactive=False, autoplay=True)

                stop_btn = gr.Button("Stop Playback", variant="primary")

        # Example voice
        with gr.Row():
            voice_samples_table = gr.Dataframe(
                headers=["Voice Preset", "Sample Text"],
                datatype=["str", "str"],
                value=[[preset, text] for preset, text in VOICE_PRESETS.items() if preset != "EMPTY"],
                interactive=False,
            )
            sample_audio = gr.Audio(label="Voice Sample", visible=True)

        # Function to play voice sample when clicking on a row
        def play_voice_sample(evt: gr.SelectData):
            try:
                # Get the preset name from the clicked row
                preset_names = [preset for preset in VOICE_PRESETS.keys() if preset != "EMPTY"]
                if evt.index[0] < len(preset_names):
                    preset = preset_names[evt.index[0]]
                    voice_path, _ = get_voice_present(preset)
                    if voice_path and os.path.exists(voice_path):
                        return voice_path
                    else:
                        gr.Warning(f"Voice sample file not found for preset: {preset}")
                        return None
                else:
                    gr.Warning("Invalid voice preset selection")
                    return None
            except Exception as e:
                logger.error(f"Error playing voice sample: {e}")
                gr.Error(f"Error playing voice sample: {e}")
                return None

        voice_samples_table.select(fn=play_voice_sample, outputs=[sample_audio])

        # Function to handle template selection
        def apply_template(template_name):
            if template_name in PREDEFINED_EXAMPLES:
                template = PREDEFINED_EXAMPLES[template_name]
                # Enable voice preset and custom reference only for voice-clone template
                is_voice_clone = template_name == "voice-clone"
                return (
                    template["system_prompt"],  # system_prompt
                    template["input_text"],  # input_text
                    gr.update(
                        value="wizard", interactive=is_voice_clone, visible=is_voice_clone
                    ),  # voice_preset (value and interactivity)
                    gr.update(visible=is_voice_clone),  # custom reference accordion visibility
                    gr.update(visible=is_voice_clone),  # voice samples table visibility
                    gr.update(visible=is_voice_clone),  # sample audio visibility
                )
            else:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )  # No change if template not found

        # Set up event handlers

        # Connect template dropdown to handler
        template_dropdown.change(
            fn=apply_template,
            inputs=[template_dropdown],
            outputs=[
                system_prompt,
                input_text,
                voice_preset,
                custom_reference_accordion,
                voice_samples_table,
                sample_audio,
            ],
        )

        # Connect submit button to the TTS function
        submit_btn.click(
            fn=text_to_speech,
            inputs=[
                input_text,
                voice_preset,
                reference_audio,
                reference_text,
                max_completion_tokens,
                temperature,
                top_p,
                top_k,
                system_prompt,
                stop_strings,
            ],
            outputs=[output_text, output_audio],
            api_name="generate_speech",
        )

        # Stop button functionality
        stop_btn.click(
            fn=lambda: None,
            inputs=[],
            outputs=[output_audio],
            js="() => {const audio = document.querySelector('audio'); if(audio) audio.pause(); return null;}",
        )

    return demo


def main():
    """Main function to parse arguments and launch the UI."""
    global DEFAULT_MODEL_PATH, DEFAULT_AUDIO_TOKENIZER_PATH, VOICE_PRESETS

    parser = argparse.ArgumentParser(description="Gradio UI for Text-to-Speech using HiggsAudioServeEngine")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the Gradio interface.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio interface.")

    args = parser.parse_args()

    # Update default values if provided via command line
    VOICE_PRESETS = load_voice_presets()

    # Load model on startup
    result = initialize_engine(DEFAULT_MODEL_PATH, DEFAULT_AUDIO_TOKENIZER_PATH)

    # Exit if model loading failed
    if not result:
        logger.error("Failed to load model. Exiting.")
        return

    logger.info(f"Model loaded: {DEFAULT_MODEL_PATH}")

    # Create and launch the UI
    demo = create_ui()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
