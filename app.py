import spaces
import os
import sys
sys.path.append("neutts-air")
from neuttsair.neutts import NeuTTSAir
import gradio as gr

SAMPLES_PATH = os.path.join(os.getcwd(), "/neutts-air/samples/")
DEFAULT_REF_TEXT = "So I'm live on radio. And I say, well, my dear friend James here clearly, and the whole room just froze. Turns out I'd completely misspoken and mentioned our other friend." 
DEFAULT_REF_PATH = os.path.join(SAMPLES_PATH, "dave.wav")
DEFAULT_GEN_TEXT = "Hello, I'm NeuTTS-Air! How're you doing today?"

tts = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air",
    backbone_device="cuda",
    codec_repo="neuphonic/neucodec",
    codec_device="cuda"
)

@spaces.GPU()
def infer(ref_text, ref_audio_path, gen_text):

    gr.Info("Starting inference request!")
    gr.Info("Encoding reference...")
    ref_codes = tts.encode_reference(ref_audio_path)

    gr.Info(f"Generating audio for input text: {gen_text}")
    wav = tts.infer(gen_text, ref_codes, ref_text)

    return (24_000, wav)

demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(label="Reference Text", value=DEFAULT_REF_TEXT),
        gr.Audio(type="filepath", label="Reference Audio", value=DEFAULT_REF_PATH),
        gr.Textbox(label="Text to Generate", value=DEFAULT_GEN_TEXT),
    ],
    outputs=gr.Audio(type="numpy", label="Generated Speech"),
    title="NeuTTS-Air☁️",
    description="Upload a reference audio sample, provide the reference text, and enter new text to synthesize."
)

if __name__ == "__main__":
    demo.launch(allowed_paths=[samples_path])