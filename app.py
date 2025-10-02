import spaces
import sys
sys.path.append("neutts-air")
from neuttsair.neutts import NeuTTSAir
import gradio as gr

# load model
tts = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu"
)

@spaces.GPU()
def infer(ref_text, ref_audio_path, gen_text):

    gr.Info("Starting inference request!")
    gr.Info("Encoding reference...")
    ref_codes = tts.encode_reference(ref_audio_path)

    gr.Info(f"Generating audio for input text: {input_text}")
    wav = tts.infer(input_text, ref_codes, ref_text)

    return (24_000, wav)

demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(label="Reference Text"),
        gr.Audio(source="upload", type="filepath", label="Reference Audio"),
        gr.Textbox(label="Text to Generate"),
    ],
    outputs=gr.Audio(type="numpy", label="Generated Speech"),
    title="NeuTTS-Air",
    description="Upload a reference audio sample, provide the reference text, and enter new text to synthesize."
)

if __name__ == "__main__":
    demo.launch()