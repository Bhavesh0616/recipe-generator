import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

def record_audio(duration=5, fs=16000, filename="temp.wav"):
    print("Recording for", duration, "seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavfile.write(filename, fs, recording)
    print("Recording saved to", filename)
    return filename

def transcribe_audio(filename):
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)
    
    result = asr(filename)
    print("Transcription:", result['text'])

if __name__ == "__main__":
    audio_file = record_audio(duration=5)  # Record 5 seconds from mic
    transcribe_audio(audio_file)
