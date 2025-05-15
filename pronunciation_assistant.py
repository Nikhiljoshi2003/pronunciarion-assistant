
import torch
import streamlit as st
import torchaudio
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

# Streamlit interface
st.title("Pronunciation Assistant")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    waveform, sample_rate = torchaudio.load("temp.wav")
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate).to(device)
        waveform = resampler(waveform)

    with torch.inference_mode():
        emissions, _ = model(waveform)
    transcripts = bundle.decode(emissions)
    st.markdown("**Transcription:**")
    st.write(transcripts[0])
