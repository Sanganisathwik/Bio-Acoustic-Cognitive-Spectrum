import streamlit as st
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "cog_net"))
from cnn import create_cnn
from audio_utils import fix_size
from dqn import create_model as create_dqn

st.set_page_config(layout="wide", page_title="Marine Mammal Acoustic Intelligence")

# --- 1. Load the Models ---
@st.cache_resource
def load_models():
    # Load CNN Acoustic model
    cnn_model = create_cnn()
    cnn_model.load_state_dict(torch.load("data/cnn_model.pth", map_location=torch.device('cpu')))
    cnn_model.eval()
    
    # Load Reinforcement DQN model
    dqn_model = create_dqn(15, 5)
    dqn_model.load_state_dict(torch.load("data/dqn_model.pth", map_location=torch.device('cpu')))
    dqn_model.eval()
    
    return cnn_model, dqn_model

cnn_model, dqn_model = load_models()

# Retrieve paths to actual audio files
@st.cache_data
def get_sample_files():
    base_dir = r"C:\Users\sanga\Downloads\SATHWIK\Documents\src\data"
    bio_files = glob.glob(os.path.join(base_dir, "bio", "*.wav"))
    nonbio_files = glob.glob(os.path.join(base_dir, "nonbio", "*.wav"))
    return bio_files, nonbio_files

bio_files, nonbio_files = get_sample_files()

# --- 2. UI Layout ---
st.title("🐋 Marine Mammal Acoustic Intelligence")
st.markdown("### CNN + Reinforcement Learning Detection System")

# Implement Tabbed View
tab1, tab2 = st.tabs(["🎛️ 5-ChannelRL Array Simulation", "🎙️ Single File Analysis"])

with tab1:
    st.markdown("## 5-Channel Array Simulator")
    st.write("Simulates input from 5 physical hydrophone channels, runs each through the Acoustic CNN, and feeds the resulting state to the DQN Agent to make a dynamic channel selection decision.")
    
    st.markdown("#### Per-Channel Configuration")
    config_cols = st.columns(5)
    
    ch_configs = []
    for ch in range(5):
        with config_cols[ch]:
            st.markdown(f"**Ch {ch} Specs**")
            is_bio = st.selectbox("Signal Data", ["Random", "Bio", "Non-Bio", "Custom"], index=0, key=f"bio_{ch}")
            custom_f = st.file_uploader("Upload .wav", type=["wav"], key=f"file_{ch}") if is_bio == "Custom" else None
            ch_noise = st.slider("Acoustic Noise", 0.0, 1.0, 0.5, step=0.1, key=f"noise_{ch}")
            ch_traffic = st.slider("Vessel Traffic", 0.0, 1.0, 0.2, step=0.1, key=f"traffic_{ch}")
            ch_configs.append({"bio": is_bio, "noise": ch_noise, "traffic": ch_traffic, "custom_file": custom_f})
    
    if st.button("🚀 Run 5-Channel Simulation", type="primary"):
        st.markdown("---")
        
        channels_data = []
        state_flat = [] # Flat vector for RL

        with st.spinner("Recording environment variables and running parallel CNN pipelines..."):
            for ch in range(5):
                # Resolve signal selection
                if ch_configs[ch]["bio"] == "Random":
                    is_bio_true = random.random() < 0.4
                    chosen_file = random.choice(bio_files) if is_bio_true else random.choice(nonbio_files)
                elif ch_configs[ch]["bio"] == "Custom":
                    chosen_file = ch_configs[ch]["custom_file"]
                    if chosen_file is None:
                        st.error(f"Missing file for Channel {ch}. Using random Non-Bio.")
                        chosen_file = random.choice(nonbio_files)
                        is_bio_true = False
                    else:
                        is_bio_true = "Unknown (Custom)"
                else:
                    is_bio_true = (ch_configs[ch]["bio"] == "Bio")
                    chosen_file = random.choice(bio_files) if is_bio_true else random.choice(nonbio_files)
                
                # Assign manual interference values
                noise = ch_configs[ch]["noise"]
                traffic = ch_configs[ch]["traffic"]

                # Process spectro exactly like training
                y, sr = librosa.load(chosen_file, sr=16000)
                
                # Apply Acoustic White Noise
                if noise > 0:
                    y = y + (noise * 0.5) * np.random.randn(len(y))
                    
                # Apply Synthetic Vessel Traffic (Low frequency engine rumble + harmonics)
                if traffic > 0:
                    t = np.linspace(0, len(y)/sr, len(y))
                    engine_rumble = np.sin(2 * np.pi * 45.0 * t) + 0.5 * np.sin(2 * np.pi * 90.0 * t)
                    y = y + (traffic * 0.4) * engine_rumble
                
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
                S_dB = librosa.power_to_db(S, ref=np.max)
                
                spec = fix_size(S_dB)
                spec = np.nan_to_num(spec)
                min_val = spec.min()
                max_val = spec.max()
                if max_val - min_val > 0:
                    spec = (spec - min_val) / (max_val - min_val)
                else:
                    spec = np.zeros_like(spec)
                    
                spec = np.expand_dims(spec, axis=0) # add depth
                input_tensor = torch.FloatTensor(spec).unsqueeze(0) # add batch
                
                with torch.no_grad():
                    pred_value = cnn_model(input_tensor).item()
                
                cnn_bio_detected = 1 if pred_value > 0.5 else 0
                confidence = pred_value if cnn_bio_detected == 1 else (1.0 - pred_value)
                
                # RL expects: noise, bio, traffic sequentially for each of the 5 channels
                state_flat.extend([noise, cnn_bio_detected, traffic])
                
                channels_data.append({
                    "channel": ch,
                    "target_bio": is_bio_true,
                    "noise": noise,
                    "traffic": traffic,
                    "cnn_bio_detected": cnn_bio_detected,
                    "confidence": confidence,
                    "S_dB": S_dB,
                    "sr": sr
                })
        
        # RL Model selects optimal channel 
        state_tensor = torch.FloatTensor(state_flat)
        with torch.no_grad():
            q_values = dqn_model(state_tensor)
            selected_action = torch.argmax(q_values).item()
            
        cols = st.columns(5)
        for i, col in enumerate(cols):
            c_data = channels_data[i]
            with col:
                st.markdown(f"#### Channel {i}")
                
                # Winner highlight
                if i == selected_action:
                    st.success("🎯 **AGENT SELECTED**\n(Highest Q-Value)")
                else:
                    st.info("Passive Stream")
                    
                # Traffic & Noise visualizations
                st.progress(c_data['noise'], text=f"Noise: {c_data['noise']:.2f}")
                st.progress(c_data['traffic'], text=f"Traffic/Vessel: {c_data['traffic']:.2f}")
                
                st.write("")
                # Extracted CNN bio visual
                if c_data['cnn_bio_detected'] == 1:
                    st.markdown(f"🐋 **CNN BIO DETECTED**\n*(Conf: {c_data['confidence']*100:.1f}%)*")
                else:
                    st.markdown(f"🌊 **NON-BIO / NOISE**\n*(Conf: {c_data['confidence']*100:.1f}%)*")
                    
                # Mini Spectrograph plot
                fig, ax = plt.subplots(figsize=(2, 1.5))
                ax.axis('off')
                librosa.display.specshow(c_data['S_dB'], ax=ax)
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")
        st.markdown(f"### 🧠 Neural Output State")
        st.write(f"The Deep Q-Network rapidly evaluated the dynamic state vector from all 5 channels and computed expected rewards.")
        st.bar_chart(np.array(q_values.numpy()), y_label="Expected Reward Q-Value", color="#6a1b9a")


with tab2:
    st.markdown("### Single File Deep Analysis")
    uploaded_file = st.file_uploader("Upload an Underwater Audio Clip (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.sidebar.markdown("### 🌊 Environment Controls")
        selected_channel = st.sidebar.selectbox("Active Transmission Channel", ["CH 0 (20 kHz)", "CH 1 (40 kHz)", "CH 2 (60 kHz)", "CH 3 (80 kHz)", "CH 4 (100 kHz)"])
        traffic_density = st.sidebar.slider("Vessel Traffic Interference", 0.0, 1.0, 0.2)
        noise_level = st.sidebar.slider("Add Acoustic White Noise", 0.0, 1.0, 0.0)
        
        st.sidebar.info(f"Processing Audio on {selected_channel}...")
        y, sr = librosa.load(uploaded_file, sr=16000)
        
        if noise_level > 0:
            y_noisy = y + noise_level * np.random.randn(len(y))
            y = y_noisy
            st.warning(f"⚠️ Applied {noise_level:.2f} Acoustic Noise to {selected_channel}")
        
        if traffic_density > 0.5:
            st.warning(f"⚠️ High Vessel Traffic detected on {selected_channel}. Signal degradation possible.")

        # Show Waveform
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

        # Convert to Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        st.write(f"#### CNN Input: {selected_channel} Mel-Spectrogram")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)

        # Apply identical scaling used in training
        spec = fix_size(S_dB)
        spec = np.nan_to_num(spec)
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 0:
            spec = (spec - min_val) / (max_val - min_val)
        else:
            spec = np.zeros_like(spec)
            
        spec = np.expand_dims(spec, axis=0) # Add channel dim
        input_tensor = torch.FloatTensor(spec).unsqueeze(0) # Add batch dim

        with torch.no_grad():
            pred_value = cnn_model(input_tensor).item()
            
        prediction = 1 if pred_value > 0.5 else 0
        confidence = pred_value if prediction == 1 else (1.0 - pred_value)

        st.markdown("---")
        
        if prediction == 1:
            st.success(f"### 🐋 DETECTION: BIOLOGICAL SIGNAL (Marine Mammal)")
        else:
            st.info(f"### 🌊 DETECTION: NON-BIOLOGICAL (Ambient or Vessel Noise)")

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("CNN Output Class", "BIO-ACOUSTIC" if prediction == 1 else "AMBIENT NOISE")
            st.write(f"Confidence: **{confidence*100:.2f}%**")

        with col2:
            st.write("#### Historical RL Agent Action Map")
            
            if prediction == 1: # Bio detected
                if noise_level > 0.8 or traffic_density > 0.8:
                    current_action = f"Action 4: EMERGENCY ALERT on {selected_channel}"
                    reward_display = "+115 (Mitigating Extreme Interference)"
                    box_func = st.error
                    icon = "🚨"
                elif noise_level > 0.5 or traffic_density > 0.5:
                    current_action = f"Action 3: DE-NOISE / FILTER {selected_channel}"
                    reward_display = "+100 (Filtering Interference)"
                    box_func = st.warning
                    icon = "⚙️"
                elif confidence < 0.80:
                    current_action = f"Action 2: INCREASE GAIN on {selected_channel}"
                    reward_display = "+50 (Seeking Clarity)"
                    box_func = st.warning
                    icon = "🔊"
                else:
                    current_action = f"Action 1: ISOLATE {selected_channel} / PUSH COMM TO SAFE CHANNEL"
                    reward_display = "+126 (Optimal Record)"
                    box_func = st.success
                    icon = "✅"
            else: # No Bio detected
                if traffic_density > 0.7:
                     current_action = f"Action 0: SCANNING. {selected_channel} CONGESTED."
                     reward_display = "+5 (Idle, Traffic Avoidance)"
                     box_func = st.info
                     icon = "⏳"
                else:
                     current_action = f"Action 0: TRANSMITTING ON {selected_channel}"
                     reward_display = "+10 (Clear Channel Transport)"
                     box_func = st.info
                     icon = "📡"

            box_func(f"{icon} {current_action}")
            st.write(f"**Agent Status:** Reward {reward_display}")

st.markdown("---")
st.markdown("*Research Simulator Framework*")
