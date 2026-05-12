# 🐋 Bio-Acoustic-Cognitive-Spectrum

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Repository Status](https://img.shields.io/badge/status-Active-brightgreen.svg)]()

A cutting-edge **hybrid AI system** combining **Convolutional Neural Networks (CNN)** and **Deep Q-Network (DQN) Reinforcement Learning** to detect and intelligently process marine mammal acoustic signals in complex underwater environments.

## 📋 Overview

Bio-Acoustic-Cognitive-Spectrum is a research framework designed to solve the critical challenge of identifying biological acoustic signals (whale, dolphin, and other marine mammal vocalizations) amidst noisy underwater environments. The system employs:

- **CNN Acoustic Classifier**: Extracts features from mel-spectrograms to classify biological vs. non-biological signals
- **DQN Decision Agent**: Dynamically selects optimal hydrophone channels and environmental response strategies based on real-time acoustic state
- **Interactive Simulation**: Provides an intuitive Streamlit-based interface for testing multi-channel acoustic scenarios

### Key Innovation

Unlike traditional acoustic detection systems, this framework incorporates **cognitive decision-making** through reinforcement learning. The DQN agent learns to optimize channel selection and interference mitigation strategies, enabling adaptive acoustic monitoring in dynamic marine environments.

---

## ✨ Features

### 🎯 Core Capabilities

- **Dual-Model Architecture**
  - CNN for robust biological signal classification
  - DQN for intelligent channel selection and environmental response
  
- **Multi-Channel Simulation**
  - Simulates 5 concurrent hydrophone channels
  - Real-time environmental variable injection (noise, vessel traffic)
  - Dynamic state vector generation for reinforcement learning

- **Advanced Audio Processing**
  - Mel-spectrogram feature extraction
  - Acoustic noise injection and mitigation
  - Vessel traffic interference modeling
  - Automatic spectrogram normalization

- **Interactive Web Application**
  - Tabbed interface for different analysis modes
  - Real-time model inference with PyTorch
  - Visual spectrograms and confidence metrics
  - Agent decision transparency and reward tracking

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sanganisathwik/Bio-Acoustic-Cognitive-Spectrum.git
   cd Bio-Acoustic-Cognitive-Spectrum
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Required packages:
   - `streamlit` - Web application framework
   - `torch` - Deep learning framework
   - `librosa` - Audio processing library
   - `numpy` - Numerical computing
   - `matplotlib` - Visualization
   - `soundfile` - Audio file I/O

3. **Prepare data**
   ```bash
   python setup_data.py
   ```
   This script:
   - Organizes biological and non-biological audio samples
   - Resamples audio to 16 kHz
   - Structures data into `/data/bio` and `/data/nonbio` directories

4. **Launch the application**
   ```bash
   streamlit run demo_app.py
   ```
   Open your browser to `http://localhost:8501`

---

## 📖 Usage Guide

### Mode 1: 5-Channel Array Simulator

Simulate a realistic multi-hydrophone setup with environmental interference:

1. **Configure each of the 5 channels:**
   - Select signal type: Random, Bio, Non-Bio, or Custom
   - Set acoustic noise level (0.0 - 1.0)
   - Set vessel traffic interference (0.0 - 1.0)

2. **Run simulation:**
   - Click "🚀 Run 5-Channel Simulation"
   - System processes all channels through the CNN pipeline
   - DQN agent analyzes state and selects optimal channel
   - View per-channel spectrograms and CNN predictions
   - Inspect DQN Q-values for channel selection reasoning

3. **Interpret results:**
   - 🎯 **AGENT SELECTED**: Highest Q-value channel
   - CNN confidence scores for each channel
   - Environmental noise and traffic metrics

### Mode 2: Single File Deep Analysis

Analyze individual audio files with detailed environmental control:

1. **Upload an underwater audio file** (.wav format)

2. **Configure environment:**
   - Select active transmission channel (20-100 kHz range labels)
   - Adjust vessel traffic interference
   - Add acoustic white noise

3. **Analyze:**
   - View audio waveform
   - Inspect mel-spectrogram
   - Get CNN classification (Bio-Acoustic or Ambient Noise)
   - See confidence percentage
   - Review RL agent decision and reward status

---

## 🏗️ Project Structure

```
Bio-Acoustic-Cognitive-Spectrum/
├── demo_app.py              # Main Streamlit application
├── setup_data.py            # Data preparation and organization script
├── check_dup.py             # Duplicate detection utility
├── remove_dups.py           # Duplicate removal utility
├── data/                    # Data directory (created by setup_data.py)
│   ├── bio/                 # Biological acoustic samples
│   ├── nonbio/              # Non-biological samples (ambient noise, traffic)
│   ├── cnn_model.pth        # Trained CNN weights
│   └── dqn_model.pth        # Trained DQN weights
├── src/
│   └── cog_net/             # Cognitive network module
│       ├── cnn.py           # CNN architecture and utilities
│       ├── dqn.py           # DQN architecture and utilities
│       └── audio_utils.py   # Audio processing functions
├── .gitignore               # Git ignore file
└── README.md                # This file
```

---

## 🧠 Technical Architecture

### CNN Acoustic Classifier

**Input**: Mel-spectrogram (128 frequency bins × temporal frames)  
**Output**: Binary classification (0: Non-Bio, 1: Bio-Acoustic)  
**Purpose**: Extract and classify acoustic features from hydrophone signals

Key processing steps:
- Audio resampling to 16 kHz
- Mel-spectrogram extraction (128 bins, 1024 FFT, 512 hop length)
- Power-to-dB conversion and normalization
- Convolution-based feature extraction
- Sigmoid activation for binary classification

### DQN Reinforcement Learning Agent

**State Space**: 15 dimensions (5 channels × 3 features each)
- Channel noise level, CNN bio-detection, vessel traffic per channel

**Action Space**: 5 discrete actions (channel selection)
- Actions 0-4 correspond to hydrophone channels

**Reward Signals**:
- +126: Optimal biological signal detection on clear channel
- +115: Emergency mitigation of extreme interference
- +100: Successful interference filtering
- +50: Gain adjustment for clarity
- +10: Clear channel communication
- +5: Traffic avoidance (idle state)

**Purpose**: Learn optimal channel selection policies and interference response strategies

---

## 📊 Model Training

The system uses pre-trained models (`cnn_model.pth` and `dqn_model.pth`) included in the data directory. 

**Training data characteristics:**
- **Biological samples**: ~3,500 marine mammal vocalizations
- **Non-biological samples**: ~1,500 ambient noise and vessel traffic recordings
- **Sample rate**: 16 kHz (mono)
- **Species coverage**: Diverse marine mammals (whales, dolphins, etc.)

### To retrain models (if you have training data):

This would require modifications to the main architecture. See the source files in `/src/cog_net/` for model definitions.

---

## 🛠️ Utilities

### check_dup.py
Identifies duplicate audio files in the dataset:
```bash
python check_dup.py
```

### remove_dups.py
Removes identified duplicates from the dataset:
```bash
python remove_dups.py
```

---

## 🔧 Customization

### Modify CNN Input Size
Edit `/src/cog_net/audio_utils.py` to change spectrogram dimensions:
```python
TARGET_SIZE = (128, 250)  # (freq_bins, time_steps)
```

### Adjust DQN Architecture
Modify `/src/cog_net/dqn.py` to change network depth or capacity:
```python
def create_model(state_size, action_size):
    model = DQNNet(state_size, action_size)
    # Customize layer sizes
    return model
```

### Customize Reward Function
Edit `/src/cog_net/dqn.py` to redefine reward signals based on your use case.

---

## 📈 Performance Metrics

The system achieves:
- **CNN Classification Accuracy**: Binary classification of bio vs. non-bio signals
- **DQN Convergence**: Learns optimal channel selection within training episodes
- **Real-time Inference**: Processes 5-channel audio streams in seconds

*Note: Specific performance metrics depend on training data and hyperparameters.*

---

## 🐳 Use Cases

1. **Marine Mammal Conservation**: Monitor whale/dolphin presence in protected waters
2. **Acoustic Monitoring Networks**: Deploy multi-hydrophone systems for environmental sensing
3. **Underwater Communication**: Intelligently manage acoustic channels in noisy marine environments
4. **Research**: Study bio-acoustic patterns while filtering anthropogenic noise
5. **Cognitive Robotics**: Test adaptive decision-making in dynamic sensor networks

---

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `demo_app.py` | Interactive Streamlit web application for model inference and visualization |
| `setup_data.py` | Organizes raw audio data into structured directories with resampling |
| `check_dup.py` | Identifies duplicate audio files using audio fingerprinting |
| `remove_dups.py` | Removes duplicates from the dataset |
| `src/cog_net/cnn.py` | CNN model definition and utilities |
| `src/cog_net/dqn.py` | DQN model definition and Q-value computation |
| `src/cog_net/audio_utils.py` | Audio processing functions (spectrograms, normalization, etc.) |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Audio Processing**: Built with [librosa](https://librosa.org/) for robust audio feature extraction
- **Deep Learning**: Powered by [PyTorch](https://pytorch.org/) for efficient model training and inference
- **Web Framework**: Interactive interface built with [Streamlit](https://streamlit.io/)
- **Marine Acoustics**: Inspired by research in bio-acoustic signal processing and marine mammal monitoring

---

## 📧 Contact & Support

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Report bugs or request features](https://github.com/Sanganisathwik/Bio-Acoustic-Cognitive-Spectrum/issues)
- **Author**: [Sanganisathwik](https://github.com/Sanganisathwik)

---

## 🔮 Future Roadmap

- [ ] GPU acceleration support for real-time multi-channel processing
- [ ] Extended species classification (beyond binary bio/non-bio)
- [ ] Temporal attention mechanisms for improved sequence modeling
- [ ] REST API for integration with IoT acoustic sensor networks
- [ ] Multi-modal fusion (acoustic + visual data)
- [ ] Transfer learning with domain adaptation
- [ ] Edge deployment for autonomous underwater vehicles (AUVs)

---

**Last Updated**: May 2026  
**Repository**: [Bio-Acoustic-Cognitive-Spectrum](https://github.com/Sanganisathwik/Bio-Acoustic-Cognitive-Spectrum)

---

<div align="center">

🐋 **Advancing Marine Mammal Acoustic Monitoring Through Intelligent AI** 🐋

*Bridging the gap between acoustic ecology and reinforcement learning*

</div>
