# Bio-Acoustic Translator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Librosa](https://img.shields.io/badge/DSP-Librosa-orange?style=flat-square&logo=waves)](https://librosa.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-RandomForest-green?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

**A Full-Duplex Bio-Acoustic Translation System based on Digital Signal Processing (DSP) and Machine Learning.**

This project bridges the communication gap between humans and animals (specifically **Cow**, **Pig**, and **Sheep**) by implementing bidirectional translation: **Acoustic Analysis** (Animal-to-Human) and **Generative Synthesis** (Human-to-Animal).

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage Guide](#-usage-guide)
- [Building Executable](#-building-executable)
- [Technical Methodology](#-technical-methodology)
- [License](#-license)


---

## 📖 Overview

Unlike traditional playback systems, **Bio-Acoustic Translator** utilizes **Real-Data Driven DSP Augmentation**. It does not generate artificial sine waves; instead, it manipulates real biological recordings using Phase Vocoder techniques to produce authentic, non-repetitive animal calls that respect physical acoustic properties.

---

## ✨ Key Features

### 1. Acoustic Analysis (Input)
*   **Precision Feature Extraction**: Utilizes the **YIN Algorithm** for accurate Fundamental Frequency (F0) detection, solving issues with non-periodic biological signals (e.g., grunts).
*   **Unvoiced Detection**: Automatically identifies unvoiced signals via Voicing Rate analysis.
*   **Visualization**: Real-time rendering of Time-Domain Waveforms and Frequency-Domain FFT Spectrums using a professional dual-color scheme.

### 2. Sound Synthesis (Output)
*   **Semantic Hash Anchoring**: A deterministic algorithm that selects the most representative "template" recording from the dataset based on the semantic context, ensuring acoustic distinctiveness between similar behaviors.
*   **DSP Augmentation**: Applies **Phase Vocoder** techniques (Micro Pitch Shift & Time Stretch) to real samples, introducing biological variability while maintaining high fidelity.
*   **Strict Cascading Logic**: Context-aware menus ensure only biologically valid behavior combinations (Species → Context) are selectable.

---

## 📂 System Architecture

```text
BioTranslator/
├── main.py               # Application Entry Point
├── gui_app.py            # Graphical User Interface (MVC View)
├── model_engine.py       # Logic Controller & ML Model (MVC Controller)
├── dsp_core.py           # Signal Processing Algorithms (MVC Model)
├── data_manager.py       # Data Loading & Parsing
├── requirements.txt      # Dependency List
└── SoundsDatabase/       # (External) Audio Dataset
```

---

## ⚙️ Installation

### Prerequisites
*   Python 3.8 or higher
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Bio-Acoustic-Translator.git
cd Bio-Acoustic-Translator
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Data Preparation 

This project requires an external dataset to function. The data is **not included** in this repository due to size constraints.

1.  Download the dataset from **[Zenodo Record 14636641](https://zenodo.org/records/14636641)**.
2.  Create a folder named `SoundsDatabase` in the root directory of this project.
3.  Extract all `.wav` files into `SoundsDatabase/`.

**Directory check:**
```text
BioTranslator/
  ├── main.py
  └── SoundsDatabase/
      ├── 9036-Cow-FullSeparation-Negative-2581.wav
      └── ...
```

### ⚠️ Data Filtering Note
To ensure acoustic distinctiveness and model accuracy, the system is currently configured to **only process** the following species:
*   **Goat**
*   **Pig**
*   **Wild Boar**

*(Other species in the dataset will be automatically skipped during the scanning process via `data_manager.py`.)*

---

## 🚀 Usage Guide

Run the application:
```bash
python main.py
```
### Mode 1: Analysis (Left Panel)
1.Click "Select Audio File".
2.Choose a valid .wav file (e.g., from the dataset).
3.The system will output the predicted Species, Behavior, and Emotion, along with F0 and Spectral Centroid metrics.
4.View the real-time Waveform (Blue) and FFT Spectrum (Blue).
### Mode 2: Synthesis (Right Panel)
1.Select a Target Species (e.g., Cow).
2.Select a Behavior Context (e.g., Separation).
3.Click "Generate Audio".
4.The system will synthesize a unique call and play it automatically.
5.View the Synthesized Waveform (Green) and FFT Spectrum (Orange).
6.Use "Save to Disk" to export the result.

---

## 🔨 Building Executable

To compile the project into a standalone Windows `.exe` folder (using PyInstaller):

```bash
python -m PyInstaller --noconsole --onedir --name="BioTranslator" --collect-all="librosa" --collect-all="sklearn" --hidden-import="librosa.display" main.py
```
**Note:** After building, navigate to dist/BioTranslator/ and manually copy the SoundsDatabase folder into it. The executable requires the audio files to function.

---

## 🔬 Technical Methodology

### Signal Processing (DSP)
*   **F0 Extraction**: Utilizes the **YIN algorithm** (`librosa.yin`) instead of simple FFT peak detection. This provides robust fundamental frequency estimation for non-periodic biological signals (e.g., grunts).
*   **Synthesis**: Implements **Phase Vocoder** techniques via STFT (Short-Time Fourier Transform). This allows for independent pitch shifting and time-scale modification to generate realistic variations without altering the signal's core texture.

### Machine Learning
*   **Classifier**: Random Forest Ensemble (100 estimators).
*   **Feature Vector**: 
    *   **Spectral Centroid** (Brightness/Timbre)
    *   **Spectral Bandwidth**
    *   **Zero-Crossing Rate (ZCR)** (Roughness/Noise level)
    *   **MFCCs** (13 coefficients)
    *   **Voicing Probability** (Distinguishes tonal vs. noise signals)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Acknowledgments**:
*   Dataset provided by [Zenodo](https://zenodo.org/records/14636641).
*   Powered by [Librosa](https://librosa.org/) and [Scikit-learn](https://scikit-learn.org/).