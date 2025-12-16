import numpy as np
import librosa
import random


class DSPCore:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def extract_features(self, y, sr=None):
        """
        Extract rigorous acoustic features from the audio signal.
        Includes F0 (YIN), Spectral Centroid, Bandwidth, ZCR, and MFCCs.
        """
        if sr is None: sr = self.sr

        # Pre-processing: Trim silence with a top_db threshold
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # Ignore extremely short audio clips
        if len(y_trimmed) < sr * 0.05: return None

        # Basic spectral features
        cent = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)

        # Advanced F0 extraction using the YIN algorithm
        try:
            f0_series = librosa.yin(y_trimmed, fmin=50, fmax=2000, sr=sr)
            # Filter out harmonic errors
            f0_series = f0_series[f0_series > 50]

            if len(f0_series) > 0:
                avg_f0 = np.mean(f0_series)
            else:
                avg_f0 = 0.0
        except:
            avg_f0 = 0.0

        features = {
            'f0_mean': float(avg_f0),
            'centroid_mean': np.mean(cent),
            'bandwidth_mean': np.mean(bw),
            'zcr_mean': np.mean(zcr),
            'duration': librosa.get_duration(y=y_trimmed, sr=sr)
        }

        # MFCC extraction for timbre classification
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        for i in range(13): features[f'mfcc_{i + 1}'] = np.mean(mfcc[i])

        return features

    def variate_sample(self, original_path):
        """
        Single-shot Synthesis via DSP Augmentation.
        Applies micro-modulations to pitch and time to simulate biological variability.
        """
        try:
            # Load the original raw audio
            y, sr = librosa.load(original_path, sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)

            # 1. Micro Pitch Shifting: +/- 0.1 semitones
            n_steps = random.uniform(-0.1, 0.1)
            try:
                y_shifted = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
            except:
                y_shifted = y

            # 2. Micro Time Stretching: +/- 2%
            rate = random.uniform(0.98, 1.02)
            y_stretched = librosa.effects.time_stretch(y_shifted, rate=rate)

            # 3. Low-level Noise Injection for realism
            noise_amp = 0.0005 * np.max(np.abs(y_stretched))
            noise = np.random.normal(0, noise_amp, len(y_stretched))
            y_final = y_stretched + noise

            # 4. Fade In/Out to prevent clicking artifacts
            fade_samples = int(0.01 * self.sr)
            if len(y_final) > 2 * fade_samples:
                y_final[:fade_samples] *= np.linspace(0, 1, fade_samples)
                y_final[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            # Normalization
            if np.max(np.abs(y_final)) > 0:
                y_final = y_final / np.max(np.abs(y_final))

            return y_final, sr

        except Exception as e:
            print(f"[DSP Error] {e}")
            # Return silence on failure
            return np.zeros(int(self.sr * 0.5)), self.sr