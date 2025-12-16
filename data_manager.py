import os
import glob
import pandas as pd
import warnings
import librosa


class DataManager:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        # Filter strictly for these distinct species
        self.ALLOWED_SPECIES = ["Goat", "Pig", "WildBoar"]

    def parse_filename(self, filepath):
        """
        Parse metadata from filename structure: ID-Species-Context-Emotion-ID.wav
        """
        base = os.path.basename(filepath)
        parts = base.replace('.wav', '').split('-')

        # Ensure filename has enough segments
        if len(parts) >= 4:
            species = parts[1]

            # Automatic Typo Correction
            if species == "WidlBoar": species = "WildBoar"

            # Filter non-target species
            if species not in self.ALLOWED_SPECIES:
                return None

            return {
                'Species': species,
                'Context': parts[2],
                'Emotion': parts[3],
                'Filepath': filepath
            }
        return None

    def scan_folder(self, dsp_instance):
        """
        Scan the target folder and extract features for valid files.
        """
        if not os.path.exists(self.data_folder):
            return pd.DataFrame()

        wav_files = glob.glob(os.path.join(self.data_folder, "*.wav"))
        data_list = []

        print(f"[Data Scanner] Scanning {len(wav_files)} files for {self.ALLOWED_SPECIES}...")

        count = 0
        for idx, f in enumerate(wav_files):
            meta = self.parse_filename(f)

            if meta:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y, sr = librosa.load(f, sr=dsp_instance.sr)
                        feats = dsp_instance.extract_features(y, sr)

                    if feats:
                        row = {**meta, **feats}
                        data_list.append(row)
                        count += 1
                except Exception:
                    pass

            if (idx + 1) % 50 == 0:
                print(f"Progress: {idx + 1}/{len(wav_files)} | Valid Samples: {count}")

        print(f"[Complete] Extracted {count} valid samples.")
        return pd.DataFrame(data_list)