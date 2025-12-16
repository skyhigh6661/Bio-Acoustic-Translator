import os
import pandas as pd
import librosa
import joblib
import zlib
from sklearn.ensemble import RandomForestClassifier
from dsp_core import DSPCore
from data_manager import DataManager


class ModelEngine:
    def __init__(self, data_path="SoundsDatabase"):
        self.dsp = DSPCore()
        self.data_mgr = DataManager(data_path)
        self.model_file = "bio_translator_model.pkl"

        self.clf_species = RandomForestClassifier(n_estimators=100)
        self.clf_context = RandomForestClassifier(n_estimators=100)
        self.clf_emotion = RandomForestClassifier(n_estimators=100)

        # Stores the chosen template path for each (Species, Context) pair
        self.template_db = {}
        self.feature_cols = []
        self.is_trained = False

    def initialize_system(self):
        # 1. Attempt to load cached model
        if os.path.exists(self.model_file):
            print("[System] Loading cache...")
            try:
                ckpt = joblib.load(self.model_file)
                self.clf_species = ckpt['clf_species']
                self.clf_context = ckpt['clf_context']
                self.clf_emotion = ckpt['clf_emotion']
                self.template_db = ckpt['template_db']
                self.feature_cols = ckpt['feature_cols']
                self.is_trained = True
                return self.template_db
            except:
                print("Cache corrupted. Retraining...")

        # 2. Full scan if no cache
        print("[System] Scanning audio files...")
        df = self.data_mgr.scan_folder(self.dsp)
        if df.empty: return None

        self.feature_cols = [c for c in df.columns if c.startswith('mfcc') or c.endswith('_mean') or c == 'duration']

        # 3. Semantic Hash Anchoring
        # Selects a deterministic template based on the spectral centroid distribution
        print("[System] Selecting templates via Semantic Hashing...")
        self.template_db = {}

        grouped = df.groupby(['Species', 'Context'])

        for key, group in grouped:
            # key: ('Horse', 'GroupReunion')
            context_name = str(key[1])

            if len(group) == 1:
                self.template_db[key] = group.iloc[0]['Filepath']
                continue

            # Sort group by brightness (Spectral Centroid)
            sorted_group = group.sort_values(by='centroid_mean')

            # Map context string to a percentage (0-99)
            hash_val = zlib.crc32(context_name.encode('utf-8')) % 100

            # Select the sample at the hashed position
            target_idx = int((hash_val / 100.0) * (len(group) - 1))
            best_filepath = sorted_group.iloc[target_idx]['Filepath']

            self.template_db[key] = best_filepath

        # 4. Train Random Forest Classifiers
        X = df[self.feature_cols].fillna(0)
        self.clf_species.fit(X, df['Species'])
        self.clf_context.fit(X, df['Context'])
        self.clf_emotion.fit(X, df['Emotion'])

        self.is_trained = True

        # Save cache
        joblib.dump({
            'clf_species': self.clf_species,
            'clf_context': self.clf_context,
            'clf_emotion': self.clf_emotion,
            'template_db': self.template_db,
            'feature_cols': self.feature_cols
        }, self.model_file)

        return self.template_db

    def force_retrain(self):
        if os.path.exists(self.model_file): os.remove(self.model_file)
        return self.initialize_system()

    def predict_audio(self, filepath):
        if not self.is_trained: return {'Error': 'Model not trained'}
        y, sr = librosa.load(filepath, sr=self.dsp.sr)
        feats = self.dsp.extract_features(y, sr)
        if not feats: return None

        df_in = pd.DataFrame([feats])
        for col in self.feature_cols:
            if col not in df_in.columns: df_in[col] = 0
        df_in = df_in[self.feature_cols]

        return {
            'Species': self.clf_species.predict(df_in)[0],
            'Context': self.clf_context.predict(df_in)[0],
            'Emotion': self.clf_emotion.predict(df_in)[0],
            'Features': feats,
            'Audio': y, 'SR': sr
        }

    def generate_audio(self, sp, ctx):
        """
        Reverse Synthesis Entry Point.
        Retrieves the template and applies DSP augmentation.
        """
        key = (sp, ctx)
        template_path = self.template_db.get(key)
        if not template_path: return None, None
        return self.dsp.variate_sample(template_path)