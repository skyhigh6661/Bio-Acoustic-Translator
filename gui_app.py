import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile
import sounddevice as sd
from model_engine import ModelEngine

# --- UI Color Scheme ---
COLOR_BG = "#f4f6f9"
COLOR_HEADER = "#2c3e50"
COLOR_CARD = "#ffffff"
COLOR_TEXT = "#34495e"
COLOR_SUBTEXT = "#7f8c8d"

# Plot Colors
COLOR_WAVE = "#3498db"  # Blue for Input
COLOR_SPEC = "#e67e22"  # Orange for Output

# Button Colors
COLOR_BTN_GEN = "#34495e"
COLOR_BTN_PLAY = "#27ae60"
COLOR_BTN_STOP = "#e74c3c"

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SUBTITLE = ("Segoe UI", 12, "bold")
FONT_LABEL = ("Segoe UI", 10)
FONT_BTN = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 10)


class BioTranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bio-Acoustic Translator")
        self.geometry("1280x860")
        self.configure(bg=COLOR_BG)

        self.engine = ModelEngine()
        self.ui_tree = {}
        self.last_wave = None
        self.last_sr = 22050

        self.setup_styles()
        self.create_ui()

        threading.Thread(target=self.run_init_task, daemon=True).start()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", padding=6, font=FONT_LABEL, borderwidth=0)
        style.map("TCombobox", fieldbackground=[('readonly', '#ecf0f1')])
        style.configure("TLabel", background=COLOR_CARD, foreground=COLOR_TEXT, font=FONT_LABEL)

    def run_init_task(self):
        self.status_var.set("Initializing database...")
        sample_db = self.engine.initialize_system()
        self.after(0, lambda: self.update_ui_state(sample_db))

    def update_ui_state(self, sample_db):
        self.ui_tree = {}
        if sample_db:
            for (sp, ctx) in sample_db.keys():
                if sp not in self.ui_tree:
                    self.ui_tree[sp] = set()
                self.ui_tree[sp].add(ctx)

            sp_list = sorted(list(self.ui_tree.keys()))
            self.combo_sp['values'] = sp_list
            if sp_list:
                self.combo_sp.current(0)
                self.on_species_change(None)

            self.btn_gen.config(state=tk.NORMAL, bg=COLOR_BTN_GEN)
            self.status_var.set(f"System Ready. Indexed {len(sample_db)} behaviors.")
        else:
            self.status_var.set("No data found.")

    def on_species_change(self, event):
        sp = self.combo_sp.get()
        if sp in self.ui_tree:
            ctx_list = sorted(list(self.ui_tree[sp]))
            self.combo_ctx['values'] = ctx_list
            if ctx_list:
                self.combo_ctx.current(0)
            else:
                self.combo_ctx.set('')

    # === Core Business Logic ===
    def on_upload(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if not path: return
        self.status_var.set(f"Analyzing: {path}")
        self.update()

        res = self.engine.predict_audio(path)
        if res and 'Error' not in res:
            f0_val = res['Features']['f0_mean']
            f0_str = f"{f0_val:.1f} Hz" if f0_val > 0 else "Unvoiced"

            self.update_text_result(res['Species'], res['Context'], res['Emotion'], f0_str,
                                    res['Features']['centroid_mean'])

            y, sr = res['Audio'], res['SR']
            self.plot_waveform(self.ax_l[0], y, sr, "Waveform", color=COLOR_WAVE)
            self.plot_spectrum(self.ax_l[1], y, sr, "Frequency Spectrum (FFT)", color=COLOR_WAVE)

            self.canvas_l.draw()
            self.status_var.set("Analysis Complete.")

    def on_generate(self):
        sp = self.combo_sp.get()
        ctx = self.combo_ctx.get()

        if not sp or not ctx: return

        self.status_var.set(f"Synthesizing: {sp} - {ctx}...")

        wave, sr = self.engine.generate_audio(sp, ctx)

        if wave is not None:
            self.last_wave = wave
            self.last_sr = sr

            self.btn_save.config(state=tk.NORMAL, bg="#95a5a6")
            self.btn_replay.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.NORMAL)

            self.play_audio()

            self.ax_r[0].clear();
            self.ax_r[1].clear()
            self.plot_waveform(self.ax_r[0], wave, sr, "Synthesized Waveform", color=COLOR_SPEC)
            self.plot_spectrum(self.ax_r[1], wave, sr, "Frequency Spectrum (FFT)", color=COLOR_SPEC)

            self.canvas_r.draw()
            self.status_var.set("Synthesis Complete.")
        else:
            messagebox.showerror("Data Error", f"No recording found for [{sp}] in [{ctx}] context.")

    def update_text_result(self, sp, ctx, emo, f0, bright):
        self.txt_res.config(state='normal')
        self.txt_res.delete("1.0", tk.END)
        content = (
            f"Species    : {sp}\n"
            f"Behavior   : {ctx}\n"
            f"Emotion    : {emo}\n"
            f"----------------------------------------\n"
            f"Fund. Freq : {f0}\n"
            f"Brightness : {bright:.0f} (Spectral Centroid)"
        )
        self.txt_res.insert(tk.END, content)
        self.txt_res.config(state='disabled')

        # === Plotting Functions ===

    def plot_waveform(self, ax, y, sr, title, color):
        ax.clear()
        t = np.linspace(0, len(y) / sr, len(y))
        ax.plot(t, y, color=color, lw=0.8)

        ax.set_title(title, fontsize=9, color=COLOR_SUBTEXT, pad=5)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.margins(x=0)

    def plot_spectrum(self, ax, y, sr, title, color):
        ax.clear()
        mag = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)

        ax.plot(freqs, mag, color=color, lw=0.8)
        ax.set_xlim(0, 4000)

        ax.set_title(title, fontsize=9, color=COLOR_SUBTEXT, pad=5)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Magnitude", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

    # === Audio Control ===
    def play_audio(self):
        if self.last_wave is not None:
            sd.stop()
            sd.play(self.last_wave, self.last_sr)

    def stop_audio(self):
        sd.stop()

    def on_save(self):
        if self.last_wave is None: return
        path = filedialog.asksaveasfilename(defaultextension=".wav")
        if path: wavfile.write(path, self.last_sr, self.last_wave.astype(np.float32))

    def on_retrain(self):
        if messagebox.askyesno("Reload", "Rescan all audio files?"):
            self.engine.force_retrain()
            self.run_init_task()

    # === Layout ===
    def create_ui(self):
        # Header
        header = tk.Frame(self, bg=COLOR_HEADER, height=60)
        header.pack(fill=tk.X)

        title_frame = tk.Frame(header, bg=COLOR_HEADER)
        title_frame.pack(side=tk.LEFT, padx=20, pady=15)
        tk.Label(title_frame, text="Bio-Acoustic Translator", font=FONT_TITLE, fg="white", bg=COLOR_HEADER).pack(
            anchor="w")

        tk.Button(header, text="Reload DB", command=self.on_retrain, bg="#34495e", fg="white", font=("Segoe UI", 9),
                  relief="flat", padx=10).pack(side=tk.RIGHT, padx=20)

        container = tk.Frame(self, bg=COLOR_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Card
        left_card = tk.Frame(container, bg=COLOR_CARD)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        tk.Label(left_card, text=" Acoustic Analysis ", font=FONT_SUBTITLE, bg=COLOR_CARD, fg=COLOR_HEADER,
                 anchor="w").pack(fill=tk.X, padx=20, pady=(15, 10))

        lc = tk.Frame(left_card, bg=COLOR_CARD, padx=15)
        lc.pack(fill=tk.BOTH, expand=True)
        tk.Button(lc, text="📁 Select Audio File", command=self.on_upload, bg="#3498db", fg="white", font=FONT_BTN,
                  relief="flat", height=2).pack(fill=tk.X)

        self.txt_res = tk.Text(lc, height=7, width=40, font=FONT_MONO, bg="#f8f9fa", fg=COLOR_TEXT, relief="flat",
                               state="disabled")
        self.txt_res.pack(fill=tk.X, pady=15)

        self.fig_l, self.ax_l = plt.subplots(2, 1, figsize=(5, 4))
        self.fig_l.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, hspace=0.6)
        self.canvas_l = FigureCanvasTkAgg(self.fig_l, lc)
        self.canvas_l.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right Card
        right_card = tk.Frame(container, bg=COLOR_CARD)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        tk.Label(right_card, text=" Sound Synthesis ", font=FONT_SUBTITLE, bg=COLOR_CARD, fg=COLOR_HEADER,
                 anchor="w").pack(fill=tk.X, padx=20, pady=(15, 10))

        rc = tk.Frame(right_card, bg=COLOR_CARD, padx=15)
        rc.pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(rc, bg=COLOR_CARD)
        ctrl.pack(fill=tk.X)
        tk.Label(ctrl, text="Target Species", font=("Segoe UI", 9, "bold"), fg=COLOR_SUBTEXT).pack(anchor="w")
        self.combo_sp = ttk.Combobox(ctrl, state="readonly")
        self.combo_sp.pack(fill=tk.X, pady=(2, 10))
        self.combo_sp.bind("<<ComboboxSelected>>", self.on_species_change)

        tk.Label(ctrl, text="Behavior Context", font=("Segoe UI", 9, "bold"), fg=COLOR_SUBTEXT).pack(anchor="w")
        self.combo_ctx = ttk.Combobox(ctrl, state="readonly")
        self.combo_ctx.pack(fill=tk.X, pady=(2, 15))

        btn_frame = tk.Frame(rc, bg=COLOR_CARD)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        self.btn_gen = tk.Button(btn_frame, text="Generate Audio", command=self.on_generate, bg="#34495e", fg="white",
                                 font=FONT_BTN, relief="flat", height=2, state=tk.DISABLED)
        self.btn_gen.pack(fill=tk.X)

        c_btns = tk.Frame(btn_frame, bg=COLOR_CARD)
        c_btns.pack(fill=tk.X, pady=(5, 0))
        self.btn_replay = tk.Button(c_btns, text="▶ Play", command=self.play_audio, bg="#27ae60", fg="white",
                                    font=FONT_BTN, relief="flat", state=tk.DISABLED)
        self.btn_replay.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.btn_stop = tk.Button(c_btns, text="⏹ Stop", command=self.stop_audio, bg="#e74c3c", fg="white",
                                  font=FONT_BTN, relief="flat", state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        self.fig_r, self.ax_r = plt.subplots(2, 1, figsize=(5, 3.5))
        self.fig_r.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, hspace=0.6)
        self.canvas_r = FigureCanvasTkAgg(self.fig_r, rc)
        self.canvas_r.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

        self.btn_save = tk.Button(rc, text="Save to Disk", command=self.on_save, bg="#bdc3c7", fg="white",
                                  relief="flat", state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=(0, 10))

        self.status_var = tk.StringVar(value="System initializing...")
        tk.Label(self, textvariable=self.status_var, bg=COLOR_HEADER, fg="white", anchor="w", padx=15, pady=5).pack(
            side=tk.BOTTOM, fill=tk.X)