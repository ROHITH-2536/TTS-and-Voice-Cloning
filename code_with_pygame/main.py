import tkinter as tk
from tkinter import ttk, filedialog
import threading
import os
import pygame
import time
import torch
import functools
from TTS.api import TTS
from PIL import Image, ImageTk, ImageDraw
import io
import numpy as np
import librosa
import soundfile as sf

class ModernTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Generator")
        self.root.geometry("800x600")
        self.root.minsize(700, 550)
        self.root.configure(bg="#f5f5f7")
        
        # Set application icon
        self.setup_icon()
        
        # Initialize variables
        self.output_file = "output.wav"
        self.original_output_file = "original_output.wav"  # For pitch modification
        self.is_playing = False
        self.is_paused = False
        self.voice_clone_sample = None  # Store path to voice sample
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        pygame.mixer.init()
        
        # Initialize pitch control variable
        self.pitch_factor = tk.DoubleVar(value=1.0)  # Default pitch (normal)
        
        # Setup theme and styles
        self.setup_styles()
        
        # Create UI components
        self.create_widgets()
        
        # Initialize TTS models
        self.tts_models = {
            "standard": {
                "female": "tts_models/en/ljspeech/vits",
                "male": "tts_models/en/vctk/vits"
            },
            "xtts": "tts_models/multilingual/multi-dataset/xtts_v2"
        }
        
        # Set default TTS mode
        self.tts_mode = tk.StringVar(value="standard")
        
    def setup_icon(self):
        # Create a simple microphone icon
        icon_size = 32
        icon_image = Image.new("RGBA", (icon_size, icon_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon_image)
        
        # Draw microphone
        draw.rectangle((12, 8, 20, 22), fill="#3b82f6")
        draw.ellipse((10, 4, 22, 10), fill="#3b82f6")
        draw.rectangle((8, 22, 24, 24), fill="#3b82f6")
        draw.rectangle((14, 24, 18, 28), fill="#3b82f6")
        
        # Convert to PhotoImage
        icon_photo = ImageTk.PhotoImage(icon_image)
        self.root.iconphoto(True, icon_photo)
        
    def setup_styles(self):
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')  # Use a modern base theme
        
        # Define colors
        self.primary_color = "#3b82f6"  # Blue
        self.secondary_color = "#1e40af"  # Darker blue
        self.bg_color = "#f5f5f7"  # Light gray
        self.accent_color = "#ef4444"  # Red
        self.text_color = "#111827"  # Dark gray
        self.light_text = "#6b7280"  # Medium gray
        
        # Configure button styles
        style.configure('TButton', 
                        font=('Segoe UI', 10),
                        background=self.primary_color,
                        foreground='white',
                        borderwidth=0,
                        focusthickness=3,
                        focuscolor=self.secondary_color)
        
        style.map('TButton',
                  background=[('active', self.secondary_color)],
                  relief=[('pressed', 'groove'),
                         ('!pressed', 'flat')])
        
        # Primary button style
        style.configure('Primary.TButton', 
                        background=self.primary_color,
                        foreground='white',
                        padding=(15, 8))
        
        # Action button style (smaller)
        style.configure('Action.TButton', 
                        background=self.primary_color,
                        foreground='white',
                        padding=(10, 5))
        
        # Danger button style
        style.configure('Danger.TButton', 
                        background=self.accent_color,
                        foreground='white')
        style.map('Danger.TButton',
                  background=[('active', '#dc2626')])  # Darker red
                  
        # Radio button style
        style.configure('TRadiobutton', 
                        background=self.bg_color,
                        foreground=self.text_color,
                        font=('Segoe UI', 10))
        
        # Label style
        style.configure('TLabel', 
                        background=self.bg_color,
                        foreground=self.text_color,
                        font=('Segoe UI', 10))
        
        # Heading label style
        style.configure('Heading.TLabel', 
                        background=self.bg_color,
                        foreground=self.primary_color,
                        font=('Segoe UI', 16, 'bold'))
        
        # Status label style
        style.configure('Status.TLabel', 
                        background='#e5e7eb',  # Light gray
                        foreground=self.text_color,
                        font=('Segoe UI', 9),
                        padding=5)
                        
        # Frame style
        style.configure('TFrame', background=self.bg_color)
        
        # Scale style
        style.configure('TScale', 
                      background=self.bg_color)
        
        # Separator style
        style.configure('TSeparator', background='#d1d5db')
        
        # Notebook style
        style.configure('TNotebook', background=self.bg_color)
        style.configure('TNotebook.Tab', 
                      background='#e5e7eb',
                      foreground=self.text_color,
                      padding=(10, 5),
                      font=('Segoe UI', 10))
        style.map('TNotebook.Tab',
                background=[('selected', self.primary_color)],
                foreground=[('selected', 'white')])
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, style='TFrame', padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # App title
        title_label = ttk.Label(main_frame, text="Voice Generator", style='Heading.TLabel')
        title_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Standard TTS tab
        standard_tab = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(standard_tab, text="Standard TTS")
        
        # Voice cloning tab
        clone_tab = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(clone_tab, text="Voice Clone")
        
        # Standard TTS tab content
        self.create_standard_tab(standard_tab)
        
        # Voice cloning tab content
        self.create_clone_tab(clone_tab)
        
        # Common controls at the bottom
        self.create_common_controls(main_frame)
        
    def create_standard_tab(self, parent):
        # Text input section
        ttk.Label(parent, text="Enter text to convert:", style='TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        # Text input with custom styling
        self.text_input = tk.Text(parent, 
                               height=8, 
                               width=50, 
                               wrap=tk.WORD,
                               font=('Segoe UI', 11),
                               borderwidth=1,
                               relief=tk.SOLID,
                               padx=8, 
                               pady=8)
        self.text_input.configure(bg='white', fg=self.text_color, insertbackground=self.primary_color)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Voice settings section
        settings_frame = ttk.Frame(parent, style='TFrame')
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Voice type selection
        voice_frame = ttk.Frame(settings_frame, style='TFrame')
        voice_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(voice_frame, text="Voice Type:", style='TLabel').pack(side=tk.LEFT, padx=(0, 10))
        
        self.voice_var = tk.StringVar(value="female")
        ttk.Radiobutton(voice_frame, 
                      text="Female Voice", 
                      variable=self.voice_var, 
                      value="female", 
                      style='TRadiobutton').pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Radiobutton(voice_frame, 
                      text="Male Voice", 
                      variable=self.voice_var, 
                      value="male", 
                      style='TRadiobutton').pack(side=tk.LEFT)
        
        # Pitch control section
        pitch_frame = ttk.Frame(settings_frame, style='TFrame')
        pitch_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(pitch_frame, text="Pitch Control:", style='TLabel').pack(side=tk.LEFT, padx=(0, 10))
        
        # Add pitch slider
        self.pitch_slider = ttk.Scale(
            pitch_frame,
            from_=0.5,  # Lower pitch
            to=1.5,     # Higher pitch
            orient=tk.HORIZONTAL,
            variable=self.pitch_factor,
            length=200,
            style='TScale'
        )
        self.pitch_slider.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        # Add pitch value label
        self.pitch_value_label = ttk.Label(
            pitch_frame, 
            text="1.0 (Normal)", 
            style='TLabel',
            width=12
        )
        self.pitch_value_label.pack(side=tk.LEFT)
        
        # Bind the slider to update the label
        self.pitch_slider.bind("<Motion>", self.update_pitch_label)
        self.pitch_slider.bind("<ButtonRelease-1>", self.update_pitch_label)
        
    def create_clone_tab(self, parent):
        # Text input section
        ttk.Label(parent, text="Enter text to convert:", style='TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        # Text input with custom styling
        self.clone_text_input = tk.Text(parent, 
                                     height=8, 
                                     width=50, 
                                     wrap=tk.WORD,
                                     font=('Segoe UI', 11),
                                     borderwidth=1,
                                     relief=tk.SOLID,
                                     padx=8, 
                                     pady=8)
        self.clone_text_input.configure(bg='white', fg=self.text_color, insertbackground=self.primary_color)
        self.clone_text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Voice sample section
        sample_frame = ttk.Frame(parent, style='TFrame')
        sample_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(sample_frame, text="Voice Sample:", style='TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        # Voice sample selection controls
        select_frame = ttk.Frame(sample_frame, style='TFrame')
        select_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.sample_path_var = tk.StringVar(value="No file selected")
        sample_path_label = ttk.Label(select_frame, textvariable=self.sample_path_var, style='TLabel')
        sample_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = ttk.Button(select_frame, 
                              text="Browse", 
                              command=self.browse_voice_sample,
                              style='Action.TButton')
        browse_btn.pack(side=tk.RIGHT)
        
        # Language selection for XTTS
        lang_frame = ttk.Frame(parent, style='TFrame')
        lang_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(lang_frame, text="Language:", style='TLabel').pack(side=tk.LEFT, padx=(0, 10))
        
        self.language_var = tk.StringVar(value="en")
        languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Polish": "pl",
            "Turkish": "tr",
            "Russian": "ru",
            "Dutch": "nl",
            "Czech": "cs",
            "Arabic": "ar",
            "Chinese": "zh-cn",
            "Korean": "ko",
            "Hindi": "hi"
        }
        
        language_dropdown = ttk.Combobox(lang_frame, 
                                       textvariable=self.language_var, 
                                       values=list(languages.values()),
                                       state="readonly",
                                       width=5)
        language_dropdown.pack(side=tk.LEFT)
        
        # Checkbutton for GPU acceleration
        gpu_check = ttk.Checkbutton(parent, 
                                 text="Use GPU acceleration (if available)", 
                                 variable=self.use_gpu,
                                 style='TCheckbutton')
        gpu_check.pack(anchor=tk.W, pady=(10, 5))
        
        # Show GPU status
        if torch.cuda.is_available():
            gpu_info = f"GPU detected: {torch.cuda.get_device_name(0)}"
        else:
            gpu_info = "No GPU detected, using CPU"
            self.use_gpu.set(False)
            
        gpu_status = ttk.Label(parent, text=gpu_info, style='Status.TLabel')
        gpu_status.pack(anchor=tk.W, pady=(0, 10))
        
    def create_common_controls(self, parent):
        # Control buttons in a nicer layout
        btn_frame = ttk.Frame(parent, style='TFrame')
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Generate button (primary action)
        self.generate_btn = ttk.Button(btn_frame, 
                                     text="Generate Speech", 
                                     command=self.generate_speech,
                                     style='Primary.TButton')
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        # Media control buttons
        controls_frame = ttk.Frame(btn_frame, style='TFrame')
        controls_frame.pack(side=tk.LEFT)
        
        self.play_btn = ttk.Button(controls_frame, 
                                 text="Play", 
                                 command=self.play_audio, 
                                 style='Action.TButton',
                                 state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 4))
        
        self.pause_btn = ttk.Button(controls_frame, 
                                  text="Pause", 
                                  command=self.pause_audio, 
                                  style='Action.TButton',
                                  state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 4))
        
        self.stop_btn = ttk.Button(controls_frame, 
                                 text="Stop", 
                                 command=self.stop_audio, 
                                 style='Action.TButton',
                                 state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.save_btn = ttk.Button(btn_frame, 
                                 text="Save As...", 
                                 command=self.save_audio, 
                                 style='Action.TButton',
                                 state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT)
        
        # Status bar with a cleaner look
        status_frame = ttk.Frame(parent, style='TFrame')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready to generate speech")
        self.status_label = ttk.Label(status_frame, 
                                   textvariable=self.status_var, 
                                   style='Status.TLabel',
                                   anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
        # Progress indicator
        self.progress = ttk.Progressbar(parent, 
                                      orient=tk.HORIZONTAL, 
                                      length=100, 
                                      mode='determinate',
                                      style='TProgressbar')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        self.progress['value'] = 0
    
    def browse_voice_sample(self):
        """Open file dialog to select voice sample"""
        file_path = filedialog.askopenfilename(
            title="Select Voice Sample",
            filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")]
        )
        
        if file_path:
            self.voice_clone_sample = file_path
            filename = os.path.basename(file_path)
            if len(filename) > 40:  # Truncate long filenames
                filename = filename[:37] + "..."
            self.sample_path_var.set(filename)
            self.status_var.set(f"Voice sample selected: {filename}")
    
    def update_pitch_label(self, event=None):
        pitch_value = self.pitch_factor.get()
        pitch_text = f"{pitch_value:.1f}"
        
        if pitch_value < 0.9:
            pitch_text += " (Lower)"
        elif pitch_value > 1.1:
            pitch_text += " (Higher)"
        else:
            pitch_text += " (Normal)"
            
        self.pitch_value_label.config(text=pitch_text)
    
    def ensure_file_available(self):
        """Make sure the output file is not in use and can be overwritten"""
        # Stop any playback
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
            
        # Unload the file from pygame
        pygame.mixer.music.unload()
        
        # Give the system a moment to release the file
        time.sleep(0.2)
        
        # Try to remove the files if they exist
        for file_path in [self.output_file, self.original_output_file]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    # If we can't remove it, create a new filename
                    timestamp = int(time.time())
                    self.original_output_file = f"original_output_{timestamp}.wav"
                    self.output_file = f"output_{timestamp}.wav"
                    self.status_var.set(f"Using new file: {self.output_file}")
                    break
    
    def generate_speech(self):
        # Determine which tab is active
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 0:  # Standard TTS tab
            text = self.text_input.get("1.0", tk.END).strip()
            mode = "standard"
        else:  # Voice Clone tab
            text = self.clone_text_input.get("1.0", tk.END).strip()
            mode = "xtts"
            
            # Check if voice sample is selected for clone mode
            if mode == "xtts" and not self.voice_clone_sample:
                self.status_var.set("Please select a voice sample for cloning")
                return
        
        if not text:
            self.status_var.set("Please enter some text to convert")
            return
        
        self.status_var.set("Preparing to generate speech...")
        self.generate_btn.configure(state=tk.DISABLED)
        self.play_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
        
        # Show progress animation
        self.progress['value'] = 0
        self.start_progress_animation()
        
        # Ensure the file is available for writing
        self.ensure_file_available()
        
        # Run TTS in a separate thread to prevent UI freezing
        self.status_var.set("Generating speech...")
        threading.Thread(target=self._generate_speech_thread, args=(text, mode), daemon=True).start()
    
    def start_progress_animation(self):
        """Animate the progress bar during generation"""
        def update_progress():
            if self.progress['value'] < 90 and self.generate_btn['state'] == tk.DISABLED:
                # Slowly fill to 90% while processing
                current = self.progress['value']
                increment = 0.5 if current < 50 else 0.2
                self.progress['value'] = current + increment
                self.root.after(100, update_progress)
        update_progress()
    
    def _generate_speech_thread(self, text, mode):
        try:
            # Patch torch.load to bypass security checks for XTTS
            if mode == "xtts":
                # Save the original torch.load function
                original_torch_load = torch.load
                # Create a wrapper that forces weights_only=False
                @functools.wraps(original_torch_load)
                def patched_torch_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)
                torch.load = patched_torch_load
            
            try:
                if mode == "standard":
                    # Standard TTS
                    voice_type = self.voice_var.get()
                    model_name = self.tts_models["standard"][voice_type]
                    
                    if voice_type == "male":
                        tts = TTS(model_name=model_name, progress_bar=False)
                        tts.tts_to_file(text=text, file_path=self.original_output_file, speaker="p226")
                    else:
                        tts = TTS(model_name=model_name, progress_bar=False)
                        tts.tts_to_file(text=text, file_path=self.original_output_file)
                else:
                    # XTTS Voice Cloning
                    device = "cuda" if self.use_gpu.get() and torch.cuda.is_available() else "cpu"
                    language = self.language_var.get()
                    
                    tts = TTS(self.tts_models["xtts"], gpu=self.use_gpu.get())
                    tts.to(device)
                    
                    tts.tts_to_file(
                        text=text,
                        file_path=self.original_output_file,
                        speaker_wav=self.voice_clone_sample,
                        language=language
                    )
            finally:
                # Restore the original torch.load function if patched
                if mode == "xtts" and 'original_torch_load' in locals():
                    torch.load = original_torch_load
            
            # Apply pitch shift with librosa if pitch is not 1.0 (normal) and in standard mode
            pitch_factor = self.pitch_factor.get()
            if mode == "standard" and abs(pitch_factor - 1.0) > 0.01:
                self.status_var.set("Applying pitch adjustment...")
                self.apply_pitch_shift(pitch_factor)
            else:
                # Just copy the file if no pitch change
                import shutil
                shutil.copy2(self.original_output_file, self.output_file)
            
            # Update UI on the main thread
            self.root.after(0, self._on_generation_complete)
        except Exception as e:
            error_msg = f"Error generating speech: {str(e)}"
            self.root.after(0, lambda: self._on_generation_error(error_msg))
    
    def apply_pitch_shift(self, pitch_factor):
        """Apply pitch shifting to the generated audio"""
        try:
            # Load the audio file
            y, sr = librosa.load(self.original_output_file, sr=None)
            
            # Calculate semitones based on pitch_factor (logarithmic scale)
            n_steps = 12 * np.log2(pitch_factor)
            
            # Apply pitch shifting
            y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
            
            # Save the result
            sf.write(self.output_file, y_shifted, sr)
        except Exception as e:
            raise Exception(f"Error applying pitch shift: {str(e)}")
    
    def _on_generation_complete(self):
        self.progress['value'] = 100
        self.status_var.set(f"Audio generated successfully")
        self.generate_btn.configure(state=tk.NORMAL)
        self.play_btn.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.NORMAL)
    
    def _on_generation_error(self, error_msg):
        self.progress['value'] = 0
        self.status_var.set(error_msg)
        self.generate_btn.configure(state=tk.NORMAL)
    
    def play_audio(self):
        if os.path.exists(self.output_file):
            try:
                if self.is_paused:
                    # Resume playback
                    pygame.mixer.music.unpause()
                    self.is_paused = False
                    self.is_playing = True
                else:
                    # Start new playback
                    pygame.mixer.music.load(self.output_file)
                    pygame.mixer.music.play()
                    self.is_playing = True
                
                self.play_btn.configure(state=tk.DISABLED)
                self.pause_btn.configure(state=tk.NORMAL)
                self.stop_btn.configure(state=tk.NORMAL)
                self.status_var.set("Playing audio...")
                
                # Check when playback has finished
                self.check_playback_finished()
            except Exception as e:
                self.status_var.set(f"Error playing audio: {str(e)}")
        else:
            self.status_var.set("No audio file available. Generate speech first.")
    
    def pause_audio(self):
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.is_playing = False
            self.play_btn.configure(state=tk.NORMAL)
            self.status_var.set("Playback paused")
    
    def check_playback_finished(self):
        if not pygame.mixer.music.get_busy() and self.is_playing and not self.is_paused:
            self.is_playing = False
            self.play_btn.configure(state=tk.NORMAL)
            self.pause_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.DISABLED)
            self.status_var.set("Playback finished")
        else:
            # Check again in 100ms if still playing
            if self.is_playing or self.is_paused:
                self.root.after(100, self.check_playback_finished)
    
    def stop_audio(self):
        if self.is_playing or self.is_paused:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
            self.play_btn.configure(state=tk.NORMAL)
            self.pause_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.DISABLED)
            self.status_var.set("Playback stopped")
    
    def save_audio(self):
        if not os.path.exists(self.output_file):
            self.status_var.set("No audio file available. Generate speech first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # If the user selected the same file, no need to copy
                if file_path != self.output_file:
                    import shutil
                    shutil.copy2(self.output_file, file_path)
                self.status_var.set(f"Audio saved to: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error saving file: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernTTSApp(root)
    root.mainloop()