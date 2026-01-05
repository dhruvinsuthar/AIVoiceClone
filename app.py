import streamlit as st
import torch
import torchaudio

# Fix for MeCab/fugashi error on Windows
import os
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = 'C:\\Program Files\\eSpeak NG\\libespeak-ng.dll'

# Disable Japanese phonemizer to avoid MeCab dependency
import sys
class DummyJapanesePhonimizer:
    pass

# Monkey patch to disable Japanese support
sys.modules['TTS.tts.utils.text.japanese'] = type(sys)('japanese')
sys.modules['TTS.tts.utils.text.japanese'].phonemizer = type(sys)('phonemizer')
sys.modules['TTS.tts.utils.text.japanese'].phonemizer.japanese_text_to_phonemes = lambda x: x

from TTS.api import TTS
import soundfile as sf
import numpy as np
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Voice Cloning Studio",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'tts_model' not in st.session_state:
    st.session_state.tts_model = None
if 'processed_audios' not in st.session_state:
    st.session_state.processed_audios = []
if 'voice_profiles' not in st.session_state:
    st.session_state.voice_profiles = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Generate Voice"

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tts_model(model_name):
    """Load TTS model with caching"""
    try:
        with st.spinner(f"Loading {model_name}..."):
            tts = TTS(model_name=model_name)
            return tts, None
    except Exception as e:
        return None, str(e)

def remove_vocals(audio_path, output_path):
    """
    Extract clean vocals from audio using Demucs (removes background music/noise)
    Returns path to vocals-only audio
    """
    try:
        import subprocess
        
        st.info("🎵 Extracting clean vocals (removing background noise/music)...")
        
        # Create output directory
        output_dir = Path(output_path).parent / "separated"
        output_dir.mkdir(exist_ok=True)
        
        # Run demucs to separate vocals from background
        # Using htdemucs model which is faster and good quality
        cmd = [
            "demucs",
            "--two-stems=vocals",  # Only separate vocals/instrumental
            "-n", "htdemucs",
            "--out", str(output_dir),
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.warning("Demucs not available, trying alternative method...")
            return audio_path  # Return original if demucs fails
        
        # Find the VOCALS file (not instrumental)
        audio_name = Path(audio_path).stem
        vocals_path = output_dir / "htdemucs" / audio_name / "vocals.wav"
        
        if vocals_path.exists():
            st.success("✅ Clean vocals extracted successfully!")
            return str(vocals_path)
        else:
            st.warning("Could not separate vocals, using original audio")
            return audio_path
            
    except ImportError:
        st.warning("⚠️ Demucs not installed. Using original audio.")
        return audio_path
    except Exception as e:
        st.warning(f"⚠️ Vocal removal failed: {e}. Using original audio.")
        return audio_path

def process_audio_file(uploaded_file, remove_vocals_flag=False):
    """Process uploaded audio file"""
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Remove vocals if requested
        if remove_vocals_flag:
            processed_path = os.path.join(temp_dir, f"processed_{uploaded_file.name}")
            temp_path = remove_vocals(temp_path, processed_path)
        
        return temp_path
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def calculate_speaking_rate(text, audio_duration):
    """
    Calculate speaking rate in words per minute (WPM)
    """
    try:
        # Count words in text
        word_count = len(text.split())
        
        # Calculate WPM
        duration_minutes = audio_duration / 60.0
        if duration_minutes > 0:
            wpm = word_count / duration_minutes
            return wpm, word_count
        else:
            return 0, word_count
    except Exception as e:
        return 0, 0

def adjust_audio_speed(audio_path, speed_factor):
    """
    Adjust the playback speed of audio without changing pitch
    speed_factor: 0.5 (slower) to 2.0 (faster)
    """
    try:
        from scipy import signal
        
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Calculate new length
        new_length = int(len(audio) / speed_factor)
        
        # Resample to change speed while maintaining pitch
        # Using high-quality resampling
        resampled = signal.resample(audio, new_length)
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "speed_adjusted_output.wav")
        sf.write(output_path, resampled, sr)
        
        return output_path
        
    except Exception as e:
        st.error(f"Speed adjustment error: {e}")
        return audio_path  # Return original if adjustment fails

def clone_voice_multi_reference(tts_model, reference_paths, text, language="en", speed_factor=1.0):
    """
    Clone voice using the profile reference audio
    speed_factor: Controls playback speed (0.5 = slower, 2.0 = faster)
    """
    try:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "cloned_output.wav")
        
        # Use the profile reference audio directly
        # Profile already has all samples concatenated with vocals extracted
        reference_audio = reference_paths[0]
        
        # Generate speech using the profile
        tts_model.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=language,
            file_path=output_path
        )
        
        if os.path.exists(output_path):
            # Apply speed adjustment if speed_factor is not 1.0
            if speed_factor != 1.0:
                output_path = adjust_audio_speed(output_path, speed_factor)
            
            return output_path
        else:
            return None
            
    except Exception as e:
        st.error(f"Voice cloning error: {e}")
        return None

def concatenate_audios(audio_paths, max_duration=None):
    """
    Concatenate multiple audio files
    No limit on duration - uses all provided audio
    """
    try:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "concatenated_reference.wav")
        
        combined_audio = []
        combined_sr = None
        
        for audio_path in audio_paths:
            audio, sr = sf.read(audio_path)
            
            if combined_sr is None:
                combined_sr = sr
            elif sr != combined_sr:
                # Resample if needed
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio).float(),
                    sr,
                    combined_sr
                ).numpy()
            
            combined_audio.append(audio)
        
        # Concatenate all audio (no duration limit)
        final_audio = np.concatenate(combined_audio)
        
        # Apply max_duration only if specified
        if max_duration is not None:
            max_samples = int(max_duration * combined_sr)
            if len(final_audio) > max_samples:
                final_audio = final_audio[:max_samples]
        
        # Save
        sf.write(output_path, final_audio, combined_sr)
        
        duration = len(final_audio) / combined_sr
        st.success(f"✅ Concatenated {len(audio_paths)} files → {duration:.1f} seconds total")
        
        return output_path
        
    except Exception as e:
        st.error(f"Error concatenating audios: {e}")
        return audio_paths[0]  # Return first one as fallback

def save_voice_profile(profile_name, reference_audio_path):
    """Save a voice profile to disk"""
    try:
        # Create profiles directory
        profiles_dir = Path("voice_profiles")
        profiles_dir.mkdir(exist_ok=True)
        
        # Save the concatenated reference audio
        profile_path = profiles_dir / f"{profile_name}.wav"
        
        # Copy the reference audio to profile
        import shutil
        shutil.copy(reference_audio_path, profile_path)
        
        # Get audio info
        audio, sr = sf.read(profile_path)
        duration = len(audio) / sr
        
        # Save metadata
        import json
        metadata = {
            'name': profile_name,
            'duration': duration,
            'sample_rate': sr,
            'created_at': str(Path(profile_path).stat().st_ctime)
        }
        
        metadata_path = profiles_dir / f"{profile_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(profile_path), duration
        
    except Exception as e:
        st.error(f"Error saving profile: {e}")
        return None, None

def load_voice_profiles():
    """Load all saved voice profiles"""
    try:
        profiles_dir = Path("voice_profiles")
        if not profiles_dir.exists():
            return {}
        
        profiles = {}
        for wav_file in profiles_dir.glob("*.wav"):
            profile_name = wav_file.stem
            
            # Load metadata if exists
            metadata_path = profiles_dir / f"{profile_name}.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                # Get basic info
                audio, sr = sf.read(wav_file)
                duration = len(audio) / sr
                metadata = {
                    'name': profile_name,
                    'duration': duration,
                    'sample_rate': sr
                }
            
            profiles[profile_name] = {
                'path': str(wav_file),
                'metadata': metadata
            }
        
        return profiles
        
    except Exception as e:
        st.error(f"Error loading profiles: {e}")
        return {}

def delete_voice_profile(profile_name):
    """Delete a voice profile"""
    try:
        profiles_dir = Path("voice_profiles")
        wav_file = profiles_dir / f"{profile_name}.wav"
        json_file = profiles_dir / f"{profile_name}.json"
        
        if wav_file.exists():
            wav_file.unlink()
        if json_file.exists():
            json_file.unlink()
        
        return True
    except Exception as e:
        st.error(f"Error deleting profile: {e}")
        return False

# Main UI
st.title("🎙️ AI Voice Cloning Studio")
st.markdown("### Clone any voice with AI - Create profiles with unlimited samples")

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["🎙️ Generate Voice", "➕ Create Profile", "📋 Manage Profiles"])

# Sidebar - Model Selection
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Check if fine-tuned model exists
    finetuned_path = Path("./xtts_hindi_finetuned")
    has_finetuned = finetuned_path.exists() and (finetuned_path / "best_model.pth").exists()
    
    model_options = {
        "XTTS v2 (Best Quality)": "tts_models/multilingual/multi-dataset/xtts_v2",
        "XTTS v1.1 (Stable)": "tts_models/multilingual/multi-dataset/xtts_v1.1",
    }
    
    # Add fine-tuned model if available at the top
    if has_finetuned:
        # Insert at beginning for prominence
        model_options = {
            "🌟 XTTS Hindi Fine-tuned (Indian Celebrities)": str(finetuned_path),
            **model_options
        }
    
    selected_model_name = st.selectbox(
        "Select TTS Model",
        options=list(model_options.keys()),
        index=0
    )
    
    # Show info about selected model
    if "Fine-tuned" in selected_model_name:
        st.success("✅ Using Hindi-optimized model trained on Indian celebrity voices!")
        st.info("📊 Optimized for: Hindi language • Indian accents • Bollywood voices")
    
    selected_model = model_options[selected_model_name]
    
    # Show fine-tuning option
    st.markdown("---")
    if not has_finetuned:
        with st.expander("🎯 Create Custom Hindi Model", expanded=False):
            st.markdown("""
            ### Fine-tune XTTS for Indian Voices
            
            **Quick Setup (5 minutes):**
            ```powershell
            python quickstart_finetune.py --quick
            ```
            
            **Or Full Training (1-3 hours):**
            ```powershell
            python finetune_quick.bat
            ```
            
            **What you get:**
            - ✅ Better Hindi pronunciation
            - ✅ Natural Indian accents
            - ✅ Trained on Bollywood celebrities
            - ✅ 18+ Indian celebrity voices
            
            **Your Dataset Includes:**
            - Amitabh Bachchan
            - Shah Rukh Khan
            - Akshay Kumar
            - And 15+ more celebrities!
            
            See `README_FINETUNING.md` for details.
            """)
    else:
        with st.expander("ℹ️ About Your Fine-tuned Model", expanded=False):
            # Try to load training info
            info_file = finetuned_path / "training_complete.txt"
            if info_file.exists():
                st.markdown("### Training Details")
                st.code(info_file.read_text())
            
            st.markdown("""
            ### Your Custom Model
            
            This model is fine-tuned on:
            - **Dataset**: VoxCeleb Indian Celebrities
            - **Languages**: Hindi + English (Indian accent)
            - **Base Model**: XTTS v2
            
            **Optimized for:**
            - Indian English accents
            - Hindi language synthesis
            - Bollywood celebrity voice cloning
            
            To retrain or update, run:
            ```powershell
            python quickstart_finetune.py
            ```
            """)

    
    # Language selection
    language_options = {
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
        "Hungarian": "hu",
        "Korean": "ko",
        "Hindi": "hi"
    }
    
    selected_language = st.selectbox(
        "Output Language",
        options=list(language_options.keys()),
        index=0
    )
    
    language_code = language_options[selected_language]
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.info(f"PyTorch: {torch.__version__}")
    st.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
if st.session_state.tts_model is None or st.session_state.get('current_model') != selected_model:
    tts_model, error = load_tts_model(selected_model)
    if tts_model:
        st.session_state.tts_model = tts_model
        st.session_state.current_model = selected_model
        st.success(f"✅ {selected_model_name} loaded successfully!")
    else:
        st.error(f"❌ Failed to load model: {error}")
        st.stop()
else:
    tts_model = st.session_state.tts_model

# TAB 1: Generate Voice using existing profile
with tab1:
    st.header("🎙️ Generate Voice from Profile")
    
    # Load available profiles
    available_profiles = load_voice_profiles()
    
    if not available_profiles:
        st.warning("⚠️ No voice profiles found. Please create a profile first in the 'Create Profile' tab.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Select Voice Profile")
            
            profile_names = list(available_profiles.keys())
            selected_profile = st.selectbox(
                "Choose a voice profile",
                options=profile_names,
                format_func=lambda x: f"{x} ({available_profiles[x]['metadata']['duration']:.1f}s)"
            )
            
            if selected_profile:
                profile_info = available_profiles[selected_profile]
                st.info(f"📊 Duration: {profile_info['metadata']['duration']:.1f} seconds")
                st.info(f"📊 Sample Rate: {profile_info['metadata']['sample_rate']} Hz")
                
                # Preview profile audio
                st.audio(profile_info['path'], format='audio/wav')
        
        with col2:
            st.subheader("Enter Text to Synthesize")
            
            text_input = st.text_area(
                "Enter the text you want to speak in the cloned voice",
                height=150,
                placeholder="Enter your text here...",
                key="generate_text"
            )
            
            # Speech Speed Control
            st.markdown("### 🎚️ Speech Speed Control")
            speed_factor = st.slider(
                "Adjust speaking speed",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="0.5 = Half speed (slower), 1.0 = Normal speed, 2.0 = Double speed (faster)",
                key="speed_slider"
            )
            
            # Show speed description
            if speed_factor < 0.8:
                speed_desc = "🐢 Very Slow"
            elif speed_factor < 1.0:
                speed_desc = "🐌 Slow"
            elif speed_factor == 1.0:
                speed_desc = "⚡ Normal"
            elif speed_factor <= 1.5:
                speed_desc = "🚀 Fast"
            else:
                speed_desc = "⚡ Very Fast"
            
            st.info(f"Current Speed: **{speed_desc}** ({speed_factor}x)")
            
            if st.button("🎙️ Generate Voice", type="primary", key="generate_btn"):
                if not text_input.strip():
                    st.error("❌ Please enter some text to synthesize!")
                else:
                    with st.spinner("Generating cloned voice..."):
                        # Generate voice using the selected profile
                        output_path = clone_voice_multi_reference(
                            tts_model,
                            [profile_info['path']],
                            text_input,
                            language_code,
                            speed_factor
                        )
                        
                        if output_path and os.path.exists(output_path):
                            st.success("✅ Voice generated successfully!")
                            
                            # Display audio info
                            try:
                                audio_data, sample_rate = sf.read(output_path)
                                duration = len(audio_data) / sample_rate
                                
                                # Calculate speaking rate
                                wpm, word_count = calculate_speaking_rate(text_input, duration)
                                
                                # Display statistics
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Duration", f"{duration:.2f}s")
                                with col_b:
                                    st.metric("Word Count", word_count)
                                with col_c:
                                    st.metric("Speaking Rate", f"{wpm:.0f} WPM")
                                
                                # Additional info
                                st.info(f"📊 Sample Rate: {sample_rate} Hz | Speed Factor: {speed_factor}x")
                                
                                # Speaking rate interpretation
                                if wpm < 100:
                                    rate_desc = "🐢 Very Slow (Good for learning)"
                                elif wpm < 130:
                                    rate_desc = "🐌 Slow (Clear and deliberate)"
                                elif wpm < 160:
                                    rate_desc = "⚡ Normal (Natural conversation)"
                                elif wpm < 200:
                                    rate_desc = "🚀 Fast (Quick presentation)"
                                else:
                                    rate_desc = "⚡ Very Fast (Rapid speech)"
                                
                                st.success(f"Speaking Rate: {rate_desc}")
                                
                            except:
                                pass
                            
                            # Play audio
                            st.audio(output_path, format='audio/wav')
                            
                            # Download button
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Generated Voice",
                                    data=f,
                                    file_name=f"{selected_profile}_output.wav",
                                    mime="audio/wav"
                                )

# TAB 2: Create New Profile
with tab2:
    st.header("➕ Create New Voice Profile")
    st.markdown("Upload unlimited audio samples to create a voice profile")
    
    profile_name = st.text_input(
        "Profile Name",
        placeholder="e.g., John_Doe, Celebrity_Name, My_Voice",
        key="profile_name"
    )
    
    uploaded_files = st.file_uploader(
        "Upload Audio Samples (No Limit!)",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        accept_multiple_files=True,
        key="profile_uploader",
        help="Upload as many files as you want (200-300+ samples supported)"
    )
    
    extract_vocals = st.checkbox(
        "🎵 Extract clean vocals (remove background music/noise)",
        value=True,
        help="Recommended: This will remove background music and noise from all samples",
        key="extract_vocals"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        # Show file list
        with st.expander(f"📁 View uploaded files ({len(uploaded_files)} files)"):
            for idx, file in enumerate(uploaded_files, 1):
                st.text(f"{idx}. {file.name}")
    
    if st.button("🎯 Create Profile", type="primary", disabled=not uploaded_files or not profile_name):
        if not profile_name.strip():
            st.error("❌ Please enter a profile name!")
        elif not uploaded_files:
            st.error("❌ Please upload at least one audio file!")
        else:
            with st.spinner(f"Creating profile '{profile_name}'..."):
                processed_files = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each file
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    processed_path = process_audio_file(uploaded_file, extract_vocals)
                    if processed_path:
                        processed_files.append(processed_path)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Concatenating all audio files...")
                
                # Concatenate all processed files (NO TIME LIMIT)
                if processed_files:
                    concatenated_path = concatenate_audios(processed_files, max_duration=None)
                    
                    # Save as profile
                    if concatenated_path:
                        profile_path, duration = save_voice_profile(profile_name, concatenated_path)
                        
                        if profile_path:
                            st.success(f"✅ Profile '{profile_name}' created successfully!")
                            st.info(f"📊 Total duration: {duration:.1f} seconds")
                            st.info(f"📊 Files processed: {len(processed_files)}")
                            
                            # Preview
                            st.audio(profile_path, format='audio/wav')
                            
                            # Clear upload
                            st.balloons()
                        else:
                            st.error("❌ Failed to save profile")
                else:
                    st.error("❌ No audio files were successfully processed")
                
                progress_bar.empty()
                status_text.empty()

# TAB 3: Manage Profiles
with tab3:
    st.header("📋 Manage Voice Profiles")
    
    # Reload profiles
    if st.button("🔄 Refresh Profiles"):
        st.rerun()
    
    available_profiles = load_voice_profiles()
    
    if not available_profiles:
        st.info("No voice profiles found. Create one in the 'Create Profile' tab.")
    else:
        st.success(f"Found {len(available_profiles)} profile(s)")
        
        for profile_name, profile_data in available_profiles.items():
            with st.expander(f"🎙️ {profile_name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Duration:** {profile_data['metadata']['duration']:.1f} seconds")
                    st.write(f"**Sample Rate:** {profile_data['metadata']['sample_rate']} Hz")
                    
                    # Audio preview
                    st.audio(profile_data['path'], format='audio/wav')
                
                with col2:
                    # Download profile
                    with open(profile_data['path'], 'rb') as f:
                        st.download_button(
                            label="📥 Download",
                            data=f,
                            file_name=f"{profile_name}.wav",
                            mime="audio/wav",
                            key=f"download_{profile_name}"
                        )
                    
                    # Delete profile
                    if st.button("🗑️ Delete", key=f"delete_{profile_name}"):
                        if delete_voice_profile(profile_name):
                            st.success(f"Deleted {profile_name}")
                            st.rerun()
                        else:
                            st.error("Failed to delete profile")

# Main content (OLD - now moved to tabs)
col1, col2 = st.columns([1, 1])

with col1:
    pass  # Content moved to tabs

with col2:
    pass  # Content moved to tabs

# Footer
st.markdown("---")
st.markdown("""
### 💡 Tips for Best Results:
- Use high-quality audio samples (clear, minimal background noise)
- Upload as many samples as you want (200-300+)
- No time limit on concatenated audio
- **Enable vocal extraction** if there's background music, instruments, or noise
- **Adjust speech speed** for different use cases (0.5x for learning, 1.5x for presentations)
- Check the **WPM (Words Per Minute)** to ensure natural speaking rate
- Create profiles once, use them anytime
""")

st.markdown("""
### 🔧 Features:
- ✅ Unlimited reference audio samples (200-300+ supported)
- ✅ No duration limit on concatenated audio
- ✅ Voice profile management (create, use, delete)
- ✅ Automatic vocal extraction (removes background music/noise)
- ✅ **Speech speed control (0.5x - 2.0x)** 🆕
- ✅ **Real-time speaking rate analysis (WPM)** 🆕
- ✅ Multi-language support (17+ languages)
- ✅ High-quality voice cloning with XTTS v2
- ✅ Real-time audio preview
- ✅ Easy download of generated audio
""")
