import streamlit as st
import moviepy.editor as mp
import cv2
import pytesseract
import tempfile
import os
import numpy as np
import yt_dlp
import logging
from pydub import AudioSegment
import time
import re
import pyperclip  # For copy to clipboard functionality

# Lightweight AI libraries
import faster_whisper
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='video_text_extractor.log'
)

class MultilingualVideoConverter:
    def __init__(self, processing_mode="Balanced"):
        """
        Initialize advanced multilingual transcription model with Hindi optimizations
        """
        try:
            # Model selection based on processing mode
            model_map = {
                "Fast (lower accuracy)": "tiny",
                "Balanced": "small",
                "High Accuracy (slower)": "medium"
            }
            
            model_size = model_map.get(processing_mode, "small")
            compute_type = 'float16' if torch.cuda.is_available() else 'int8'
            
            st.info(f"Loading {model_size} transcription model for {processing_mode} mode...")
            
            self.whisper_model = faster_whisper.WhisperModel(
                model_size_or_path=model_size,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                compute_type=compute_type,
                download_root="./models",
                local_files_only=False
            )
            # st.success("Multilingual transcription model loaded successfully!")
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            st.error(f"Model loading error: {e}")
            self.whisper_model = None

    def get_supported_languages(self):
        """Return dictionary of supported languages with codes and names"""
        return {
            'auto': 'Auto-detect',
            'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 
            'fr': 'French', 'de': 'German', 'zh': 'Chinese',
            'ar': 'Arabic', 'ru': 'Russian', 'pt': 'Portuguese',
            'ja': 'Japanese', 'ko': 'Korean', 'it': 'Italian',
            'nl': 'Dutch', 'pa': 'Punjabi', 'ta': 'Tamil',
            'te': 'Telugu', 'mr': 'Marathi', 'bn': 'Bengali'
        }

    def detect_language(self, audio_path):
        """
        Detect language of the audio using Whisper with Hindi-specific checks
        """
        if not self.whisper_model:
            st.error("Transcription model not available")
            return {'code': 'en', 'name': 'English'}

        try:
            # Use transcribe method with language detection
            segments, info = self.whisper_model.transcribe(
                audio_path, 
                beam_size=5,
                vad_filter=True,
                language=None  # Force auto-detection
            )
            
            # Get detected language
            detected_language = info.language
            
            # Special handling for potential Hindi misclassification
            if detected_language in ['mr', 'bn', 'pa']:  # Languages often confused with Hindi
                # Analyze some segments to confirm
                hindi_keywords = ['है', 'और', 'में', 'के', 'लिए']
                segments_list = list(segments)
                sample_text = " ".join([seg.text for seg in segments_list[:5]])
                
                # If Hindi keywords are found, override detection
                if any(keyword in sample_text for keyword in hindi_keywords):
                    detected_language = 'hi'
            
            # Get language name from our supported languages
            lang_name = self.get_supported_languages().get(detected_language, detected_language.capitalize())
            
            return {
                'code': detected_language, 
                'name': lang_name
            }
        except Exception as e:
            logging.error(f"Language detection error: {e}")
            st.error(f"Language detection error: {e}")
            return {'code': 'en', 'name': 'English'}

    def transcribe_audio(self, audio_path, language='auto'):
        """
        Advanced multilingual transcription with Hindi-specific optimizations
        """
        if not self.whisper_model:
            st.error("Transcription model not available")
            return "Transcription failed", 'en', 'English'

        try:
            # If language is auto, detect it first
            if language == 'auto':
                lang_info = self.detect_language(audio_path)
                language_code = lang_info['code']
                language_name = lang_info['name']
            else:
                language_code = language
                language_name = self.get_supported_languages().get(language, language.capitalize())

            # Hindi-specific parameters
            if language_code == 'hi':
                beam_size = 7
                best_of = 7
                temperature = (0.0, 0.2, 0.4)
                patience = 2.0
                initial_prompt = "हिंदी भाषा में बातचीत, औपचारिक भाषा"
            else:
                beam_size = 5
                best_of = 5
                temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                patience = 1.0
                initial_prompt = None

            # Transcribe with detected/specified language and better parameters
            segments, info = self.whisper_model.transcribe(
                audio_path, 
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                language=None if language == 'auto' else language_code,
                vad_filter=True,
                word_timestamps=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                temperature=temperature,
                initial_prompt=initial_prompt)
            
            # If we auto-detected, update language info from the actual transcription
            if language == 'auto':
                language_code = info.language
                language_name = self.get_supported_languages().get(language_code, language_code.capitalize())

            # Combine segments with proper punctuation and spacing
            transcription = []
            previous_end = None
            
            with st.spinner("Processing transcription segments..."):
                segments_list = list(segments)
                total_segments = len(segments_list)
                
                if total_segments == 0:
                    return "No speech detected", language_code, language_name
                
                progress_bar = st.progress(0)
                
                for i, segment in enumerate(segments_list):
                    progress = int((i + 1) / total_segments * 100)
                    progress_bar.progress(progress)
                    
                    if previous_end is not None and segment.start > previous_end + 1.0:
                        transcription.append("\n\n")
                    elif previous_end is not None and segment.start > previous_end + 0.5:
                        transcription.append(" ")
                    
                    transcription.append(segment.text)
                    previous_end = segment.end
                
                progress_bar.empty()
            
            # Join all segments
            full_transcription = "".join(transcription).strip()
            
            # Language-specific post-processing
            if language_code == 'hi':
                full_transcription = self._post_process_hindi(full_transcription)
            else:
                full_transcription = self._post_process_transcription(full_transcription)
            
            return full_transcription or "No text detected", language_code, language_name

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            st.error(f"Transcription error: {e}")
            return "Transcription failed", 'en', 'English'

    def _post_process_hindi(self, text):
        """Special post-processing for Hindi text"""
        if not text:
            return text
            
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common punctuation issues in Hindi
        text = text.replace(' ,', ',')
        text = text.replace(' ।', '। ')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        
        # Capitalize proper nouns (simple heuristic)
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip():
                # Capitalize first letter of each line
                if len(line) > 0:
                    line = line[0].upper() + line[1:]
                processed_lines.append(line)
        
        text = '\n'.join(processed_lines)
        
        return text

    def _post_process_transcription(self, text):
        """Clean up the transcribed text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common punctuation issues
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        
        # Capitalize sentences
        sentences = text.split('. ')
        sentences = [s.strip().capitalize() for s in sentences if s.strip()]
        text = '. '.join(sentences)
        
        return text

    def advanced_ocr(self, video_path):
        """
        Enhanced OCR with multiple preprocessing techniques
        """
        # Dynamic Tesseract path detection
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\w\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract'
        ]
        
        # Find the first existing Tesseract path
        tesseract_cmd = next((path for path in tesseract_paths if os.path.exists(path)), None)
        
        if not tesseract_cmd:
            st.error("Tesseract OCR not found. Please install Tesseract.")
            return "OCR preprocessing failed"
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Advanced OCR configuration
        custom_config = r'--oem 3 --psm 6'
        
        cap = cv2.VideoCapture(video_path)
        ocr_texts = []

        # Smart frame sampling - process more frames for better coverage
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = max(1, int(fps))  # Sample 1 frame per second

        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % sample_interval == 0:
                try:
                    # Update progress
                    progress = min(100, int((frame_count / total_frames) * 100))
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count} of {total_frames}...")

                    # Advanced preprocessing pipeline
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Multiple preprocessing techniques
                    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                    equalized = cv2.equalizeHist(denoised)
                    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
                    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Try different configurations
                    text = pytesseract.image_to_string(thresholded, config=custom_config)
                    if text.strip():
                        ocr_texts.append(text.strip())

                    processed_frames += 1

                except Exception as e:
                    logging.warning(f"OCR processing frame error: {e}")

        cap.release()
        progress_bar.empty()
        status_text.empty()

        # Combine results with timestamps
        final_text = []
        for i, text in enumerate(ocr_texts):
            timestamp = time.strftime('%H:%M:%S', time.gmtime(i))
            final_text.append(f"[{timestamp}]\n{text}\n")

        return "\n".join(final_text) if final_text else "No text detected in video frames"

def download_video(url):
    """
    Optimized video download function with comprehensive error handling
    Supports YouTube and Instagram Reel URLs
    """
    try:
        # Validate URL first
        if not url or not (url.startswith(('http://', 'https://')) and 
                           (('youtube.com' in url) or ('instagram.com' in url) or ('reels.instagram.com' in url))):
            st.error("Invalid URL. Please provide a complete YouTube or Instagram Reel URL.")
            return None

        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio/best',
            'nooverwrites': True,
            'no_color': True,
            'noplaylist': True,
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
        }

        st.info(f"Attempting to download: {url}")
        logging.info(f"Download attempt for URL: {url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Verify video availability
                video_info = ydl.extract_info(url, download=False)
                
                if not video_info:
                    st.error("Could not retrieve video information. Check URL or video availability.")
                    return None

                # Check video duration
                duration = video_info.get('duration', 0)
                if duration > 7200:  # Limit to 2-hour videos
                    st.error("Video too long. Please select a video under 2 hours.")
                    return None

                # Perform actual download
                download_result = ydl.extract_info(url, download=True)
                
                if not download_result:
                    st.error("Download failed. Video might be unavailable.")
                    return None

                # Get the downloaded file path
                downloaded_file = ydl.prepare_filename(download_result)
                downloaded_file = downloaded_file.replace(downloaded_file.split('.')[-1], 'wav')

                if not os.path.exists(downloaded_file):
                    st.error(f"Downloaded file not found: {downloaded_file}")
                    logging.error(f"File not found after download: {downloaded_file}")
                    return None

                # Distinguish source in success message
                source = "YouTube" if "youtube.com" in url else "Instagram"
                st.success(f"Successfully downloaded {source} video: {os.path.basename(downloaded_file)}")
                logging.info(f"Successfully downloaded {source} file: {downloaded_file}")

                return downloaded_file

            except yt_dlp.utils.DownloadError as e:
                st.error(f"Download error: {e}")
                logging.error(f"Download Error: {e}")
                return None
            except Exception as e:
                st.error(f"Unexpected download error: {e}")
                logging.error(f"Unexpected download error: {e}", exc_info=True)
                return None

    except Exception as e:
        st.error(f"Critical download error: {e}")
        logging.critical(f"Critical download error: {e}", exc_info=True)
        return None

# def youtube_progress_hook(d):
#     """
#     Progress hook for YouTube download to provide status updates
#     """
#     if d['status'] == 'downloading':
#         downloaded_bytes = d.get('downloaded_bytes', 0)
#         total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        
#         if total_bytes > 0:
#             percent = downloaded_bytes * 100 / total_bytes
#             st.info(f"Downloading: {percent:.1f}%")

def main():
    st.set_page_config(
        page_title="Multilingual Video Text Extractor", 
        page_icon="🌐",
        layout="wide"
    )
    st.markdown("""
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7220800899817072"
     crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

    st.title("🌐 Video Text Extractor")
    st.markdown("""
    Extract complete and accurate text from video/audio in multiple languages.
    Supports both uploaded files and YouTube/Insta URLs.
    """)

    st.markdown("""
    <div style="width: 100%; margin: 10px 0;">
        <ins class="adsbygoogle"
            style="display:block"
            data-ad-client="ca-pub-7220800899817072"
            data-ad-slot="1297128580"
            data-ad-format="auto"
            data-full-width-responsive="true"></ins>
        <script>
            (adsbygoogle = window.adsbygoogle || []).push({});
        </script>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("Settings")
    processing_mode = st.sidebar.radio(
        "Processing Mode",
        ("Fast (lower accuracy)", "Balanced", "High Accuracy (slower)"),
        index=0
    )
    
    # Initialize converter with selected processing mode
    converter = MultilingualVideoConverter(processing_mode)
    supported_languages = converter.get_supported_languages()
    
    # Language selection with Hindi recommendation
    selected_language = st.sidebar.selectbox(
        "Transcription Language",
        options=list(supported_languages.keys()),
        format_func=lambda x: f"{supported_languages[x]} {'(recommended)' if x == 'hi' else ''}",
        index=0  # Auto-detect by default
    )

    # Hindi-specific note
    if selected_language == 'hi':
        st.sidebar.info("For best Hindi results, use high-quality audio and select 'High Accuracy' mode")

    # Main input section
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Video/Audio", 
            type=['mp4', 'avi', 'mov', 'mkv', 'wav', 'mp3', 'm4a', 'ogg'],
            help="Supported formats: MP4, AVI, MOV, MKV, WAV, MP3, M4A, OGG"
        )
    
    with col2:
        youtube_url = st.text_input(
            "Or Paste YouTube/Instagram Reel URL",
            placeholder="https://www.youtube.com/watch?v=... or https://www.instagram.com/reels/...",
            help="Paste a YouTube or Instagram Reel URL to extract text from the video"
        )

    if st.button("Extract Text", type="primary"):
        # Reset session state
        st.session_state.input_path = None
        st.session_state.audio_path = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Validate input
            if not uploaded_file and not youtube_url:
                st.error("Please upload a file or provide a YouTube URL")
                return

            status_text.text("Processing input...")
            
            # Determine input source
            if uploaded_file:
                try:
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                        temp_file.write(uploaded_file.read())
                        st.session_state.input_path = temp_file.name
                    
                    if not st.session_state.input_path or not os.path.exists(st.session_state.input_path):
                        st.error("Failed to process uploaded file")
                        return
                    
                    file_size = os.path.getsize(st.session_state.input_path)
                    if file_size == 0:
                        st.error("Uploaded file is empty")
                        return
                
                except Exception as upload_error:
                    logging.error(f"File upload error: {upload_error}")
                    st.error(f"File upload error: {upload_error}")
                    return

            elif youtube_url:
                status_text.text("Downloading YouTube video...")
                st.session_state.input_path = download_video(youtube_url)
                if not st.session_state.input_path:
                    st.error("Failed to download YouTube video")
                    return

            progress_bar.progress(20)
            status_text.text("Preparing audio for transcription...")

            # Convert video to audio if needed
            if st.session_state.input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                try:
                    # Ensure the video file is not corrupted
                    with st.spinner("Checking video file integrity..."):
                        video = mp.VideoFileClip(st.session_state.input_path)
                        if video.duration == 0:
                            st.error("The video file appears to be corrupted or empty")
                            return
                            
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                        st.session_state.audio_path = temp_audio.name
                    
                    with st.spinner("Extracting audio from video..."):
                        video.audio.write_audiofile(
                            st.session_state.audio_path, 
                            codec='pcm_s16le',
                            ffmpeg_params=['-ac', '1'],  # Convert to mono for better ASR
                            verbose=False,
                            logger=None
                        )
                        video.close()
                        
                    # Verify audio file was created properly
                    if not os.path.exists(st.session_state.audio_path) or os.path.getsize(st.session_state.audio_path) == 0:
                        st.error("Failed to extract valid audio from video")
                        return
                        
                except Exception as e:
                    st.error(f"Failed to extract audio: {e}")
                    return
            else:
                st.session_state.audio_path = st.session_state.input_path

            progress_bar.progress(40)
            status_text.text("Starting transcription...")

            # Transcribe audio with selected language
            transcription, lang_code, detected_language = converter.transcribe_audio(
                st.session_state.audio_path,
                language=selected_language
            )
            
            # Show language info based on selection
            if selected_language == 'auto':
                st.info(f"Detected Language: {detected_language}")
            else:
                st.info(f"Transcribed in: {detected_language} (forced)")

            progress_bar.progress(60)
            status_text.text("Extracting text from video frames...")

            # Extract text from video frames (if original input was a video)
            if st.session_state.input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                ocr_text = converter.advanced_ocr(st.session_state.input_path)
            else:
                ocr_text = "No video frames to extract text from"

            progress_bar.progress(80)
            status_text.text("Preparing results...")

            # Display results
            tab1, tab2 = st.tabs(["Audio Transcription", "Video Text Extraction"])
            
            with tab1:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(f"🎙️ {detected_language} Audio Transcription")
                with col2:
                    if st.button("📋 Copy Transcription", key="copy_transcription"):
                        pyperclip.copy(transcription)
                        st.success("Copied to clipboard!")
                
                st.download_button(
                    label=f"Download {detected_language} Transcription",
                    data=transcription,
                    file_name=f"{detected_language.lower()}_transcription.txt",
                    mime="text/plain",
                    key="transcription_download"
                )
                st.text_area(
                    "Transcribed Text", 
                    transcription, 
                    height=400,
                    label_visibility="collapsed",
                    key="transcription_text_area"
                )
            
            with tab2:
                st.subheader("📝 Extracted Text from Video Frames")
                st.download_button(
                    label="Download OCR Text",
                    data=ocr_text,
                    file_name="ocr_text.txt",
                    mime="text/plain",
                    key="ocr_download"
                )
                st.text_area(
                    "OCR Text", 
                    ocr_text, 
                    height=400,
                    label_visibility="collapsed"
                )

            progress_bar.progress(100)
            status_text.text("Processing complete!")
            time.sleep(1)
            status_text.empty()

        except Exception as e:
            logging.error(f"Unexpected processing error: {e}", exc_info=True)
            st.error(f"Processing error: {e}")
        finally:
            # Clean up temporary files
            for path_var in ['input_path', 'audio_path']:
                path = getattr(st.session_state, path_var, None)
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception as cleanup_error:
                        logging.warning(f"File cleanup error for {path_var}: {cleanup_error}")

    # Footer
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([3, 1])
    with footer_col1:
        st.markdown("© 2025 textfromvideo.com. All rights reserved.")
    with footer_col2:
        st.markdown("Made with ❤️ in India by Mohit Kaushal")

if __name__ == "__main__":
    main()
