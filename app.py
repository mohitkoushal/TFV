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

def get_video_info(url):
    """Get available formats for a YouTube/Instagram video with specific quality options"""
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                st.error("Could not retrieve video information")
                return None
            
            formats = []
            if 'formats' in info:
                # Filter only formats that have both video and audio
                for f in info['formats']:
                    if (f.get('ext') in ['mp4', 'webm'] and 
                        f.get('vcodec') != 'none' and 
                        f.get('acodec') != 'none'):
                        
                        # Get resolution
                        resolution = f.get('resolution', 'unknown')
                        height = 0
                        if 'x' in resolution:
                            height = int(resolution.split('x')[1])
                        
                        # Categorize into our desired quality levels
                        quality = None
                        if height >= 720:
                            quality = 'HD (720p+)'
                        elif height >= 480:
                            quality = '480p'
                        elif height >= 320:
                            quality = '320p'
                        
                        if quality:
                            formats.append({
                                'format_id': f['format_id'],
                                'ext': f['ext'],
                                'resolution': resolution,
                                'quality': quality,
                                'height': height,
                                'fps': f.get('fps', 0),
                                'filesize': f.get('filesize', 0),
                                'url': f.get('url')
                            })
            
            # Sort formats by quality (HD first)
            quality_order = {'HD (720p+)': 0, '480p': 1, '320p': 2}
            formats.sort(key=lambda x: quality_order[x['quality']])
            
            # Group by quality and select best format for each quality
            quality_groups = {}
            for fmt in formats:
                if fmt['quality'] not in quality_groups:
                    quality_groups[fmt['quality']] = fmt
            
            # Get our desired quality formats
            desired_qualities = ['HD (720p+)', '480p', '320p']
            formats = [quality_groups[q] for q in desired_qualities if q in quality_groups]
            
            # For Instagram, sometimes we need to handle differently
            if not formats and 'url' in info:
                formats.append({
                    'format_id': '0',
                    'ext': 'mp4',
                    'resolution': 'sd',
                    'quality': 'SD',
                    'height': 360,
                    'fps': 30,
                    'filesize': 0,
                    'url': info['url']
                })
            
            return {
                'title': info.get('title', 'video'),
                'thumbnail': info.get('thumbnail'),
                'duration': info.get('duration', 0),
                'formats': formats
            }
    except Exception as e:
        st.error(f"Error getting video info: {e}")
        return None

def download_video_with_quality(url, format_id, output_path):
    """Download video with specific quality"""
    ydl_opts = {
        'format': format_id,
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

def format_duration(duration_seconds):
    """Format duration in seconds to MM:SS or HH:MM:SS"""
    try:
        duration_seconds = float(duration_seconds)
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    except:
        return "Unknown"

def main():
    st.set_page_config(
        page_title="Multilingual Video Text Extractor", 
        page_icon="🌐",
        layout="wide"
    )
    st.markdown("""
    <meta name="google-adsense-account" content="ca-pub-7220800899817072">
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

    # Main input section - new tab layout
    tab1, tab2 = st.tabs(["Text Extraction", "Video Downloader"])
    
    with tab1:
        # Original text extraction functionality
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
                help="Paste a YouTube or Instagram Reel URL to extract text from the video",
                key="youtube_url_extract"
            )

        if st.button("Extract Text", type="primary", key="extract_text_btn"):
            # Reset session state
            st.session_state.input_path = None
            st.session_state.audio_path = None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Validate input
                if not uploaded_file and not youtube_url:
                    st.error("Please upload a file or provide a URL")
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
                    status_text.text("Downloading video...")
                    st.session_state.input_path = download_video(youtube_url)
                    if not st.session_state.input_path:
                        st.error("Failed to download video")
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
                result_tab1, result_tab2 = st.tabs(["Audio Transcription", "Video Text Extraction"])
                
                with result_tab1:
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
                
                with result_tab2:
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

    with tab2:
        # Video Downloader functionality
        st.subheader("Video Downloader")
        st.markdown("Download videos in specific quality options with audio")
        
        download_url = st.text_input(
            "Enter YouTube/Instagram Video URL",
            placeholder="https://www.youtube.com/watch?v=... or https://www.instagram.com/reels/...",
            key="download_url"
        )
        
        if st.button("Get Download Options", key="get_options_btn"):
            if not download_url:
                st.error("Please enter a valid YouTube or Instagram URL")
            else:
                with st.spinner("Fetching video information..."):
                    video_info = get_video_info(download_url)
                    
                    if video_info:
                        st.session_state.video_info = video_info
                        
                        # Display video info
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            if video_info['thumbnail']:
                                st.image(video_info['thumbnail'], caption="Video Thumbnail", width=200)
                        
                        with col2:
                            st.markdown(f"**Title:** {video_info['title']}")
                            formatted_duration = format_duration(video_info['duration'])
                            st.markdown(f"**Duration:** {formatted_duration}")
                            
                            if video_info['formats']:
                                st.success(f"Found {len(video_info['formats'])} quality options with audio")
                            else:
                                st.warning("No downloadable formats found")
        
        if 'video_info' in st.session_state and st.session_state.video_info:
            st.subheader("Available Quality Options")
            
            # Display download buttons for each quality
            for fmt in st.session_state.video_info['formats']:
                file_size = fmt.get('filesize', 0)
                size_mb = file_size / (1024 * 1024) if file_size and file_size > 0 else "Unknown"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    **Quality:** {fmt['quality']}  
                    **Resolution:** {fmt['resolution']}  
                    **Size:** {size_mb if isinstance(size_mb, str) else f'{size_mb:.1f} MB'}
                    """)
                with col2:
                    # Create a download button for each quality
                    output_filename = f"{st.session_state.video_info['title']}_{fmt['quality']}.{fmt['ext']}"
                    output_filename = re.sub(r'[^\w\-_. ]', '_', output_filename)
                    
                    if st.button(f"Download {fmt['quality']}", key=f"dl_{fmt['format_id']}"):
                        with st.spinner(f"Preparing {fmt['quality']} download..."):
                            temp_dir = tempfile.mkdtemp()
                            output_path = os.path.join(temp_dir, output_filename)
                            
                            if download_video_with_quality(download_url, fmt['format_id'], output_path):
                                with open(output_path, 'rb') as f:
                                    st.download_button(
                                        label="Click to Save Video",
                                        data=f,
                                        file_name=output_filename,
                                        mime=f"video/{fmt['ext']}",
                                        key=f"save_{fmt['format_id']}"
                                    )
                            else:
                                st.error("Download failed")


    # Footer
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([3, 1])
    with footer_col1:
        st.markdown("© 2025 textfromvideo.com. All rights reserved.")
    with footer_col2:
        st.markdown("Made with ❤️ in India by Mohit Kaushal")

if __name__ == "__main__":
    main()