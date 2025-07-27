Multilingual Video Text Extractor
A Streamlit-based web application for extracting text from videos and audio files in multiple languages, with support for YouTube/Instagram URL downloads and advanced transcription capabilities.
Features

Multilingual Transcription: Extracts audio text using the faster_whisper model, supporting languages like English, Hindi, Spanish, French, German, Chinese, Arabic, Russian, Portuguese, Japanese, Korean, and more.
Video Text Extraction (OCR): Extracts text from video frames using Tesseract OCR with advanced preprocessing for improved accuracy.
YouTube/Instagram Download: Downloads videos or audio from YouTube and Instagram Reels in various quality options (1080p, 720p, 480p, 360p, or MP3).
Processing Modes: Offers three transcription modes:
Fast (lower accuracy)
Balanced
High Accuracy (slower)


Hindi Optimization: Includes specific optimizations for Hindi transcription and post-processing.
User-Friendly Interface: Built with Streamlit, featuring tabs for text extraction and video downloading, progress bars, and clipboard functionality.

Prerequisites
Before running the application, ensure you have the following installed:

Python: Version 3.8 or higher
Tesseract OCR: Required for text extraction from video frames. Install it based on your operating system:
Windows: Download and install from Tesseract at UB Mannheim.
Linux: sudo apt-get install tesseract-ocr
MacOS: brew install tesseract


FFmpeg: Required for video/audio processing. Install it:
Windows: Download from FFmpeg website and add to PATH.
Linux: sudo apt-get install ffmpeg
MacOS: brew install ffmpeg



Installation

Clone the Repository:
git clone https://github.com/your-username/multilingual-video-text-extractor.git
cd multilingual-video-text-extractor


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Create a requirements.txt file with the following:
streamlit
moviepy
opencv-python
pytesseract
yt-dlp
pydub
faster-whisper
torch
pyperclip
numpy


Set Up Tesseract Path:

Ensure Tesseract is installed and its path is correctly set in the script (default paths are included for Windows, Linux, and MacOS).
Update the tesseract_paths list in the code if Tesseract is installed in a non-standard location.



Usage

Run the Application:
streamlit run app.py


Access the Web Interface:

Open your browser and navigate to http://localhost:8501.


Text Extraction:

Upload a File: Use the file uploader to select a video (MP4, AVI, MOV, MKV) or audio (WAV, MP3, M4A, OGG) file.
Enter a URL: Paste a YouTube or Instagram Reel URL to extract text.
Select a language (or use "Auto-detect") and processing mode, then click "Extract Text".
Results are displayed in two tabs: Audio Transcription and Video Text Extraction (OCR).
Download or copy the extracted text to the clipboard.


Video/Audio Download:

Go to the "Video Downloader" tab.
Enter a YouTube or Instagram URL and click "Get Download Options".
Select from available quality options (1080p, 720p, 480p, 360p, or MP3) and download the file.



Configuration

Processing Mode: Choose from Fast, Balanced, or High Accuracy modes in the sidebar to balance speed and transcription quality.
Language Selection: Select a specific language or use "Auto-detect" for automatic language detection.
Logging: The application logs errors and warnings to video_text_extractor.log for debugging.

Notes

Hindi Support: The application includes optimizations for Hindi, such as keyword detection and specific transcription parameters.
Temporary Files: Temporary video/audio files are automatically cleaned up after processing.
Google AdSense: The code includes placeholders for Google AdSense integration (disabled by default). Replace YOUR_AD_CLIENT_ID with your actual AdSense client ID if desired.
Pyodide Compatibility: The application is designed for local execution but can be adapted for Pyodide if needed (e.g., for browser-based deployment).

Limitations

Requires a stable internet connection for downloading videos from URLs.
Transcription accuracy depends on audio quality and background noise.
OCR results may vary based on video quality and text visibility.
Large video files or high-resolution downloads may require significant disk space and processing time.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Create a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or support, contact Mohit Kaushal at mkmoney09@gmail.com or open an issue on GitHub.
© 2025 textfromvideo.com. Made with ❤️ in India by Mohit Kaushal.
