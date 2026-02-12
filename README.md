🎬 Audio/Video Transcription to SRT with Chunked Upload

A complete web application for audio/video transcription and subtitle generation,this was originaly vibecoded with deepseek and now is managed by google julles, with support for massive files (up to 50GB), automatic translation into 30+ languages, and a user-friendly interface.
✨ Key Features
📁 Smart Upload

    Simple upload – for small files (<500MB)

    Chunked upload – for large files, up to 50GB

    10MB chunks with resume capability

    Progress bar, estimated speed, individual chunk indicators

🧠 Multiple Whisper Models

    Supports all Whisper models: tiny, base, small, medium, large, large-v3

    Dynamic loading on demand, cached in memory for reuse

    Automatic CUDA detection (GPU) + CPU fallback

    Works with both audio and video files (automatically extracts audio stream)

🌍 Automatic Multilingual Translation

    Translates subtitles into 30+ languages using NLLB-200 and MarianMT/Opus-MT models

    Multiple translations per file – create separate subtitle sets for different languages

    Preserves original timestamps with intelligent segmentation adjustment

    Post-transcription translation – add new languages anytime

🎥 Video Preview with Subtitles

    In-browser video playback with synchronized subtitles

    Static image preview for large video files

    Automatic MP4 conversion for browser playback

    Current segment highlighting + auto-scroll

✏️ Editing & Export

    Direct text editing of transcribed content

    Save modifications to server

    Export to SRT format (standard subtitles)

    Full transcription text view

⚡ Optimized for Large Files

    Processes files >1GB by splitting audio into chunks

    Efficient memory management

    Background processing with real-time status updates

    Automatic cleanup of temporary files

🖥️ System Information

    Real-time GPU/CPU monitoring

    Model loading status

    Memory usage tracking

    CUDA availability check

🚀 Quick Start
Prerequisites

    Python 3.8+

    FFmpeg (required for audio extraction)

    CUDA-capable GPU (optional, for faster processing)

Installation

    Clone the repository

bash

git clone https://github.com/yourusername/whisper-transcriber-chunked.git
cd whisper-transcriber-chunked

    Install dependencies

bash

pip install -r requirements.txt

    Run the application

bash

python app.py

    Open your browser

text

http://localhost:5000

📦 Requirements
txt

flask>=2.3.0
openai-whisper>=20231117
torch>=2.0.0
transformers>=4.35.0
sentencepiece>=0.1.99
protobuf>=3.20.0
ffmpeg-python>=0.2.0
psutil>=5.9.0
werkzeug>=2.3.0

🎯 Usage Guide
1. Select Upload Method

    Simple Upload – Files under 500MB

    Chunked Upload – Files up to 50GB (recommended for large videos)

2. Choose Whisper Model
Model	Size	Speed	Quality
tiny	39MB	⚡⚡⚡⚡⚡	⭐⭐
base	74MB	⚡⚡⚡⚡	⭐⭐⭐
small	244MB	⚡⚡⚡	⭐⭐⭐⭐
medium	769MB	⚡⚡	⭐⭐⭐⭐⭐
large	1.55GB	⚡	⭐⭐⭐⭐⭐⭐
large-v3	1.55GB	⚡	⭐⭐⭐⭐⭐⭐⭐
3. Set Language & Segmentation

    Auto-detect – Let Whisper identify the language

    Manual selection – Choose from 30+ languages

    Segmentation tuning – Adjust min/max duration and character limits

4. Enable Translation (Optional)

    Select target language from 30+ options

    Add multiple translations to the same file

    Switch between original and translated subtitles

5. Upload & Process

    Click "Start Transcription"

    Monitor progress in real-time

    Once complete, edit, download, or play with subtitles

🌍 Supported Languages
Transcription (Whisper)

Auto-detect + 30+ languages including: English, Romanian, French, German, Spanish, Italian, Russian, Japanese, Chinese, Arabic, and more.
Translation (NLLB-200 / MarianMT)

30+ target languages including:

    English, Romanian, French, German, Spanish, Italian

    Russian, Chinese, Japanese, Korean, Arabic

    Hindi, Portuguese, Dutch, Polish, Turkish

    Swedish, Danish, Finnish, Norwegian

    Czech, Slovak, Slovenian, Hungarian

    Bulgarian, Greek, Ukrainian, Vietnamese

    Thai, Hebrew, Indonesian, Malay, Persian, Urdu, Swahili

🧠 Architecture
text

┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐
│   File Upload   │────▶│  Chunk Assembly │────▶│  Audio/Video  │
│  (Simple/Chunk) │     │   (Up to 50GB)  │     │   Processing  │
└─────────────────┘     └─────────────────┘     └────────┬──────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐
│  Subtitle Sync  │◀────│   Translation   │◀────│    Whisper    │
│  & Video Player │     │  (30+ languages)│     │  Transcription│
└─────────────────┘     └─────────────────┘     └───────────────┘

Key Components

    Flask – Web server and API endpoints

    Whisper – Speech-to-text transcription

    Transformers – NLLB-200 & MarianMT for translation

    FFmpeg – Audio extraction and video conversion

    Threading – Background processing with status tracking

    JSON storage – Persistent segment data per process

📁 Project Structure
text

whisper-transcriber-chunked/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
├── templates/
│   └── index.html        # Main interface
└── data/                 # Auto-created at runtime
    ├── uploads/          # Processed files
    └── chunk_uploads/    # Temporary chunk storage

🔧 Configuration

Edit app.py to customize:
python

# File size limits
app.config['MAX_FILE_SIZE'] = 50 * 1024 * 1024 * 1024  # 50GB
app.config['CHUNK_SIZE'] = 10 * 1024 * 1024            # 10MB chunks
app.config['SIMPLE_UPLOAD_LIMIT'] = 500 * 1024 * 1024  # 500MB

# Processing
app.config['PROCESS_TIMEOUT'] = 7200  # 2 hours timeout

# Default model
DEFAULT_MODEL = 'small'

🐛 Troubleshooting
"FFmpeg not found"

Install FFmpeg:

    Ubuntu/Debian: sudo apt install ffmpeg

    macOS: brew install ffmpeg

    Windows: Download from ffmpeg.org

"CUDA out of memory"

    Use a smaller model (tiny/base)

    Enable CPU fallback (automatic)

    Reduce chunk size in large file processing

"File too large for simple upload"

    Switch to chunked upload method

    Files over 500MB require chunked upload

🤝 Contributing

Contributions are welcome! Here's how you can help:

    Fork the repository

    Create a feature branch (git checkout -b feature/amazing-feature)

    Commit your changes (git commit -m 'Add some amazing feature')

    Push to the branch (git push origin feature/amazing-feature)

    Open a Pull Request

Ideas for contributions:

    Add support for more translation models

    Implement subtitle format export (VTT, ASS, etc.)

    Add user authentication and persistent storage

    Create Docker container

    Add unit tests

    Improve UI/UX

📄 License

This project is open source and available under the MIT License.
text

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...

Free for personal and commercial use. Do whatever you want with it!
🙏 Acknowledgments

    OpenAI Whisper – Amazing speech recognition

    Hugging Face Transformers – NLLB-200 & MarianMT

    FFmpeg – Audio/video processing

    Flask – Lightweight web framework

📊 Performance Tips
For very large files (>5GB):

    Use large-v3 model only on high-end GPUs

    Enable chunked processing (automatic for files >1GB)

    Extract audio only first, then transcribe

    Use extract_audio_only option to separate steps

For best translation quality:

    Use MarianMT models when available (specific language pairs)

    Fallback to NLLB-200 for rare language combinations

    Translate after transcription (faster, reusable)

For CPU-only systems:

    Stick to tiny or base models

    Expect longer processing times

    Disable translation for very long files

💬 Support

    Issues: GitHub Issues

    Discussions: GitHub Discussions

    Email: your-email@example.com

🌟 Star History

If you find this project useful, please star it on GitHub! It helps others discover it.

Happy transcribing! 🎬🎙️📝
