# data/

Place your meeting audio files here before running the notebook.

## Supported Formats
- `.wav` ✅
- `.mp3` ✅
- `.m4a` ✅
- `.ogg` ✅
- `.flac` ✅

## Note
Audio files are excluded from GitHub via `.gitignore` (too large).
The notebook downloads a sample audio automatically in Cell 3.

## To use your own audio:
1. Place your file here: `data/your_meeting.mp3`
2. In Cell 3 of the notebook, update:
   ```python
   AUDIO_PATH = "/content/your_meeting.mp3"
   ```
