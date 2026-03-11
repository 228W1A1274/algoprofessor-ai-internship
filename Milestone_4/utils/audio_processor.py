"""
audio_processor.py
==================
MeetScribe — Audio Processing Module
AlgoProfessor AI Internship | Milestone 4 | Day 25

Responsibilities:
    1. Preprocess audio to Whisper-compatible format (16kHz mono WAV)
    2. Transcribe audio using faster-whisper
    3. Run speaker diarisation using pyannote.audio
    4. Merge transcript segments with speaker labels
"""

import os
import time
import json
import warnings
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# 1. AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────

def preprocess_audio(audio_path: str, output_path: str = None) -> tuple[str, float]:
    """
    Convert any audio file to 16kHz mono WAV.

    Why 16kHz?
        Whisper was trained exclusively on 16kHz audio.
        Wrong sample rate → garbled transcription output.

    Supports: mp3, wav, m4a, ogg, flac, aac

    Args:
        audio_path  : Path to input audio file (any format)
        output_path : Path for output WAV file (auto-generated if None)

    Returns:
        Tuple of (processed_wav_path, duration_in_seconds)
    """
    audio_path = str(audio_path)

    if output_path is None:
        stem = Path(audio_path).stem
        output_path = str(Path(audio_path).parent / f"{stem}_16k_mono.wav")

    print(f"🔧 Preprocessing audio: {Path(audio_path).name}")

    # librosa.load handles any format and resamples in one call
    audio, original_sr = librosa.load(
        audio_path,
        sr=16000,    # Resample to 16kHz
        mono=True    # Mix stereo channels to mono
    )

    sf.write(output_path, audio, 16000)

    duration_sec = len(audio) / 16000

    print(f"   Original sample rate : {original_sr:,} Hz")
    print(f"   Output sample rate   : 16,000 Hz ✅")
    print(f"   Channels             : Mono ✅")
    print(f"   Duration             : {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print(f"   Saved to             : {output_path}")

    return output_path, duration_sec


# ─────────────────────────────────────────────────────────
# 2. WHISPER TRANSCRIPTION
# ─────────────────────────────────────────────────────────

def load_whisper_model(model_size: str = "base"):
    """
    Load the faster-whisper model.

    Model size guide:
        tiny   → 1GB RAM,  fastest,  good accuracy
        base   → 1GB RAM,  fast,     very good accuracy  ← recommended
        small  → 2GB RAM,  moderate, great accuracy
        medium → 5GB RAM,  slow,     excellent accuracy
        large  → 10GB RAM, slowest,  best accuracy

    Args:
        model_size : One of tiny / base / small / medium / large-v3

    Returns:
        Loaded WhisperModel instance
    """
    from faster_whisper import WhisperModel

    print(f"⏳ Loading Whisper '{model_size}' model...")
    print(f"   (Downloads model on first run — cached after that)")

    model = WhisperModel(
        model_size,
        device="cpu",        # Use "cuda" if GPU available
        compute_type="int8"  # 8-bit quantisation: 2x faster, half memory
    )

    print(f"✅ Whisper '{model_size}' model loaded!")
    return model


def transcribe_audio(
    audio_path: str,
    model,
    language: Optional[str] = None,
    beam_size: int = 5
) -> dict:
    """
    Transcribe audio file using faster-whisper.

    Args:
        audio_path : Path to 16kHz mono WAV file
        model      : Loaded WhisperModel instance
        language   : Language code (e.g. 'en', 'hi') or None for auto-detect
        beam_size  : Beam search width (higher = more accurate, slower)

    Returns:
        {
            'full_text': str,
            'segments' : [{'start': float, 'end': float, 'text': str}, ...],
            'language' : str,
            'duration' : float
        }
    """
    print(f"\n🎙️ Transcribing: {Path(audio_path).name}")
    print(f"   Language: {'auto-detect' if language is None else language}")
    t0 = time.time()

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        word_timestamps=True,   # Required for speaker alignment
        language=language,
        vad_filter=True,        # Skip silent regions automatically
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    segments = []
    full_text = ""

    for seg in segments_gen:
        entry = {
            "start": round(seg.start, 2),
            "end":   round(seg.end,   2),
            "text":  seg.text.strip(),
        }
        segments.append(entry)
        full_text += seg.text + " "
        print(f"   [{seg.start:6.1f}s → {seg.end:6.1f}s] {seg.text.strip()[:70]}")

    elapsed = time.time() - t0

    result = {
        "full_text": full_text.strip(),
        "segments":  segments,
        "language":  info.language,
        "duration":  info.duration,
    }

    print(f"\n✅ Transcription complete!")
    print(f"   Language detected : {info.language} ({info.language_probability:.1%} confidence)")
    print(f"   Segments          : {len(segments)}")
    print(f"   Processing time   : {elapsed:.1f}s  ({info.duration/elapsed:.1f}x realtime)")

    return result


# ─────────────────────────────────────────────────────────
# 3. SPEAKER DIARISATION
# ─────────────────────────────────────────────────────────

def run_diarisation(
    audio_path: str,
    hf_token: str,
    num_speakers: Optional[int] = None
) -> list:
    """
    Identify speakers using pyannote.audio 3.1.

    Prerequisites:
        1. HuggingFace account + token
        2. Accept model license at:
           huggingface.co/pyannote/speaker-diarization-3.1

    Args:
        audio_path   : Path to processed audio file
        hf_token     : HuggingFace access token
        num_speakers : Exact speaker count (None = auto-detect)

    Returns:
        List of {'start': float, 'end': float, 'speaker': str}
    """
    from pyannote.audio import Pipeline

    print(f"\n👥 Running speaker diarisation...")
    print(f"   Model: pyannote/speaker-diarization-3.1")
    print(f"   Speakers: {'auto-detect' if num_speakers is None else num_speakers}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    params = {}
    if num_speakers:
        params["num_speakers"] = num_speakers

    diarisation = pipeline(audio_path, **params)

    segments = []
    for turn, _, speaker in diarisation.itertracks(yield_label=True):
        segments.append({
            "start":   round(turn.start, 2),
            "end":     round(turn.end,   2),
            "speaker": speaker,
        })

    unique = set(s["speaker"] for s in segments)
    print(f"✅ Diarisation complete!")
    print(f"   Speakers found : {len(unique)} → {sorted(unique)}")
    print(f"   Segments       : {len(segments)}")

    return segments


def run_mock_diarisation(transcript_segments: list, num_speakers: int = 2) -> list:
    """
    Mock diarisation for testing without HuggingFace token.
    Alternates between N speakers based on segment index.

    Args:
        transcript_segments : Whisper transcript segments
        num_speakers        : Number of mock speakers (default 2)

    Returns:
        List of {'start': float, 'end': float, 'speaker': str}
    """
    print(f"\n⚠️  Mock diarisation active (no HF token)")
    print(f"   Alternating between {num_speakers} speakers")

    speakers = [f"SPEAKER_{i:02d}" for i in range(num_speakers)]
    segments = []

    for i, seg in enumerate(transcript_segments):
        segments.append({
            "start":   seg["start"],
            "end":     seg["end"],
            "speaker": speakers[i % num_speakers],
        })

    print(f"✅ Mock diarisation applied to {len(segments)} segments")
    return segments


# ─────────────────────────────────────────────────────────
# 4. MERGE TRANSCRIPT + SPEAKERS
# ─────────────────────────────────────────────────────────

def merge_transcript_speakers(
    transcript_segments: list,
    speaker_segments: list
) -> list:
    """
    Combine Whisper text segments with pyannote speaker labels.

    Algorithm:
        For each transcript segment, compute its midpoint timestamp.
        Find which speaker segment contains that midpoint.
        Assign that speaker label to the text block.
        Then merge consecutive blocks from the same speaker.

    Args:
        transcript_segments : Output from transcribe_audio()['segments']
        speaker_segments    : Output from run_diarisation()

    Returns:
        List of merged blocks:
        [{'speaker': str, 'start': float, 'end': float, 'text': str}, ...]
    """
    print(f"\n🔀 Merging transcript + speaker labels...")

    merged = []

    for t_seg in transcript_segments:
        # Use midpoint to find the speaker at this time
        t_mid = (t_seg["start"] + t_seg["end"]) / 2
        assigned = "UNKNOWN"

        for s_seg in speaker_segments:
            if s_seg["start"] <= t_mid <= s_seg["end"]:
                assigned = s_seg["speaker"]
                break

        merged.append({
            "speaker": assigned,
            "start":   t_seg["start"],
            "end":     t_seg["end"],
            "text":    t_seg["text"],
        })

    # Merge consecutive blocks from same speaker
    combined = []
    for seg in merged:
        if combined and combined[-1]["speaker"] == seg["speaker"]:
            combined[-1]["text"] += " " + seg["text"]
            combined[-1]["end"]   = seg["end"]
        else:
            combined.append(seg.copy())

    print(f"✅ Merge complete!")
    print(f"   Input segments  : {len(transcript_segments)}")
    print(f"   Output blocks   : {len(combined)}")
    print(f"   Speakers found  : {sorted(set(b['speaker'] for b in combined))}")

    return combined


def save_transcript(diarised: list, output_path: str = "outputs/diarised_transcript.json"):
    """Save the diarised transcript to JSON."""
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diarised, f, indent=2, ensure_ascii=False)
    print(f"💾 Transcript saved → {output_path}")


# ─────────────────────────────────────────────────────────
# 5. QUICK USAGE EXAMPLE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Usage: python audio_processor.py <audio_file> [hf_token]
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "data/sample_meeting.wav"
    hf_token   = sys.argv[2] if len(sys.argv) > 2 else None

    # Step 1: Preprocess
    processed_path, duration = preprocess_audio(audio_file)

    # Step 2: Transcribe
    whisper = load_whisper_model("base")
    transcript = transcribe_audio(processed_path, whisper)

    # Step 3: Diarise
    if hf_token:
        speakers = run_diarisation(processed_path, hf_token)
    else:
        speakers = run_mock_diarisation(transcript["segments"])

    # Step 4: Merge
    diarised = merge_transcript_speakers(transcript["segments"], speakers)

    # Step 5: Save
    save_transcript(diarised, "outputs/diarised_transcript.json")

    # Preview
    print("\n--- DIARISED TRANSCRIPT PREVIEW ---")
    for block in diarised[:5]:
        print(f"[{block['speaker']}] ({block['start']:.0f}s): {block['text'][:80]}")
