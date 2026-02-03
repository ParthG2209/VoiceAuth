#!/usr/bin/env python3
"""
Script to convert human voice audio files to AI-generated voices.
This script:
1. Transcribes human audio using OpenAI Whisper
2. Converts the text to AI-generated speech using Edge TTS
3. Organizes output into train/test/valid folders
"""

import os
import asyncio
import random
import shutil
from pathlib import Path

# Check and install required packages
def install_packages():
    import subprocess
    import sys
    
    packages = ['edge-tts', 'openai-whisper']
    for package in packages:
        try:
            if package == 'openai-whisper':
                import whisper
            elif package == 'edge-tts':
                import edge_tts
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

install_packages()

import whisper
import edge_tts

# Configuration
DATASET_BASE = Path(__file__).parent.parent / "dataset"
LANGUAGES = ["english", "hindi", "tamil", "telugu", "malyalam"]

# AI voices for different languages (Edge TTS voices)
VOICES = {
    "english": [
        "en-US-GuyNeural",
        "en-US-JennyNeural", 
        "en-US-AriaNeural",
        "en-GB-RyanNeural",
        "en-GB-SoniaNeural",
        "en-AU-NatashaNeural",
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
    ],
    "hindi": [
        "hi-IN-MadhurNeural",
        "hi-IN-SwaraNeural",
    ],
    "tamil": [
        "ta-IN-PallaviNeural",
        "ta-IN-ValluvarNeural",
    ],
    "telugu": [
        "te-IN-MohanNeural",
        "te-IN-ShrutiNeural",
    ],
    "malyalam": [
        "ml-IN-MidhunNeural",
        "ml-IN-SobhanaNeural",
    ],
}

# Train/Test/Valid split ratios
SPLIT_RATIOS = {
    "train": 0.7,
    "valid": 0.15,
    "test": 0.15,
}


def load_whisper_model():
    """Load Whisper model for transcription."""
    print("Loading Whisper model (this may take a while on first run)...")
    model = whisper.load_model("base")
    print("Whisper model loaded!")
    return model


def transcribe_audio(model, audio_path: Path) -> str:
    """Transcribe audio file to text using Whisper."""
    try:
        result = model.transcribe(str(audio_path))
        return result["text"].strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""


async def text_to_speech(text: str, voice: str, output_path: Path) -> bool:
    """Convert text to speech using Edge TTS."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False


def get_split_assignments(files: list, ratios: dict) -> dict:
    """Assign files to train/test/valid splits."""
    random.shuffle(files)
    n = len(files)
    
    n_train = int(n * ratios["train"])
    n_valid = int(n * ratios["valid"])
    
    assignments = {}
    assignments["train"] = files[:n_train]
    assignments["valid"] = files[n_train:n_train + n_valid]
    assignments["test"] = files[n_train + n_valid:]
    
    return assignments


async def process_language(language: str, whisper_model):
    """Process all files for a specific language."""
    print(f"\n{'='*60}")
    print(f"Processing language: {language}")
    print(f"{'='*60}")
    
    ai_folder = DATASET_BASE / language / "ai"
    
    if not ai_folder.exists():
        print(f"AI folder not found for {language}, skipping...")
        return
    
    # Get all audio files in the AI folder (not in subfolders)
    audio_files = [f for f in ai_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.mp3', '.wav', '.flac', '.m4a']]
    
    if not audio_files:
        print(f"No audio files found in {ai_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files to convert")
    
    # Create output directories
    for split in ["train", "test", "valid"]:
        (ai_folder / split).mkdir(exist_ok=True)
    
    # Get voices for this language
    voices = VOICES.get(language, VOICES["english"])
    
    # Assign files to splits
    split_assignments = get_split_assignments(audio_files, SPLIT_RATIOS)
    
    # Process each split
    for split_name, files in split_assignments.items():
        print(f"\nProcessing {split_name} split ({len(files)} files)...")
        
        for i, audio_file in enumerate(files):
            # Check if already processed (resume capability)
            output_path = ai_folder / split_name / audio_file.name
            if output_path.exists():
                print(f"  [{i+1}/{len(files)}] Skipping (already exists): {audio_file.name}")
                continue
            
            # Select a random voice for variety
            voice = random.choice(voices)
            
            # Transcribe the audio
            print(f"  [{i+1}/{len(files)}] Transcribing: {audio_file.name}")
            text = transcribe_audio(whisper_model, audio_file)
            
            if not text:
                print(f"    Skipping (no transcription)")
                continue
            
            # Generate AI voice
            print(f"    Converting with voice: {voice}")
            
            success = await text_to_speech(text, voice, output_path)
            
            if success:
                print(f"    ✓ Saved to: {output_path.name}")
            else:
                print(f"    ✗ Failed to generate")
    
    # Move or delete original files
    print(f"\nCleaning up original files...")
    backup_folder = ai_folder / "original_backup"
    backup_folder.mkdir(exist_ok=True)
    
    for audio_file in audio_files:
        if audio_file.exists():
            shutil.move(str(audio_file), str(backup_folder / audio_file.name))
    
    print(f"Original files backed up to: {backup_folder}")


async def main():
    """Main function to process all languages."""
    print("="*60)
    print("AI Voice Generator for VoiceAuth Dataset")
    print("="*60)
    
    # Load Whisper model once
    whisper_model = load_whisper_model()
    
    # Process each language
    for language in LANGUAGES:
        await process_language(language, whisper_model)
    
    print("\n" + "="*60)
    print("All conversions complete!")
    print("="*60)
    print("\nSummary:")
    for language in LANGUAGES:
        ai_folder = DATASET_BASE / language / "ai"
        if ai_folder.exists():
            for split in ["train", "test", "valid"]:
                split_folder = ai_folder / split
                if split_folder.exists():
                    count = len(list(split_folder.glob("*.mp3")))
                    print(f"  {language}/{split}: {count} files")


if __name__ == "__main__":
    asyncio.run(main())
