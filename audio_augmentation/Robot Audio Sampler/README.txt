ROBOT AUDIO SAMPLER
===================

A small local Python app for collecting labeled robot-arm audio.
Audio stays on your computer. Nothing is uploaded.

SETUP (Python 3.10 or newer)

Easiest: extract the ZIP, then double-click Start.command on macOS or
Start.bat on Windows. Python 3.10+ must already be installed. The launcher
creates a local .venv environment and installs the audio library on first
launch, which needs internet. Later launches use that environment.
On Linux, run: bash Start.command
If double-clicking is unavailable, use the manual commands below.

Open a terminal in this folder. On macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt
    python3 robot_audio_sampler.py

On Windows, open PowerShell in this folder:

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe robot_audio_sampler.py

The macOS and Windows pip packages normally include PortAudio. Linux may
need the system packages python3-tk and libportaudio2 (package names vary).
Python must include Tkinter; check with: python3 -m tkinter
Official Python installers at https://www.python.org/downloads/ include Tk.

NEXT TIME

macOS/Linux: activate .venv, then run python3 robot_audio_sampler.py again.
Windows: run .\.venv\Scripts\python.exe robot_audio_sampler.py again.

USING IT

1. Browse to the parent folder where you want to keep all features.
2. Choose an existing feature from the list, or type a new name, such as
   "picks up bucket". Its folder is created when recording starts.
3. Select your microphone. Refresh lists currently available inputs.
4. Choose a clip length from 2 to 5 seconds (default: 2). You can type a
   decimal value such as 2.5. Default audio is 48,000 Hz, mono, 16-bit WAV.
5. Press Record. After the microphone opens, capture runs continuously.
   Every full clip is saved immediately; the microphone stays open between
   clips. Settings are locked during recording so labels cannot change.
   Consecutive clips overlap by 1.5 seconds: each clip begins with the last
   1.5 s of the previous one, so with 2-second clips a new clip is saved
   every 0.5 s.
6. Choose "Final clip on Stop", then press Record and Stop as usual:
   - Discard (default): discard the final incomplete clip.
   - Save shorter: keep it at its recorded length, with "_partial" in its name.
   - Pad with silence: keep the recorded audio and append digital silence
     to reach the selected clip length (2 seconds by default), with "_padded"
     in its name. For example, 1.3 s of audio gets 0.7 s of silence at 2 s.
     Choosing 5-second clips pads the last clip to 5 seconds instead.
   Padding is added immediately; the microphone does not keep recording
   to fill the clip. No extra silent clip is created if there is no tail.

Example output:

    features/
        picks up bucket/
            picks up bucket_a3f91c2d_000001.wav
            picks up bucket_a3f91c2d_000002_padded.wav
        moves left/
            moves left_6b82d04e_000001.wav
    features_metadata/
        picks up bucket/
            a3f91c2d_session.json
        moves left/
            6b82d04e_session.json

Each press of Record creates an eight-character session identifier. Existing
session IDs are checked and collisions retried so recordings are not
overwritten. New JSON files go outside the audio folder, in a sibling
folder named <audio-folder-name>_metadata, grouped by feature. For example,
choosing /dataset/features saves WAV files under /dataset/features and
JSON files under /dataset/features_metadata. Previously saved files are
left where they are.
Feature names become folder names and prefix
every clip filename, including spaces. Each WAV contains exactly
round(sample_rate * clip_length) frames,
except an optional final partial file. Stop finishes the current input
block (about 21 ms at 48 kHz), then closes the microphone and finishes saving.
Actual microphone opening/closing time depends on the driver.

The session JSON records the feature, sample format, chosen duration,
device index, UTC start/end timestamps, saved count, and recording status.
It also records the final-clip setting and the number of added silence
frames; the last clip's entry distinguishes recorded audio from padding.
Full WAV files are finalized via a temporary file before appearing as .wav.
If the app crashes or loses power, previous completed WAV files remain;
the unsaved current clip is lost. A leftover .part file is incomplete.
A session marked "recording" after a crash was not closed normally.

TROUBLESHOOTING

- Grant microphone access to the app hosting Python (for example Terminal).
  On macOS: System Settings > Privacy & Security > Microphone.
- If the microphone rejects 48000 Hz, try 44100 or another supported rate.
  Keep the same rate across the dataset when possible.
- If audio is lost due to an input-buffer overflow, the app stops and
  reports it, discarding the current unfinished clip. Already completed
  clips remain valid. Close busy apps, avoid slow network drives, and retry.
- A file-saving failure also stops recording and displays an error.
- "Very quiet" and "Clipping!" are simple peak-level indicators. Adjust
  microphone gain and placement using your system or interface controls.

DATA COLLECTION NOTES

- This splits elapsed audio into overlapping windows; it does not detect
  robot actions or align clip boundaries to an action automatically.
- Start when the selected action is happening. Capture idle/background as
  a separate feature if your model needs to recognize those conditions.
- For repeatable measurements, keep mic position and gain consistent and
  disable automatic gain/noise suppression in your device settings if
  available. This program applies no normalization or audio effects; the
  operating system and microphone may still process the signal.
- Keep all clips from one recording session in the same train/validation/
  test split to reduce leakage from neighboring audio windows. The session
  ID in each filename makes grouping possible.

Implementation uses Python's wave module and python-sounddevice:
https://python-sounddevice.readthedocs.io/en/latest/api/raw-streams.html
