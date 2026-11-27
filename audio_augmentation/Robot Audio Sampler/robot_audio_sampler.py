#!/usr/bin/env python3
"""Local, continuous microphone recording into labeled, fixed-length WAV clips."""

from __future__ import annotations

import array
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
import wave


FINAL_CLIP_OPTIONS = {'Discard': 'discard', 'Save shorter': 'shorter', 'Pad with silence': 'pad'}


def feature_name(value: str) -> str:
    """Keep the user's label verbatim, while requiring a portable folder name."""
    value = value.strip()
    if not value or value in {'.', '..'} or len(value) > 100:
        raise ValueError('Enter a feature name with 1–100 characters.')
    if any(ord(c) < 32 for c in value) or re.search(r'[<>:"/\\|?*]', value):
        raise ValueError('Feature names cannot contain / \\ : * ? " < > | or control characters.')
    if value.endswith('.') or value.startswith('.'):
        raise ValueError('Feature names cannot begin or end with a dot.')
    if value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                                       *(f'COM{i}' for i in range(1, 10)),
                                       *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('That name is reserved by Windows. Choose another feature name.')
    return value


@dataclass(frozen=True)
class Settings:
    root: Path
    feature: str
    seconds: float = 2.0
    overlap: float = 1.5
    rate: int = 48000
    device: int | None = None
    final_clip: str = 'discard'

    def __post_init__(self):
        feature_name(self.feature)
        if not math.isfinite(self.seconds) or not 2 <= self.seconds <= 5:
            raise ValueError('Clip length must be between 2 and 5 seconds.')
        if not math.isfinite(self.overlap) or not 0 <= self.overlap < self.seconds:
            raise ValueError('Clip overlap must be at least 0 s and shorter than the clip length.')
        if self.rate not in (16000, 44100, 48000, 96000):
            raise ValueError('Choose a supported sample rate.')
        if self.final_clip not in FINAL_CLIP_OPTIONS.values():
            raise ValueError('Choose how to save the final clip.')

    @property
    def metadata_root(self) -> Path:
        return self.root.parent / f'{self.root.name}_metadata'


class ClipWriter:
    """Split PCM by frame count, never by a wall-clock timer."""

    def __init__(self, settings: Settings, on_clip=lambda info: None):
        self.settings = settings
        self.folder = settings.root / feature_name(settings.feature)
        self.folder.mkdir(parents=True, exist_ok=True)
        metadata_folder = settings.metadata_root / feature_name(settings.feature)
        metadata_folder.mkdir(parents=True, exist_ok=True)
        # Reserve an eight-character ID exclusively. Retry collisions, including
        # WAV files whose metadata may have been removed, instead of overwriting.
        for _ in range(32):
            self.session = uuid.uuid4().hex[:8]
            if next(self.folder.glob(f'*_{self.session}_*.wav'), None) is not None:
                continue
            self.manifest_path = metadata_folder / f'{self.session}_session.json'
            try:
                self.manifest_path.touch(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError('Could not allocate a session ID. Please try again.')
        self.target_frames = round(settings.rate * settings.seconds)
        self.target_bytes = self.target_frames * 2  # mono PCM, signed 16-bit
        # Each full clip starts with the last overlap_frames of the previous one.
        self.overlap_frames = round(settings.rate * settings.overlap)
        self.buffer = bytearray()
        self.carried_bytes = 0  # leading buffer bytes already saved in the previous clip
        self.recorded_frames = 0
        self.saved_frames = 0
        self.padded_frames = 0
        self.count = 0
        self.on_clip = on_clip
        self.manifest = {
            'session_id': self.session,
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'feature': settings.feature,
            'sample_rate_hz': settings.rate,
            'channels': 1,
            'format': 'PCM_16',
            'requested_clip_seconds': settings.seconds,
            'target_frames': self.target_frames,
            'overlap_seconds': settings.overlap,
            'overlap_frames': self.overlap_frames,
            'device_index': settings.device,
            'final_clip': settings.final_clip,
            'padding_frames': 0,
            'status': 'recording',
            'clips_saved': 0,
        }
        self._manifest()

    def _manifest(self):
        temporary = self.manifest_path.with_suffix('.json.part')
        try:
            temporary.write_text(json.dumps(self.manifest, indent=2) + '\n', encoding='utf-8')
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    def feed(self, pcm: bytes):
        if len(pcm) % 2:
            raise ValueError('PCM data must contain complete 16-bit frames.')
        self.recorded_frames += len(pcm) // 2
        view = memoryview(pcm)
        offset = 0
        while offset < len(view):
            take = min(self.target_bytes - len(self.buffer), len(view) - offset)
            self.buffer.extend(view[offset:offset + take])
            offset += take
            if len(self.buffer) == self.target_bytes:
                self._save(False)

    def _save(self, partial: bool = False, padding_frames: int = 0):
        frames = len(self.buffer) // 2
        number = self.count + 1
        suffix = '_partial' if partial else ('_padded' if padding_frames else '')
        path = self.folder / f'{feature_name(self.settings.feature)}_{self.session}_{number:06d}{suffix}.wav'
        temporary = path.with_suffix('.wav.part')
        try:
            # Exclusive temporary creation plus a unique session ID avoids overwrites.
            with temporary.open('xb') as output:
                with wave.open(output, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.settings.rate)
                    wav.writeframes(self.buffer)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.count = number
        self.saved_frames += frames
        self.padded_frames += padding_frames
        if partial or padding_frames:
            self.buffer.clear()
            self.carried_bytes = 0
        else:
            self.carried_bytes = self.overlap_frames * 2
            del self.buffer[:len(self.buffer) - self.carried_bytes]
        self.manifest['clips_saved'] = self.count
        self.manifest['frames_saved'] = self.saved_frames
        self.manifest['padding_frames'] = self.padded_frames
        self.manifest['last_clip'] = {'filename': path.name, 'frames': frames,
                                      'recorded_frames': frames - padding_frames,
                                      'padding_frames': padding_frames}
        self._manifest()
        self.on_clip({'path': str(path), 'frames': frames,
                      'seconds': frames / self.settings.rate, 'partial': partial,
                      'count': self.count, 'padding_frames': padding_frames})

    def finish(self, error: str | None = None):
        # Only audio not yet in any saved clip counts as an unfinished tail.
        discarded = (len(self.buffer) - self.carried_bytes) // 2
        if discarded and self.settings.final_clip != 'discard' and not error:
            if self.settings.final_clip == 'pad':
                padding_frames = self.target_frames - len(self.buffer) // 2
                self.buffer.extend(b'\x00\x00' * padding_frames)
                self._save(padding_frames=padding_frames)
            else:
                self._save(partial=True)
            discarded = 0
        self.buffer.clear()
        self.manifest.update(status='error' if error else 'stopped', error=error,
                             ended_utc=datetime.now(timezone.utc).isoformat(),
                             discarded_tail_frames=discarded)
        self._manifest()
        return discarded


class Recorder:
    """All microphone and file I/O happens outside Tk's UI thread."""

    def __init__(self, settings: Settings, events: queue.Queue, audio):
        self.settings = settings
        self.events = events
        self.audio = audio
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='audio-recorder', daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):
        writer = None
        error = None
        discarded = 0
        try:
            s = self.settings
            self.audio.check_input_settings(device=s.device, channels=1, dtype='int16', samplerate=s.rate)
            writer = ClipWriter(s, lambda info: self.events.put(('clip', info)))
            with self.audio.RawInputStream(device=s.device, channels=1, samplerate=s.rate,
                                          dtype='int16', blocksize=1024, latency='high') as stream:
                self.events.put(('started', str(writer.folder)))
                last_meter = 0.0
                while not self.stop_event.is_set():
                    raw, overflowed = stream.read(1024)
                    if overflowed:
                        raise RuntimeError('The microphone buffer overflowed (audio was lost). '
                                           'Recording stopped; the unfinished clip was discarded. '
                                           'Close busy apps or try a lower sample rate.')
                    # PortAudio uses native endian; WAV always uses little endian.
                    samples = array.array('h')
                    samples.frombytes(raw)
                    peak = max((abs(v) for v in samples), default=0) / 32768
                    if sys.byteorder != 'little':
                        samples.byteswap()
                    writer.feed(samples.tobytes())
                    now = time.monotonic()
                    if now - last_meter >= 0.1:
                        self.events.put(('meter', {'peak': peak,
                            'progress': len(writer.buffer) / writer.target_bytes,
                            'captured': writer.recorded_frames / s.rate}))
                        last_meter = now
        except Exception as exc:
            error = str(exc)
        finally:
            if writer is not None:
                try:
                    discarded = writer.finish(error)
                except Exception as exc:
                    error = f'{error + chr(10) if error else ""}Could not finish saving: {exc}'
            self.events.put(('done', {'error': error, 'discarded': discarded,
                                     'count': writer.count if writer else 0,
                                     'padded': writer.padded_frames if writer else 0}))


def run_app():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('Audio library missing',
            f'{exc}\n\nInstall the audio dependency first:\n'
            'python3 -m pip install -r requirements.txt\n\nSee README.txt for setup.')
        root.destroy()
        return

    class App:
        def __init__(self, root):
            self.root = root
            self.recorder = None
            self.events = queue.Queue()
            self.closing = False
            root.title('Robot Audio Sampler')
            root.geometry('740x660')
            root.minsize(640, 600)
            root.protocol('WM_DELETE_WINDOW', self.close)
            self.folder = tk.StringVar(value=str(Path(__file__).resolve().parent / 'features'))
            self.feature = tk.StringVar(value='picks up bucket')
            self.seconds = tk.StringVar(value='2.0')
            self.rate = tk.StringVar(value='48000')
            self.mic = tk.StringVar()
            self.final_clip = tk.StringVar(value='Discard')
            self.status = tk.StringVar(value='Ready — choose a feature, then press Record.')
            self.destination = tk.StringVar()
            self.stats = tk.StringVar(value='0 clips saved · 0.0 s captured')
            self.devices = {}
            main = ttk.Frame(root, padding=24)
            main.pack(fill='both', expand=True)
            main.columnconfigure(1, weight=1)
            ttk.Label(main, text='Robot Audio Sampler', font=('Helvetica', 23, 'bold')).grid(
                row=0, column=0, columnspan=3, sticky='w')
            ttk.Label(main, text='Capture labeled sounds as continuous, fixed-length WAV clips.').grid(
                row=1, column=0, columnspan=3, sticky='w', pady=(4, 22))
            ttk.Label(main, text='Features folder').grid(row=2, column=0, sticky='w', padx=(0, 16))
            self.folder_entry = ttk.Entry(main, textvariable=self.folder)
            self.folder_entry.grid(row=2, column=1, sticky='ew')
            self.browse = ttk.Button(main, text='Browse…', command=self.choose_folder)
            self.browse.grid(row=2, column=2, padx=(8, 0))
            ttk.Label(main, text='Feature').grid(row=3, column=0, sticky='w', pady=12)
            self.features = ttk.Combobox(main, textvariable=self.feature)
            self.features.grid(row=3, column=1, columnspan=2, sticky='ew')
            ttk.Label(main, text='Pick an existing feature or type a new name.').grid(
                row=4, column=1, columnspan=2, sticky='w', pady=(0, 12))
            ttk.Label(main, text='Microphone').grid(row=5, column=0, sticky='w')
            self.microphones = ttk.Combobox(main, textvariable=self.mic, state='readonly')
            self.microphones.grid(row=5, column=1, sticky='ew')
            self.refresh = ttk.Button(main, text='Refresh', command=self.refresh_devices)
            self.refresh.grid(row=5, column=2, padx=(8, 0))
            options = ttk.Frame(main)
            options.grid(row=6, column=0, columnspan=3, sticky='ew', pady=18)
            ttk.Label(options, text='Clip length (s)').pack(side='left')
            self.duration = ttk.Spinbox(options, from_=2, to=5, increment=0.5,
                                        textvariable=self.seconds, width=5)
            self.duration.pack(side='left', padx=(8, 24))
            ttk.Label(options, text='Sample rate (Hz)').pack(side='left')
            self.rates = ttk.Combobox(options, textvariable=self.rate, width=8,
                                      values=('16000', '44100', '48000', '96000'), state='readonly')
            self.rates.pack(side='left', padx=8)
            final_options = ttk.Frame(main)
            final_options.grid(row=7, column=0, columnspan=3, sticky='w')
            ttk.Label(final_options, text='Final clip on Stop').pack(side='left', padx=(0, 12))
            self.final_clips = ttk.Combobox(final_options, textvariable=self.final_clip,
                                           values=tuple(FINAL_CLIP_OPTIONS), state='readonly', width=20)
            self.final_clips.pack(side='left')
            ttk.Separator(main).grid(row=8, column=0, columnspan=3, sticky='ew', pady=18)
            ttk.Label(main, textvariable=self.destination, wraplength=650).grid(
                row=9, column=0, columnspan=3, sticky='w')
            buttons = ttk.Frame(main)
            buttons.grid(row=10, column=0, columnspan=3, sticky='ew', pady=16)
            self.record_button = ttk.Button(buttons, text='●  Record', command=self.start)
            self.record_button.pack(side='left', ipadx=22, ipady=7)
            self.stop_button = ttk.Button(buttons, text='■  Stop', command=self.stop, state='disabled')
            self.stop_button.pack(side='left', padx=10, ipadx=22, ipady=7)
            ttk.Button(buttons, text='Open folder', command=self.open_folder).pack(side='right')
            ttk.Label(main, textvariable=self.stats, font=('Helvetica', 12, 'bold')).grid(
                row=11, column=0, columnspan=3, sticky='w')
            meters = ttk.Frame(main)
            meters.grid(row=12, column=0, columnspan=3, sticky='ew', pady=10)
            meters.columnconfigure(1, weight=1)
            ttk.Label(meters, text='Input level', width=13).grid(row=0, column=0, sticky='w')
            self.level = ttk.Progressbar(meters, maximum=1)
            self.level.grid(row=0, column=1, sticky='ew')
            self.signal = ttk.Label(meters, text='—', width=17)
            self.signal.grid(row=0, column=2, padx=(8, 0))
            ttk.Label(meters, text='Current clip').grid(row=1, column=0, sticky='w', pady=8)
            self.progress = ttk.Progressbar(meters, maximum=1)
            self.progress.grid(row=1, column=1, columnspan=2, sticky='ew')
            ttk.Label(main, textvariable=self.status, wraplength=650).grid(
                row=13, column=0, columnspan=3, sticky='w')
            self.controls = [self.folder_entry, self.browse, self.features, self.microphones,
                             self.refresh, self.duration, self.rates, self.final_clips]
            self.folder.trace_add('write', lambda *_: self.update_features())
            self.feature.trace_add('write', lambda *_: self.update_destination())
            self.refresh_devices()
            self.update_features()
            self.root.after(80, self.poll)

        def update_destination(self):
            self.destination.set(f'Save to: {Path(self.folder.get()).expanduser() / self.feature.get()}')

        def update_features(self):
            try:
                folder = Path(self.folder.get()).expanduser()
                names = sorted(p.name for p in folder.iterdir()
                               if p.is_dir() and not p.name.startswith('.')) if folder.is_dir() else []
                self.features['values'] = names
            except OSError:
                self.features['values'] = ()
            self.update_destination()

        def choose_folder(self):
            selected = filedialog.askdirectory(title='Choose the parent folder for all features')
            if selected:
                self.folder.set(selected)

        def refresh_devices(self):
            selected = self.mic.get()
            self.devices = {'System default input': None}
            try:
                devices = sd.query_devices()
                hosts = sd.query_hostapis()
                for index, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        label = f'{index}: {device["name"]} ({hosts[device["hostapi"]]["name"]})'
                        self.devices[label] = index
                self.status.set('Ready — choose a feature, then press Record.' if len(self.devices) > 1
                                else 'No microphone found. Connect one and press Refresh.')
            except Exception as exc:
                self.status.set(f'Could not list microphones: {exc}')
            self.microphones['values'] = list(self.devices)
            self.mic.set(selected if selected in self.devices else next(iter(self.devices)))

        def start(self):
            if self.recorder is not None:
                return
            try:
                if not self.folder.get().strip():
                    raise ValueError('Choose a features folder first.')
                label = feature_name(self.feature.get())
                self.feature.set(label)
                settings = Settings(root=Path(self.folder.get()).expanduser().resolve(),
                                    feature=label, seconds=float(self.seconds.get()),
                                    rate=int(self.rate.get()), device=self.devices[self.mic.get()],
                                    final_clip=FINAL_CLIP_OPTIONS[self.final_clip.get()])
            except (ValueError, KeyError) as exc:
                messagebox.showerror('Check recording settings', str(exc))
                return
            self.count = 0
            self.captured = 0.0
            self.stats.set('0 clips saved · 0.0 s captured')
            self.level['value'] = self.progress['value'] = 0
            for widget in self.controls:
                widget.configure(state='disabled')
            self.record_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            self.status.set('Opening microphone…')
            self.recorder = Recorder(settings, self.events, sd)
            self.recorder.start()

        def stop(self):
            if self.recorder:
                self.recorder.stop()
                self.stop_button.configure(state='disabled')
                self.status.set('Stopping and finishing files…')

        def poll(self):
            try:
                while True:
                    kind, info = self.events.get_nowait()
                    if kind == 'started':
                        if not self.recorder.stop_event.is_set():
                            self.status.set('Recording continuously. Press Stop to finish.')
                    elif kind == 'clip':
                        self.count = info['count']
                        self.stats.set(f'{self.count} clips saved · {self.captured:.1f} s captured')
                    elif kind == 'meter':
                        self.captured = info['captured']
                        self.level['value'] = info['peak']
                        self.progress['value'] = info['progress']
                        self.signal.configure(text='Clipping!' if info['peak'] >= 0.99 else
                                              ('Very quiet' if info['peak'] < 0.003 else 'Signal detected'))
                        self.stats.set(f'{self.count} clips saved · {self.captured:.1f} s captured')
                    elif kind == 'done':
                        self.recorder = None
                        for widget in self.controls:
                            widget.configure(state='normal')
                        self.microphones.configure(state='readonly')
                        self.rates.configure(state='readonly')
                        self.final_clips.configure(state='readonly')
                        self.record_button.configure(state='normal')
                        self.stop_button.configure(state='disabled')
                        self.level['value'] = self.progress['value'] = 0
                        self.signal.configure(text='—')
                        self.update_features()
                        tail = info['discarded'] / int(self.rate.get())
                        padded = info['padded'] / int(self.rate.get())
                        self.status.set(f'Stopped. Saved {info["count"]} clips.' +
                                        (f' Discarded the final {tail:.2f} s.' if tail else '') +
                                        (f' Added {padded:.2f} s of silence to the final clip.' if padded else ''))
                        if info['error']:
                            self.status.set(f'Recording stopped with an error. {info["count"]} clips saved.')
                            messagebox.showerror('Recording stopped', info['error'] +
                                '\n\nAny WAV files already saved remain in the feature folder. '
                                'Check microphone permission and the selected sample rate.')
                        if self.closing:
                            self.root.destroy()
                            return
            except queue.Empty:
                pass
            self.root.after(80, self.poll)

        def open_folder(self):
            try:
                parent = Path(self.folder.get()).expanduser().resolve()
                try:
                    target = parent / feature_name(self.feature.get())
                except ValueError:
                    target = parent
                target = target if target.is_dir() else parent
                target.mkdir(parents=True, exist_ok=True)
                if sys.platform == 'darwin':
                    subprocess.Popen(['open', str(target)])
                elif os.name == 'nt':
                    os.startfile(str(target))
                else:
                    subprocess.Popen(['xdg-open', str(target)])
            except Exception as exc:
                messagebox.showerror('Could not open folder', str(exc))

        def close(self):
            if self.recorder:
                self.closing = True
                self.stop()
            else:
                self.root.destroy()

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    run_app()
