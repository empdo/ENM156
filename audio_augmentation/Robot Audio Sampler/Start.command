#!/bin/bash
# Double-click on macOS, or run with bash on Linux.
cd -- "$(dirname -- "$0")" || exit 1
fail() {
    echo ""
    echo "Setup or launch failed. See README.txt for manual setup."
    read -r -p "Press Enter to close… "
    exit 1
}
if ! command -v python3 >/dev/null 2>&1; then
    echo "Install Python 3.10 or newer from https://www.python.org/downloads/ first."
    fail
fi
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv || fail
fi
if ! .venv/bin/python -c "import sounddevice" >/dev/null 2>&1; then
    echo "Installing the microphone library (first launch needs internet)…"
    .venv/bin/python -m pip install -r requirements.txt || fail
fi
.venv/bin/python robot_audio_sampler.py || fail
