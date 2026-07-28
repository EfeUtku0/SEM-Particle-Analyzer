#!/bin/bash
# Double-click to launch the SEM Particle Analyzer (development mode).
cd "$(dirname "$0")"
exec .venv/bin/python app/gui.py
