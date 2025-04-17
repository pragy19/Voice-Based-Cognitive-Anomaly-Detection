#!/bin/bash

# Update and install ffmpeg
apt-get update
apt-get install -y ffmpeg

# Install dependencies from requirements.txt
pip install -r requirements.txt
