#!/bin/bash

# TEACHER NOTE: The first line tells the computer this is a bash script.

echo "Setting up AI Internship Environment..."

# 1. Upgrade pip to make sure we have the latest installer
python -m pip install --upgrade pip

# 2. Install the required libraries from our list
# We use -r to read from the text file
pip install -r requirements.txt

# 3. Create a folder for our results (Good organization!)
mkdir -p outputs

echo "Setup Complete! You are ready to code."