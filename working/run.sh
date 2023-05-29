#!/bin/bash
file=${1:-"train"}

clear
sudo rm -r "/tmp/pymp*"
jupyter nbconvert --to script "$file.ipynb"
python "$file.py"
