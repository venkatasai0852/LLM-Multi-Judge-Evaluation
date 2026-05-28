#!/bin/bash

echo "============================================================"
echo "Creating Python Virtual Environment 'env' (Linux/macOS)"
echo "============================================================"
python3 -m venv env
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    echo "You may need to install python3-venv (e.g. 'sudo apt-get install python3-venv' on Debian/Ubuntu)."
    exit 1
fi

echo ""
echo "============================================================"
echo "Activating Virtual Environment..."
echo "============================================================"
source env/bin/activate

echo ""
echo "============================================================"
echo "Upgrading Pip..."
echo "============================================================"
python3 -m pip install --upgrade pip

echo ""
echo "============================================================"
echo "Installing Dependencies from requirements.txt..."
echo "============================================================"
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo "To activate this environment in the future, run: source env/bin/activate"
echo "You can run the code now using: python main.py --model llama-3.1-8b"
echo ""
chmod +x main.py 2>/dev/null || true
