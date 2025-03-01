#!/bin/bash

echo "LectureLogAI Installation Script"
echo "================================"
echo

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found! Please install Python 3.8 or higher."
    echo "Visit https://www.python.org/downloads/"
    exit 1
fi

# Run the setup script
echo "Running setup script..."
python3 ../../setup/setup.py
if [ $? -ne 0 ]; then
    echo "Setup failed. Please try manual installation."
    exit 1
fi

echo
echo "Installation completed!"
echo
echo "To start LectureLogAI:"
echo "1. Run \"./start.sh\""
echo

# Create a start.sh file for easy launching
cat > ../../start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python3 app/main.py
EOF

chmod +x ../../start.sh

echo "Created start.sh for easy launching."
echo 