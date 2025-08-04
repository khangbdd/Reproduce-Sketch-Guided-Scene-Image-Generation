#!/bin/bash

# Setup script for Sketch-Guided Scene Image Generation
echo "=== Setting up Sketch-Guided Scene Image Generation ==="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "Warning: Python 3.8+ is recommended. Current version: $python_version"
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    echo "Please install pip and try again."
    exit 1
fi

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "You can now run the generation pipeline:"
    echo "  ./run_examples.sh    # For guided examples"
    echo "  python sketch_guide_scene_gen_model.py --help    # For manual usage"
    echo ""
    echo "Example usage:"
    echo "  python sketch_guide_scene_gen_model.py \\"
    echo "      --sketch_path \"./sketches/bread_coffee.png\" \\"
    echo "      --global_prompt \"A photo of a bread and a cup of tea in a tray.\" \\"
    echo "      --object_prompt \". bread . cup of tea . .\" \\"
    echo "      --bg_prompt \"in a tray\""
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    echo "You may need to:"
    echo "  - Install Python development headers"
    echo "  - Update pip: pip install --upgrade pip"
    echo "  - Install with conda if you're using Anaconda"
    exit 1
fi
