#!/bin/bash

# Sketch-Guided Scene Image Generation Runner
# This script provides easy-to-use examples for running the generation pipeline

echo "=== Sketch-Guided Scene Image Generation ==="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if the main script exists
if [ ! -f "sketch_guide_scene_gen_model.py" ]; then
    echo "Error: sketch_guide_scene_gen_model.py not found in current directory"
    exit 1
fi

# Function to run with example parameters
run_example() {
    local sketch_name=$1
    local global_prompt=$2
    local object_prompt=$3
    local bg_prompt=$4
    
    echo "Running example: $sketch_name"
    echo "Global prompt: $global_prompt"
    echo "Object prompt: $object_prompt"
    echo "Background prompt: $bg_prompt"
    echo ""
    
    python sketch_guide_scene_gen_model.py \
        --sketch_path "./sketches/$sketch_name" \
        --global_prompt "$global_prompt" \
        --object_prompt "$object_prompt" \
        --bg_prompt "$bg_prompt" \
        --seed 42
}

# Show available options
echo "Available examples:"
echo "1. Bread and Coffee"
echo "2. Bicycle"
echo "3. House, Tree and Car"
echo "4. Food scene"
echo "5. Custom input"
echo ""

# Get user choice
read -p "Choose an example (1-5): " choice

case $choice in
    1)
        run_example "bread_coffee.png" \
                   "A photo of a bread and a cup of tea in a tray." \
                   ". bread . cup of tea . ." \
                   "in a tray"
        ;;
    2)
        run_example "bicycle.png" \
                   "A photo of a bicycle in a park." \
                   ". bicycle . ." \
                   "in a park"
        ;;
    3)
        run_example "house-tree-car.png" \
                   "A photo of a house, tree and car in a suburban street." \
                   ". house . tree . car . ." \
                   "in a suburban street"
        ;;
    4)
        run_example "food.png" \
                   "A photo of food on a table." \
                   ". food . ." \
                   "on a table"
        ;;
    5)
        echo "Custom input mode:"
        read -p "Enter sketch filename (in sketches/ folder): " sketch_file
        read -p "Enter global prompt: " global_prompt
        read -p "Enter object prompt (format: '. object1 . object2 . .'): " object_prompt
        read -p "Enter background prompt: " bg_prompt
        
        run_example "$sketch_file" "$global_prompt" "$object_prompt" "$bg_prompt"
        ;;
    *)
        echo "Invalid choice. Please run the script again and choose 1-5."
        exit 1
        ;;
esac

echo ""
echo "Generation completed! Check the generated_images/ folder for results."
