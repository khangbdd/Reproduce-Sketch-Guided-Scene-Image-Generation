# Reproduce-Sketch-Guided-Scene-Image-Generation
Reproduce for paper Sketch-Guided Scene Image Generation (Unofficial code)

## Overview
This project implements a sketch-guided scene image generation system that converts hand-drawn sketches into realistic images. The system uses a multi-stage pipeline:

1. **Object Detection & Segmentation**: Uses Grounded SAM to detect and segment objects from sketches
2. **Object Image Generation**: Uses ControlNet with scribble input to generate realistic images of detected objects
3. **Model Fine-tuning**: Fine-tunes a Stable Diffusion model on the generated objects
4. **Scene Composition**: Composes the final scene using the fine-tuned model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khangbdd/Reproduce-Sketch-Guided-Scene-Image-Generation.git
cd Reproduce-Sketch-Guided-Scene-Image-Generation
```

2. Run the setup script:
```bash
./setup.sh
```

3. Test the installation:
```bash
python test_setup.py
```

### Manual Installation

If the setup script doesn't work, you can install dependencies manually:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Setup**: Run `./setup.sh` to install dependencies
2. **Test**: Run `python test_setup.py` to verify installation
3. **Demo**: Run `python demo_cli.py --help` to see available options
4. **Generate**: Use `./run_examples.sh` for guided examples or run manually:

```bash
python sketch_guide_scene_gen_model.py \
    --sketch_path "./sketches/bread_coffee.png" \
    --global_prompt "A photo of a bread and a cup of tea in a tray." \
    --object_prompt ". bread . cup of tea . ." \
    --bg_prompt "in a tray"
```

## Usage

### Quick Start with Examples

For guided examples, simply run:
```bash
./run_examples.sh
```

This will present you with pre-configured examples and guide you through the process.

### Command Line Interface

Run the generation pipeline with the following command:

```bash
python sketch_guide_scene_gen_model.py \
    --sketch_path "./sketches/bread_coffee.png" \
    --global_prompt "A photo of a bread and a cup of tea in a tray." \
    --object_prompt ". bread . cup of tea . ." \
    --bg_prompt "in a tray"
```

### Required Arguments

- `--sketch_path`: Path to the input sketch image
- `--global_prompt`: Global prompt describing the entire scene
- `--object_prompt`: Object detection prompt with objects separated by dots
- `--bg_prompt`: Background prompt describing the setting

### Optional Arguments

- `--num_epochs`: Number of fine-tuning epochs (default: 2)
- `--lr`: Learning rate for fine-tuning (default: 1e-5)
- `--alpha`: Alpha parameter for scene composition (default: 0.5)
- `--num_inference_steps`: Number of inference steps for diffusion (default: 50)
- `--guidance_scale`: Guidance scale for diffusion (default: 7.5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to use (auto, cpu, cuda) (default: auto)

### Examples

Generate scene from bread and coffee sketch:
```bash
python sketch_guide_scene_gen_model.py \
    --sketch_path "./sketches/bread_coffee.png" \
    --global_prompt "A photo of a bread and a cup of tea in a tray." \
    --object_prompt ". bread . cup of tea . ." \
    --bg_prompt "in a tray" \
    --num_epochs 3 \
    --seed 123
```

Generate scene from bicycle sketch:
```bash
python sketch_guide_scene_gen_model.py \
    --sketch_path "./sketches/bicycle.png" \
    --global_prompt "A photo of a bicycle in a park." \
    --object_prompt ". bicycle . ." \
    --bg_prompt "in a park"
```

## Output

The system generates images in the `generated_images/` directory with timestamps. The pipeline creates:
- Segmentation masks in `generated_images/masks/`
- Individual object images in `generated_images/objects_image/`
- Final composed scene images in `generated_images/`

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- At least 8GB of GPU memory for optimal performance

## Notes

- The first run will download pre-trained models (~10GB total)
- Models are cached locally in the `cache/` directory
- Processing time varies based on image complexity and hardware (5-15 minutes per image)

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or use CPU with `--device cpu`
2. **Model download issues**: Check internet connection and disk space
3. **Sketch not detected**: Ensure sketch has clear, distinct objects and adjust object_prompt accordingly
