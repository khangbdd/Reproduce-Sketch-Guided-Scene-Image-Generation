import finetuning_stable
import controlnet_scribble
import grounded_sam
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import sys

device =  "cuda" if torch.cuda.is_available() else "cpu"

def sketches_guide_scene_gen(
        sketch_path, 
        global_prompt, 
        object_prompt, 
        bg_prompt,
        num_epochs= 50,
        lr = 1e-5,
        alpha = 0.8,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        seed=42,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
    
    # Image processing constants
    image_size = 512
    
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
        ]
    )
    image_preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]
    )
    sketch = preprocess(Image.open(sketch_path).convert("RGB"))
    masks,labels = grounded_sam.grounded_sam_pipeline(sketch, object_prompt, device)

    # get each sketch object iamge
    isolated_sketch_objects = grounded_sam.isolate_objects_on_full_canvas(sketch, masks, labels)

    # gen images from each sketch object image.
    gen_object_images, object_promts = controlnet_scribble.gen_images_by_controlnet_scribble(
        isolated_sketch_objects, 
        labels
    )

    # grouping object and it's mask to 1 image again. 
    grouped_objects, grouped_objects_mask, object_masks = grounded_sam.grouping_objects(gen_object_images, labels, device)

    finetuning_stable.finetuning_sd_model(
        gen_object_images, 
        object_masks, 
        object_promts,
        num_epochs= num_epochs,
        lr = lr,
        seed= seed
    )
    finetuning_stable.scene_level_construction(
        grouped_gen_object = grouped_objects,
        grouped_gen_mask = grouped_objects_mask,
        bg_prompt = bg_prompt,
        gl_prompt = global_prompt,
        alpha = alpha,
        num_inference_steps = num_inference_steps,
        device = device,
        guidance_scale = guidance_scale,
        seed = seed
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sketch-Guided Scene Image Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--sketch_path", 
        type=str, 
        required=True,
        help="Path to the input sketch image"
    )
    parser.add_argument(
        "--global_prompt", 
        type=str, 
        required=True,
        help="Global prompt describing the entire scene (e.g., 'A photo of a bread and a cup of tea in a tray.')"
    )
    parser.add_argument(
        "--object_prompt", 
        type=str, 
        required=True,
        help="Object detection prompt with objects separated by dots (e.g., '. bread . cup of tea . .')"
    )
    parser.add_argument(
        "--bg_prompt", 
        type=str, 
        required=True,
        help="Background prompt describing the setting (e.g., 'in a tray')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=50,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-5,
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.8,
        help="Alpha parameter for scene composition"
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=50,
        help="Number of inference steps for diffusion"
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=7.5,
        help="Guidance scale for diffusion"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation"
    )
    
    return parser.parse_args()

def validate_inputs(args):
    """Validate input arguments."""
    # Check if sketch file exists
    if not os.path.exists(args.sketch_path):
        print(f"Error: Sketch file '{args.sketch_path}' does not exist.")
        sys.exit(1)
    
    # Check if sketch file is an image
    try:
        Image.open(args.sketch_path)
    except Exception as e:
        print(f"Error: Cannot open '{args.sketch_path}' as an image: {e}")
        sys.exit(1)
    
    # Validate device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    return args

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validate inputs
    args = validate_inputs(args)
    
    print("Starting Sketch-Guided Scene Image Generation...")
    print(f"Sketch path: {args.sketch_path}")
    print(f"Global prompt: {args.global_prompt}")
    print(f"Object prompt: {args.object_prompt}")
    print(f"Background prompt: {args.bg_prompt}")
    print(f"Device: {args.device}")
    
    try:
        # Run the main pipeline
        sketches_guide_scene_gen(
            sketch_path=args.sketch_path,
            global_prompt=args.global_prompt,
            object_prompt=args.object_prompt,
            bg_prompt=args.bg_prompt,
            num_epochs=args.num_epochs,
            lr=args.lr,
            alpha=args.alpha,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device=args.device
        )
        print("Generation completed successfully!")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)
