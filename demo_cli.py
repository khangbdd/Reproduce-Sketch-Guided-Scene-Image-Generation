#!/usr/bin/env python3
"""
Demo script to show the command-line interface without running the full pipeline.
This is useful for testing the CLI arguments without needing all dependencies installed.
"""

import argparse
import sys
import os

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sketch-Guided Scene Image Generation (Demo Mode)",
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
        help="Global prompt describing the entire scene"
    )
    parser.add_argument(
        "--object_prompt", 
        type=str, 
        required=True,
        help="Object detection prompt with objects separated by dots"
    )
    parser.add_argument(
        "--bg_prompt", 
        type=str, 
        required=True,
        help="Background prompt describing the setting"
    )
    
    # Optional arguments
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha parameter for scene composition")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for diffusion")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for diffusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for computation")
    
    return parser.parse_args()

def main():
    """Demo main function."""
    args = parse_arguments()
    
    print("🎨 Sketch-Guided Scene Image Generation - Demo Mode")
    print("=" * 60)
    print("")
    print("Configuration:")
    print(f"  Sketch path: {args.sketch_path}")
    print(f"  Global prompt: {args.global_prompt}")
    print(f"  Object prompt: {args.object_prompt}")
    print(f"  Background prompt: {args.bg_prompt}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print("")
    
    # Check if sketch file exists
    if not os.path.exists(args.sketch_path):
        print(f"❌ Error: Sketch file '{args.sketch_path}' does not exist.")
        return 1
    else:
        print(f"✅ Sketch file found: {args.sketch_path}")
    
    print("")
    print("✨ Configuration validated successfully!")
    print("💡 To run the actual generation, use: python sketch_guide_scene_gen_model.py [arguments]")
    print("")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
