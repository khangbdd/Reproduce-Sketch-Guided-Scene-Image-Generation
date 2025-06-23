import torch
import requests
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
import utils
import numpy as np

def grounded_sam_pipeline(image, text_prompt, device = "cpu"):
    """
    A complete pipeline to perform Grounded SAM segmentation using only Hugging Face transformers.
    This function loads models, processes an image with a text prompt, and visualizes the results.
    """
    # --- 1. Setup: Load Models and Processors from Hugging Face ---
    # Load Grounding DINO model & processor
    # This model is responsible for detecting objects based on a text prompt.
    print("Loading Grounding DINO model...")
    grounding_dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    print("Grounding DINO model loaded.")

    # Load Segment-Anything-Model (SAM) model & processor
    # This model is responsible for generating high-quality masks from prompts (like bounding boxes).
    print("Loading Segment Anything Model (SAM)...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    print("SAM loaded.")

    # --- 3. Stage 1: Object Detection with Grounding DINO ---
    
    print("\n--- Running Grounding DINO for object detection ---")
    # Process the image and text prompt
    inputs = grounding_dino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = grounding_dino_model(**inputs)

    # Post-process the results to get bounding boxes
    results = grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.2,
        target_sizes=[image.size[::-1]] # (height, width)
    )
    
    # Extract detected boxes, scores, and labels
    # We send these boxes to SAM as prompts
    detection_boxes = results[0]["boxes"]
    detection_labels = results[0]["labels"]
    detection_scores = results[0]["scores"]
    print(f"Detected {len(detection_boxes)} objects.")
    for label, box, score in zip(detection_labels, detection_boxes, detection_scores):
        print(f"- Label: '{label}', Box: {box.tolist()}, Score: {score}")


    # --- 4. Stage 2: Segmentation with SAM ---
    
    if len(detection_boxes) == 0:
        print("\nNo objects detected, skipping SAM.")
        return

    print("\n--- Running SAM for segmentation ---")
    # Prepare the bounding boxes for SAM. It expects a list of lists of boxes.
    # Each inner list can contain multiple boxes for a single image.
    sam_input_boxes = [[box.tolist() for box in detection_boxes]]
    print(f"sam_input_boxes: {sam_input_boxes}")

    # Process the image and bounding box prompts for SAM
    sam_inputs = sam_processor(image, input_boxes=sam_input_boxes, return_tensors="pt").to(device)

    # Generate masks with SAM
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    # Post-process SAM's output to get the final segmentation masks
    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu()
    )[0]

    # Overlay masks
    i = 0
    for mask in masks:    
        numpy_mask = mask.cpu().numpy()[2]
        print(numpy_mask.shape)
        # Create a PIL Image, converting 0s to 0 and 1s to 255 (black and white)
        mask_image = Image.fromarray((numpy_mask * 255).astype('uint8'), mode="L")
        
        # Now you can save the individual mask image
        utils.saveImage(mask_image, folder = "masks", prefix = "mask")
    return masks, detection_labels

def isolate_objects_on_full_canvas(original_image, masks, labels):
    # Convert original image to RGBA to handle transparency during composition
    original_image_rgba = original_image.convert('RGBA')
    isolated_object_images = []

    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Create a binary mask image from the tensor
        binary_mask_pil = Image.fromarray(mask[2].cpu().numpy())

        # Create a new, blank white canvas with the same dimensions as the original
        white_background = Image.new("RGBA", original_image.size, (255, 255, 255, 255))

        # The 'Image.composite' function takes the original image and pastes it onto
        # the white background, using the binary mask as a stencil.
        # This preserves the object's original position and size.
        isolated_object_image = Image.composite(original_image_rgba, white_background, binary_mask_pil)
        
        # Save the result as an RGB image
        utils.saveImage(isolated_object_image.convert("RGB"), folder = "isolated_object", prefix = "iso_object")

        isolated_object_images.append(isolated_object_image)
        
    return isolated_object_images

def grouping_objects(object_images, labels, device = "cpu"):
    grouped_objects_image = Image.new("RGBA", object_images[0].size, (255, 255, 255, 255))
    based_combined_mask = Image.new("L", object_images[0].size)
    combined_mask_np = np.zeros(np.array(based_combined_mask).shape, dtype=np.uint8)
    object_masks = []
    for object_image, label in zip(object_images, labels):
        mask, n_label = grounded_sam_pipeline(object_image, f". {label}. ." ,device)
        individual_mask_np = mask[0][2].cpu().numpy()
        object_masks.append(mask[0][2])
        binary_mask_np = (individual_mask_np * 255).astype(np.uint8)
        binary_mask_pil = Image.fromarray(mask[0][2].cpu().numpy())
        grouped_objects_image = Image.composite(object_image, grouped_objects_image, binary_mask_pil)
        combined_mask_np = np.bitwise_or(combined_mask_np, binary_mask_np)
    utils.saveImage(Image.fromarray(combined_mask_np), folder="grouped", prefix="grouped_mask")
    utils.saveImage(grouped_objects_image.convert("RGB"), folder="grouped", prefix="grouped_image")
    return grouped_objects_image, combined_mask_np, object_masks

if __name__ == "__main__":
    image, masks,labels = grounded_sam_pipeline()
    isolate_objects_on_full_canvas(image, masks, labels)
