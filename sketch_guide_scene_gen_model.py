import finetuning_stable
import controlnet_scribble
import grounded_sam
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image

device =  "cuda" if torch.cuda.is_available() else "cpu"
image_path = "./sketches/bread_coffee.png" 
image_size = 512
object_prompt = ". bread . cup of tea . ."
global_prompt = "A photo of a bread and a cup of tea in a tray."
bg_prompt = "in a tray"

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

def sketches_guide_scene_gen(
        sketch_path, 
        global_prompt, 
        object_prompt, 
        bg_prompt,
        num_epochs= 2,
        lr = 1e-5,
        alpha = 0.5,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        seed=42,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
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

if __name__ == "__main__":
    sketches_guide_scene_gen(image_path, global_prompt, object_prompt, bg_prompt)
