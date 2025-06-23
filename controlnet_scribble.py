from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from PIL import Image
import utils
import os


#preparing const and download func for controlnet
CONTROLNET_LOCAL_PATH = "./cache/sd-controlnet-scribble-local"
CONTROLNET_REMOTE_PATH = "lllyasviel/sd-controlnet-scribble"
def get_controlnet_local_or_download():
    if os.path.isdir(CONTROLNET_LOCAL_PATH):
        print("Use local model")
        pipe = ControlNetModel.from_pretrained(CONTROLNET_LOCAL_PATH)
    else:
        print("Download model")
        pipe = ControlNetModel.from_pretrained(
            CONTROLNET_REMOTE_PATH, 
            safety_checker=None, 
            requires_safety_checker=False
        )
        pipe.save_pretrained(CONTROLNET_LOCAL_PATH)
    return pipe

#preparing const and download func for sd controlnet
SD_1_5_LOCAL_PATH = "./cache/stable-diffusion-v1-5-local"
SD_1_5_REMOTE_PATH = "runwayml/stable-diffusion-v1-5"
def get_sd_controlnet_local_or_download(controlnet):
    if os.path.isdir(SD_1_5_LOCAL_PATH):
        print("Use local model")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_1_5_LOCAL_PATH)
    else:
        print("Download model")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_1_5_REMOTE_PATH, 
            controlnet = controlnet, 
            safety_checker=None, 
            requires_safety_checker=False
        )
        pipe.save_pretrained(SD_1_5_LOCAL_PATH)
    return pipe

def gen_images_by_controlnet_scribble(isolated_sketch_objects, labels):
    image = Image.open("./bicycle.png")
    controlnet = get_controlnet_local_or_download()
    pipe =  get_sd_controlnet_local_or_download(controlnet)
    gen_images = []
    object_prompts = []
    for sketch_object, labels in zip(isolated_sketch_objects, labels):
        prompt= f"A photo of a small Real-life {labels} with position, size, shape fit reference object"
        image = pipe(prompt, sketch_object).images[0]
        utils.saveImage(image, folder = "objects_image", prefix = "gen_{labels}")
        gen_images.append(image)
        object_prompts.append(prompt)
    return gen_images, object_prompts
