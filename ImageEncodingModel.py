import torch
import numpy as np
from PIL import Image

"""
This is first test model. We use clip model ViT-B/32.  We use HUGGINGFACE transformers library here to integrate CLIP rather than CLIP directly
"""

from transformers import CLIPVisionModelWithProjection, CLIPProcessor

model_id = "openai/clip-vit-base-patch32" #huggingface model id for CLIP model
model = CLIPVisionModelWithProjection.from_pretrained(model_id, return_dict=False)  #load the pretrained model
processor = CLIPProcessor.from_pretrained(model_id) #load the processor which will process the image and/or text
model.eval() #tbh i'm not sure what this does...i'll keep it here for now

img = Image.open("./assets/pigeon.jpeg") #use PIL to load example image
example_input = processor(images=img, return_tensors="pt") #process image, which will return a dictionary containing the processed image tensor under the key 'pixel_values'.
example_input = example_input['pixel_values']  #get the tensor


traced_model = torch.jit.trace(model, example_input) 

traced_model.save("traced_clip_vision_model.pt")

