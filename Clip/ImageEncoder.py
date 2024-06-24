import torch
import numpy as np
import clip
from PIL import Image
#from IPython.display import display




from transformers import CLIPVisionModelWithProjection, CLIPProcessor

model_id = "openai/clip-vit-base-patch32"
model = CLIPVisionModelWithProjection.from_pretrained(model_id, return_dict=False)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

img = Image.open("./assets/andy_cat.jpeg")
example_input = processor(images=img, return_tensors="pt")
example_input = example_input['pixel_values']
traced_model = torch.jit.trace(model, example_input)

traced_model.save("traced_clip_vision_model.pt")





traced_model = torch.jit.load("traced_clip_vision_model.pt")

# Load and preprocess a new image
new_img = Image.open("./assets/pigeon.jpeg")
new_example_input = processor(images=new_img, return_tensors="pt")['pixel_values']

# Run inference with the traced model
with torch.no_grad():
    output = traced_model(new_example_input)

# Process the output as needed
print("Model output shape:", output)