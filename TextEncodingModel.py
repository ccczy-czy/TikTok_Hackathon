import torch
import numpy as np
import clip
from PIL import Image


from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

#Text Encoder using clip
model_id = "openai/clip-vit-base-patch32"
model = CLIPTextModelWithProjection.from_pretrained(model_id, return_dict=False)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
model.eval()

example_input = tokenizer("a photo of a cat", return_tensors="pt")
example_input = example_input.data['input_ids']

traced_model = torch.jit.trace(model, example_input)

#save the model so we can use it in other files
traced_model.save("traced_clip_text_model.pt")

