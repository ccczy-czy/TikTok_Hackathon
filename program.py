import glob
import torch
import re
from PIL import Image
from transformers import CLIPProcessor

#text-input to test for now
text_input = "a sad looking cat"
#get the path/directory of where your are stored (currently our local)
folder_dir = 'assets' 
#dictionary where we will store our image embeddings
img_embeddings_dict = {}


#load text/img Processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#load our Image encoding model
traced_model = torch.jit.load("traced_clip_vision_model.pt")



"""
# Load and preprocess a new image
new_img = Image.open("./assets/pigeon.jpeg")
new_example_input = processor(images=new_img, return_tensors="pt")['pixel_values']

# Run inference with the traced model
with torch.no_grad():
    output = traced_model(new_example_input)

# Process the output as needed
print("Model output shape:", output)
"""

#iterate over images folder, run the ImageEncoding and add the embeddigns to dictionary
for image in glob.iglob(f'{folder_dir}/*'):
    if (image.endswith((".jpeg",".png",".JPG"))):  #Idk how images are recieved from s3...url? binary? Base64? But we'll probably need some conversion
        match = re.search(r"assets/([^/.]+)", image) #for now, we use regex to get name of img. Later on we'll probs assign a unique key/id to uploaded images
        new_img = Image.open(image)
        img_encodings = processor(images=new_img, return_tensors="pt")['pixel_values']
        img_embeddings_dict[match.group(1)] = img_encodings

print(img_embeddings_dict)


#create cosine simalarity function/part






#text encoding part














"""
notes:

Eventually, we want to only encode image each time an image is uploaded.
That way we don't do unnesecary encodings, and only run the cosine similarity on the dictionary of embeddings
Alsooooo...this is gonna be slow in python. We might need to batch process. Can use JobIt? not sure yet.


"""