import glob
import torch
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizerFast

#text-input to test for now
text_input = "a fashion dog that looks super rich"
#get the path/directory of where your are stored (currently our local)
folder_dir = 'assets' 
#dictionary where we will store our image embeddings
img_embeddings_dict = {}
#dictionary where we store simalirty rakings for now
sim_rankings = {}


#load text/img Processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#load our Image encoding model
traced_model = torch.jit.load("traced_clip_vision_model.pt")
#load Tect encoding model
loaded_traced_model = torch.jit.load("traced_clip_text_model.pt")


#iterate over images folder, run the ImageEncoding and add the embeddigns to dictionary
for image in glob.iglob(f'{folder_dir}/*'):
    if (image.endswith((".jpeg",".png",".JPG"))):  #Idk how images are recieved from s3...url? binary? Base64? But we'll probably need some conversion
        match = re.search(r"assets/([^/.]+)", image) #for now, we use regex to get name of img. Later on we'll probs assign a unique key/id to uploaded images
        new_img = Image.open(image)
        img_processed = processor(images=new_img, return_tensors="pt")['pixel_values']
        with torch.no_grad():
            output = traced_model(img_processed)
        img_embeddings_dict[match.group(1)] = output[0]
   

#text encoding part
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
textprocessed = tokenizer(text_input, return_tensors="pt")['input_ids']
with torch.no_grad():
    output = loaded_traced_model(textprocessed)
text_embeddings = output[0] / output[0].norm(dim=-1, keepdim=True)



#calculate cos sim for each
for img_name, img_embedding in img_embeddings_dict.items():
    img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
    similarity = torch.nn.functional.cosine_similarity(text_embeddings, img_embedding)
    print(f"Similarity between text and {img_name}: {similarity.item()}")









"""
notes:

Eventually, we want to only encode image each time an image is uploaded.
That way we don't do unnesecary encodings, and only run the cosine similarity on the dictionary of embeddings
Alsooooo...this is gonna be slow in python. We might need to batch process. Can use JobIt? not sure yet.


"""