import torch
from transformers import CLIPProcessor, CLIPModel

def CLIP_emb_from_IMG(IMG):
  inputs = processor(images=IMG, return_tensors="pt", padding=True).to(DEVICE)

  with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

  return image_features

def CLIP_emb_from_TEXT(TEXT):
  print(DEVICE)
  inputs = processor(text=TEXT, return_tensors="pt", padding=True).to(DEVICE)

  with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
  return text_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(DEVICE)
