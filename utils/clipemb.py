import torch
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T

clip_preprocess = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),  # Resize to 224x224
    T.CenterCrop(224),
    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

import clip


def CLIP_emb_from_tensor(tensor_img):
  model, _ = clip.load("ViT-B/32", device=DEVICE)

  # Load and preprocess image
  image_preprocessed = clip_preprocess(tensor_img).to(DEVICE)

  image_features = model.encode_image(image_preprocessed)
  image_features = image_features / image_features.norm(dim=1, keepdim=True)
  return image_features


def CLIP_emb_from_IMG(IMG):
  inputs = processor(images=IMG, return_tensors="pt", padding=True).to(DEVICE)

  with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

  return image_features

def CLIP_emb_from_TEXT(TEXT):
  inputs = processor(text=TEXT, return_tensors="pt", padding=True).to(DEVICE)

  with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
  return text_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai/clip-vit-base-patch32"

# Try with bigger model
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(DEVICE)
