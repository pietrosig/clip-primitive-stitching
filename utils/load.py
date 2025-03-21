import os
from PIL import Image

def pad_resize(img, dim):
    img.thumbnail(dim)
    return img

def spec_img(dir, name, dic, dim):
  # Assume name is of format name.png
  dic[name.split('.')[0]] = pad_resize(Image.open(os.path.join(dir, name)), dim)


def spec_path(dir, name, dic):
  # Assume name is of format name.png
  dic[name.split('.')[0]] = os.path.join(dir, name)

def add_synonyms(synonyms, dic):
  for s in synonyms:
    dic[s] = dic[synonyms[s]]

def get_segment_imgs(SEGMENT_DIR, DIM):
    segment_imgs = {}

    for filename in os.listdir(SEGMENT_DIR):
        full_path = os.path.join(SEGMENT_DIR, filename)
        # Check if filename is dir
        if os.path.isdir(full_path):
            # Create all primitives dict (composed of segments)
            segment_primitives = {}
            for segment_name in os.listdir(full_path):
                if segment_name.lower().endswith(".png"):
                    spec_img(full_path, segment_name, segment_primitives, DIM)

            # Add segments to all primitives dict
            segment_imgs[filename] = segment_primitives
    
    return segment_imgs

def get_primitive_imgs(PRIMITIVES_DIR, SYNONYMS = None):
    primitives_img_paths = {}
    for filename in os.listdir(PRIMITIVES_DIR):
        if filename.lower().endswith(".png"):
            spec_path(filename, primitives_img_paths)
    
    if SYNONYMS:
        add_synonyms(SYNONYMS, primitives_img_paths)

    return primitives_img_paths