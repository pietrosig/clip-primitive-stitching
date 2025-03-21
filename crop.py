from PIL import Image
import os

def crop_image(input_path, output_path):
    # Open the image
    img = Image.open(input_path).convert("RGBA")
    pixels = img.load()
    
    # Get image dimensions
    width, height = img.size

    # Find bounding box of non-transparent pixels
    left, right, top, bottom = width, 0, height, 0

    for y in range(height):
        for x in range(width):
            _, _, _, alpha = pixels[x, y]
            if alpha > 0:  # Non-transparent pixel
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

    # Ensure valid cropping dimensions
    if left < right and top < bottom:
        img = img.crop((left, top, right + 1, bottom + 1))
        img.save(output_path)
        print(f"Cropped image saved to {output_path}")
    else:
        print("No non-transparent pixels found.")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            crop_image(input_path, output_path)

generalPath = '/Users/pietrosig/Desktop/DL'
imgPath = os.path.join(generalPath, 'DONE')
imgCrop = os.path.join(generalPath, 'CROP')

process_directory(imgPath, imgCrop)