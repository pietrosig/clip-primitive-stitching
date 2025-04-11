from PIL import Image
import torch.nn.functional as F
import torch

def differentiable_rasterize(image_list, positions, layers, canvas_dim):
    """
    Differentiably composites a list of images onto a canvas using affine transformations.
    
    Args:
        image_list (list[Tensor]): List of image tensors, each of shape (C, H, W) or (H, W, C).
        positions (Tensor): Tensor of shape (N, 4) (or (N, 3) if square) containing 
                            normalized positions (x, y, h, w) with values in [0, 1]. 
                            (Ensure positions.requires_grad is True.)
        layers (list or Tensor): List of layer indices (lower layers are drawn first).
        canvas_dim (tuple): (canvas_height, canvas_width)
        
    Returns:
        Tensor: Composited canvas of shape (1, 4, canvas_height, canvas_width) with a grad_fn.
    """
    canvas_height, canvas_width = canvas_dim
    device = positions.device

    # Initialize canvas as zeros in float; use 4 channels (RGBA).
    canvas = torch.zeros((1, 4, canvas_height, canvas_width), device=device)

    # Sort the images, positions, and layers by layers (lowest first).
    sorted_indices = sorted(range(len(layers)), key=lambda i: layers[i])
    
    for i in sorted_indices:
        img = image_list[i]
        pos = positions[i]  # shape: (4,) or (3,)

        # Convert image to channel-first format if needed.
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img.permute(2, 0, 1)
        
        # If pos has 3 values, assume h == w.
        if pos.shape[0] == 3:
            pos = torch.cat([pos, pos[2:3]], dim=0)
        
        # Unpack normalized position and size: (x, y, h, w)
        # (x, y) is the top-left corner in normalized coordinates.
        px, py, ph, pw = pos

        # Convert the top-left corner from normalized [0,1] to [-1,1] (for canvas grid).
        left = 2 * px - 1  # differentiable
        top  = 2 * py - 1  # differentiable

        # The scaling factors to map the canvas region to image normalized coordinates.
        scale_x = 1.0 / pw
        scale_y = 1.0 / ph

        # Compute translations so that the canvas region starting at (left, top) maps to -1.
        trans_x = -1 - (left * scale_x)
        trans_y = -1 - (top * scale_y)

        # Instead of torch.tensor(), use torch.stack to keep gradients.
        row1 = torch.stack([scale_x, torch.zeros_like(scale_x), trans_x])
        row2 = torch.stack([torch.zeros_like(scale_y), scale_y, trans_y])
        theta = torch.stack([row1, row2], dim=0).unsqueeze(0)  # shape: (1, 2, 3)

        # Create a sampling grid for the canvas.
        grid = F.affine_grid(theta, size=(1, img.shape[0], canvas_height, canvas_width), align_corners=True)
        
        # Add batch dimension to the image.
        img_batch = img.unsqueeze(0)
        # Sample the image using differentiable bilinear interpolation.
        sampled_img = F.grid_sample(img_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # If image has 3 channels (RGB), add an alpha channel of ones.
        if sampled_img.shape[1] == 3:
            alpha = torch.ones_like(sampled_img[:, :1, :, :])
            sampled_img = torch.cat([sampled_img, alpha], dim=1)
        
        # Composite the sampled image onto the canvas using the "over" operation:
        # new_canvas = sampled_img + (1 - sampled_img_alpha) * canvas.
        src_alpha = sampled_img[:, 3:4, :, :]
        
        # Convert sampled image to premultiplied form.
        src_color = sampled_img[:, :3, :, :] * src_alpha
        dst_color = canvas[:, :3, :, :] * canvas[:, 3:4, :, :]
        dst_alpha = canvas[:, 3:4, :, :]

        # Standard "over" operation for premultiplied alpha.
        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
        out_color = src_color + dst_color * (1 - src_alpha)

        # Unpremultiply the color channels (avoid division by zero).
        canvas = torch.cat([out_color / (out_alpha + 1e-8), out_alpha], dim=1)


    return canvas



def rasterize_shapes(shapes, positions, output_size=(800, 800)):
    """
    Rasterizes a list of shapes onto a single image, sorted by layers.
    
    Parameters:
        shapes (list of str): File paths to the PNG RGBA images.
        positions (list of tuples): A list of tuples (x, y, s, r, l) where:
                                    - x, y are normalized positions (0 to 1),
                                    - s is the normalized new scale (0 to 1),
                                    - r is a normalized rotation (0 to 1 mapped to 0-360 degrees).
                                    - l is the layer 
        output_size (tuple): Size of the output image (width, height).
    
    Returns:
        Image: A PIL Image object with all the shapes rasterized.
    """
    # Sort shapes, positions, and layers based on layer order (ascending)
    sorted_data = sorted(zip(shapes, positions), key=lambda item: item[1][-1])
    
    # Create a blank canvas with a transparent background
    canvas = Image.new("RGBA", output_size, (0, 0, 0, 0))
    width, height = output_size
    for shape, pos in sorted_data:
        x, y, s, r, _ = pos
        # Load the shape image
        shape = shape.convert("RGBA")

        # Scale the shape
        new_size = (int(shape.width * s), int(shape.height * s))
        shape = shape.resize(new_size, resample=Image.LANCZOS)
        
        # Map r from [0,1] to [0,360] degrees
        rotation_angle = r * 360

        # Rotate the shape
        rotated_shape = shape.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)
        
        # Calculate the position on the canvas
        paste_x = int(x * width)
        paste_y = int(y * height)
        
        # Paste the rotated shape onto the canvas
        canvas.alpha_composite(rotated_shape, (paste_x, paste_y))
    
    return canvas


def rasterize_shapes_2(shapes, positions, output_size=(800, 800)):
    """
    Rasterizes a list of shapes onto a single image, sorted by layers.
    
    Parameters:
        shapes (list of Image): PIL Image objects in RGBA mode.
        positions (list of tuples): A list of tuples (x, y, s, r, l) where:
                                    - x, y are normalized positions (0 to 1) used as anchors,
                                    - s is the normalized scale (0 to 1),
                                    - r is a normalized rotation (0 to 1 mapped to 0-360 degrees),
                                    - l is the layer.
        output_size (tuple): Size of the output image (width, height).
    
    Returns:
        Image: A PIL Image object with all the shapes rasterized, with clipping to the canvas.
    """
    # Sort shapes and positions based on layer order (ascending)
    sorted_data = sorted(zip(shapes, positions), key=lambda item: item[1][-1])
    
    # Create a blank canvas with a transparent background
    canvas = Image.new("RGBA", output_size, (0, 0, 0, 0))
    canvas_width, canvas_height = output_size
    
    for shape, pos in sorted_data:
        x, y, s, r, _ = pos
        
        # Ensure the shape is in RGBA mode
        shape = shape.convert("RGBA")
        
        # Scale the shape
        new_size = (int(shape.width * s), int(shape.height * s))
        shape = shape.resize(new_size, resample=Image.LANCZOS)
        
        # Map r from [0,1] to [0,360] degrees and rotate the shape.
        rotation_angle = r * 360
        rotated_shape = shape.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)
        
        # Compute the anchor point on the canvas (in pixels)
        canvas_anchor_x = x * canvas_width
        canvas_anchor_y = y * canvas_height
        
        # Compute the corresponding anchor on the rotated shape
        shape_anchor_x = x * rotated_shape.width
        shape_anchor_y = y * rotated_shape.height
        
        # Determine where to paste the shape so that the shape's anchor aligns with the canvas anchor
        paste_x = int(canvas_anchor_x - shape_anchor_x)
        paste_y = int(canvas_anchor_y - shape_anchor_y)
        
        # Calculate the destination rectangle on the canvas
        dest_box = (paste_x, paste_y, paste_x + rotated_shape.width, paste_y + rotated_shape.height)
        
        # Calculate intersection of dest_box with the canvas bounds
        intersect_left = max(0, dest_box[0])
        intersect_top = max(0, dest_box[1])
        intersect_right = min(canvas_width, dest_box[2])
        intersect_bottom = min(canvas_height, dest_box[3])
        
        # Only composite if there is an intersection
        if intersect_left < intersect_right and intersect_top < intersect_bottom:
            # Determine the region of the rotated shape that lies within the canvas.
            crop_left = intersect_left - paste_x
            crop_top = intersect_top - paste_y
            crop_right = crop_left + (intersect_right - intersect_left)
            crop_bottom = crop_top + (intersect_bottom - intersect_top)
            
            # Crop the rotated shape to the intersecting region.
            cropped_shape = rotated_shape.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Composite the cropped shape onto the canvas at the intersecting position.
            canvas.alpha_composite(cropped_shape, (intersect_left, intersect_top))
    
    return canvas
