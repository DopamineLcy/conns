import numpy as np
import torch
import torch.nn.functional as F


def get_interpolated_map(similarity_score, image_size, processed_size=1024, fill_value=0.0):
    height, width = image_size
    n_patches = similarity_score.shape[0]
    grid_size = int(np.sqrt(n_patches))
    similarity_score = similarity_score.view(grid_size, grid_size)
    similarity_score = F.interpolate(
        similarity_score.unsqueeze(0).unsqueeze(0),
        size=(processed_size, processed_size),
        mode="bilinear",
        align_corners=False,
    )

    shortest = min(height, width)
    scale = processed_size / shortest
    new_h = int(height * scale)
    new_w = int(width * scale)
    full_map = torch.full((new_h, new_w), fill_value, dtype=similarity_score.dtype)
    crop_y = (new_h - processed_size) // 2
    crop_x = (new_w - processed_size) // 2
    full_map[crop_y : crop_y + processed_size, crop_x : crop_x + processed_size] = similarity_score.squeeze()

    return F.interpolate(
        full_map.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()


def max_point(interpolated_map):
    _, indices = interpolated_map.view(-1).max(dim=0)
    y, x = torch.unravel_index(indices, interpolated_map.shape)
    return x.item(), y.item()


def point_in_boxes(point, boxes):
    x, y = point
    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

