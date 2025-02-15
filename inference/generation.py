import numpy as np
import torch
import cv2
import torchvision
torch.set_grad_enabled(False)

from typing import List
from PIL import Image

from mashiro.hyperstroke.modules.pipeline import HyperstrokePipeline


def make_grid(images: List[Image.Image], nrow: int=6) -> Image.Image:
    # use torchvision.utils.make_grid
    # first convert to torch, then use make_grid, then convert back to PIL
    tensors = [torch.tensor(np.array(img)).permute(2,0,1) for img in images]
    grid = torchvision.utils.make_grid(tensors, nrow=nrow)
    grid = grid.permute(1,2,0)
    grid = grid.numpy()
    return Image.fromarray(grid)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = HyperstrokePipeline.from_pretrained("gyrojeff/Hyperstroke-Quickdraw").to(device)


    # blank canvas
    canvas_image = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 255)

    # control text
    control_text = "cat"

    output = pipeline(
        canvas_image,
        control_text=control_text,
        do_sample=True,
        temperature=0.7,
    )

    ## Final result rendering
    strokes = output.images[0]
    coordinates = output.coordinates[0]
    blends = []

    blend = canvas_image.copy()
    for stroke, (x1, y1, x2, y2) in zip(strokes, coordinates):
        # tensor to scalar
        x1 = x1.item()
        y1 = y1.item()
        x2 = x2.item()
        y2 = y2.item()
        # blend
        blend = blend.convert("RGBA")
        blend.alpha_composite(stroke.resize((x2 - x1, y2 - y1)), (x1, y1))
        blend = blend.convert("RGB")
        # bbox
        bboxed = cv2.rectangle(np.array(blend), (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cache
        blends.append(Image.fromarray(bboxed))


    make_grid(blends).save("blend.png")
    make_grid(strokes).save("strokes.png")
