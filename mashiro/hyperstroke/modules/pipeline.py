from typing import Dict, List, Optional, Tuple, Union, Any

from .vqgan.model import VQModel
from .transformer import CLIPControlVisionEncoderDecoderModel
from .config import HyperstrokeConfig

import numpy as np
import torch

import os
from huggingface_hub.utils import validate_hf_hub_args

from dataclasses import dataclass

from PIL import Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput


##################################################################################
# tested under diffusers v0.32.2
# fix `from_pretrained(<hub>)` issue by bypassing custom components check
# i.e. `py` files no need to exist in the remote repository
import diffusers

def _get_custom_components_and_folders(
    pretrained_model_name: str,
    config_dict: Dict[str, Any],
    filenames: Optional[List[str]] = None,
    variant_filenames: Optional[List[str]] = None,
    variant: Optional[str] = None,
):
    config_dict = config_dict.copy()

    # retrieve all folder_names that contain relevant files
    folder_names = [k for k, v in config_dict.items() if isinstance(v, list) and k != "_class_name"]
    custom_components = {}
    return custom_components, folder_names

diffusers.pipelines.pipeline_utils._get_custom_components_and_folders = _get_custom_components_and_folders
##################################################################################

@dataclass
class HyperstrokePipelineOutput(BaseOutput):
    images: List[List[Image.Image]]
    coordinates: torch.Tensor
    tokens: torch.Tensor


class HyperstrokePipeline(DiffusionPipeline):
    def __init__(
        self, 
        vqgan_model: VQModel, 
        transformer: CLIPControlVisionEncoderDecoderModel,
        hyperstroke_config: HyperstrokeConfig
    ):
        super().__init__()
        self.register_modules(vqgan_model=vqgan_model, transformer=transformer, hyperstroke_config=hyperstroke_config)


    # workaround for upstream: https://github.com/huggingface/diffusers/pull/10779
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        if not os.path.isdir(pretrained_model_name_or_path):
            from huggingface_hub import snapshot_download
            cache_dir = kwargs.get("cache_dir", None)
            proxies = kwargs.get("proxies", None)
            local_files_only = kwargs.get("local_files_only", None)
            token = kwargs.get("token", None)
            revision = kwargs.get("revision", None)

            _ = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=["transformer/generation_config.json"],
                user_agent={"pipeline_class": cls.__name__},
            )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


    @torch.no_grad()
    def __call__(
        self,
        canvas_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        control_image: Optional[Union[torch.Tensor, Image.Image, List[Image.Image]]]=None,
        control_text: Optional[List[str]]=None,
        **kwargs
    ):
        device = self._execution_device
        canvas_processor = self.hyperstroke_config.canvas_processor
        control_processor = self.hyperstroke_config.control_processor

        num_codebook_token = self.hyperstroke_config.num_codebook_token
        num_bbox_token = self.hyperstroke_config.num_bbox_token


        # image -> pixel_values, and obtain h/w
        if not isinstance(canvas_image, torch.Tensor):
            if not isinstance(canvas_image, list):
                canvas_image = [canvas_image]

            image_width = torch.zeros((len(canvas_image),), device=device)
            image_height = torch.zeros((len(canvas_image),), device=device)

            for i, img in enumerate(canvas_image):
                w, h = img.size
                image_width[i] = w
                image_height[i] = h

            canvas_image = canvas_processor(canvas_image, return_tensors="pt")["pixel_values"].to(device)
        else:
            image_width = canvas_image.shape[-1]
            image_height = canvas_image.shape[-2]


        # control
        if control_image is not None:
            if not isinstance(control_image, torch.Tensor):
                if not isinstance(control_image, list):
                    control_image = [control_image]
                control_image = control_processor(control_image, return_tensors="pt")["pixel_values"].to(device)
            kwargs["control_pixel_values"] = control_image
        elif control_text is not None:
            if not isinstance(control_text, list):
                control_text = [control_text]
            control_text = control_processor(control_text, return_tensors="pt", padding=False)["input_ids"].to(device)
            kwargs["control_input_ids"] = control_text


        # inference
        raw_tokens = self.transformer.generate(canvas_image, **kwargs)
        # discard BOS token
        tokens = raw_tokens[:, 1:]


        # decode
        ret_coordinates = []
        ret_images = [[] for _ in range(tokens.shape[0])]

        for i in range(tokens.shape[1] // (num_bbox_token + num_codebook_token)):
            ## bbox
            bbox_tokens = tokens[:, i*(num_bbox_token + num_codebook_token):
                                    i*(num_bbox_token + num_codebook_token)+num_bbox_token]
            
            x1, y1, x2, y2 = bbox_tokens[:, 0], bbox_tokens[:, 1], bbox_tokens[:, 2], bbox_tokens[:, 3]
            
            real_x1 = (x1 * image_width / self.hyperstroke_config.bbox_vocab_size).to(torch.int32)
            real_y1 = (y1 * image_height / self.hyperstroke_config.bbox_vocab_size).to(torch.int32)
            real_x2 = (x2 * image_width / self.hyperstroke_config.bbox_vocab_size).to(torch.int32)
            real_y2 = (y2 * image_height / self.hyperstroke_config.bbox_vocab_size).to(torch.int32)

            # check if bbox tokens are valid
            if ((real_x1 >= real_x2) | (real_y1 >= real_y2) |
               (x1 > self.hyperstroke_config.bbox_vocab_size) | (y1 > self.hyperstroke_config.bbox_vocab_size) |
               (x2 > self.hyperstroke_config.bbox_vocab_size) | (y2 > self.hyperstroke_config.bbox_vocab_size)).any().item():
                msg = f"Invalid bbox token: {bbox_tokens} at generated stroke {i}, skipping"
                raise Exception(msg)

            real_coordinates = torch.stack([real_x1, real_y1, real_x2, real_y2], dim=-1)[:, None, :]
            ret_coordinates.append(real_coordinates)

            ## images
            codebook_tokens = tokens[:, i*(num_bbox_token + num_codebook_token)+num_bbox_token:
                                        (i+1)*(num_bbox_token + num_codebook_token)]

            # offset the bbox tokens
            t = codebook_tokens - (self.hyperstroke_config.bbox_vocab_size + 1)
            
            # inference
            d = self.vqgan_model.decode_code(t)

            # post process
            d = torch.clamp(d, -1., 1.).cpu()
            d = (((d + 1.0) / 2.0).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

            for j, m in enumerate(d):
                ret_images[j].append(Image.fromarray(m).convert("RGBA"))

        # batch, num_stroke, 4
        ret_coordinates = torch.cat(ret_coordinates, dim=1)

        return HyperstrokePipelineOutput(images=ret_images, coordinates=ret_coordinates, tokens=raw_tokens)
