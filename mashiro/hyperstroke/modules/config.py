from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from transformers import AutoProcessor


class HyperstrokeConfig(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        pretrained_canvas_processor: str,
        pretrained_control_processor: str,
        bbox_vocab_size: int,
        vqgan_input_width: int,
        vqgan_input_height: int,
        num_codebook_token: int,
        num_bbox_token: int=4,
    ):
        super(HyperstrokeConfig, self).__init__()
        self.canvas_processor = AutoProcessor.from_pretrained(pretrained_canvas_processor)
        self.control_processor = AutoProcessor.from_pretrained(pretrained_control_processor)
        self.bbox_vocab_size = bbox_vocab_size
        self.vqgan_input_width = vqgan_input_width
        self.vqgan_input_height = vqgan_input_height
        self.num_bbox_token = num_bbox_token
        self.num_codebook_token = num_codebook_token
