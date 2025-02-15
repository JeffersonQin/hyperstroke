from transformers import AutoModel, AutoConfig
from transformers import PreTrainedModel, PretrainedConfig
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
import torch.nn as nn


class ControllableVisionEncoderDecoderConfig(VisionEncoderDecoderConfig):
    """
    Based on transformers v4.39.3 source code
    """

    def __init__(self, control=None, control_size=0, **kwargs):
        """
        (load): will be invoked during `from_pretrained`
        """
        super().__init__(**kwargs)
        if isinstance(control, dict):
            control_type = control.pop("model_type")
            control = AutoConfig.for_model(control_type, **control)
        self.control = control
        self.control_size = control_size


    @classmethod
    def from_pretrained_and_control(
        cls,
        pretrained_model_name_or_path: str = None,
        control_config = None,
        control_size = 0,
        *model_args,
        **kwargs,
    ) -> PretrainedConfig:
        """
        (create): method to create config from existing ones
        """
        config = VisionEncoderDecoderConfig.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        config.control = control_config
        config.control_size = control_size
        return config


class ControllableVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    """
    Based on transformers v4.39.3 source code
    """

    config_class = ControllableVisionEncoderDecoderConfig


    def __init__(self, config):
        """
        (load): will be invoked during `from_pretrained`
        """
        super().__init__(config)
        self.control = AutoModel.from_config(config.control)
        self.control_to_dec_proj = nn.Linear(config.control_size, self.decoder.config.hidden_size)


    @classmethod
    def from_pretrained_and_control(
        cls,
        pretrained_model_name_or_path: str = None,
        control_model = None,
        control_size = 0,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        (create): method to create config from existing ones
        NOTE: still have the class of `VisionEncoderDecoderModel` but with additional `control` and `control_to_dec_proj`
        NOTE: need to save then load to get the class right
        """
        ved_model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        ved_model.config = ControllableVisionEncoderDecoderConfig.from_pretrained_and_control(
            pretrained_model_name_or_path,
            control_config=control_model.config,
            control_size=control_size,
            *model_args,
            **kwargs,
        )
        ved_model.control = control_model
        ved_model.control_to_dec_proj = nn.Linear(control_size, ved_model.decoder.config.hidden_size)

        return ved_model
