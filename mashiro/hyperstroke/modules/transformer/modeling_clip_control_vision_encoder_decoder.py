from . import ControllableVisionEncoderDecoderModel

from transformers.generation.utils import GenerateOutput

import torch
from typing import Optional, Tuple, Union, List, Callable
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right

from transformers.modeling_utils import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList


class CLIPControlVisionEncoderDecoderModel(ControllableVisionEncoderDecoderModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # NOTE: when `low_cpu_mem_usage` set to True the model will output random thing. don't know why
        kwargs["low_cpu_mem_usage"] = False
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


    """
    Based on transformers v4.39.3 source code
    + balance loss also supported seemlessly
    """
    @classmethod
    def from_pretrained_and_clip(
        cls,
        pretrained_model_name_or_path: str = None,
        clip_model = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        (create): method to create model from existing ones
        NOTE: still have the class of `VisionEncoderDecoderModel` but with additional `control` and `control_to_dec_proj`
        NOTE: need to save then load to get the class right
        """
        return ControllableVisionEncoderDecoderModel.from_pretrained_and_control(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            control_model=clip_model,
            control_size=clip_model.config.projection_dim,
            *model_args,
            **kwargs,
        )


    def _forward_control(
        self,
        control_pixel_values: Optional[torch.FloatTensor] = None,
        control_input_ids: Optional[torch.LongTensor] = None,
        control_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if control_pixel_values is None and control_input_ids is None:
            raise ValueError("You have to specify control_pixel_values or control_input_ids")

        if control_pixel_values is not None:
            control_outputs = self.control.vision_model(pixel_values=control_pixel_values)[1]
            control_outputs = self.control.visual_projection(control_outputs)
        else:
            control_outputs = self.control.text_model(input_ids=control_input_ids, attention_mask=control_attention_mask)[1]
            control_outputs = self.control.text_projection(control_outputs)

        # normalize (see CLIPModel implementation)
        control_outputs = control_outputs / control_outputs.norm(p=2, dim=-1, keepdim=True)

        return control_outputs


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        # ----------------- Modification Start ----------------- #
        control_pixel_values: Optional[torch.FloatTensor] = None,
        control_input_ids: Optional[torch.LongTensor] = None,
        control_attention_mask: Optional[torch.LongTensor] = None,
        control_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        # ----------------- Modification End ----------------- #
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, VisionEncoderDecoderModel
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        >>> # load image from the IAM dataset
        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> # training
        >>> model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> pixel_values = processor(image, return_tensors="pt").pixel_values
        >>> text = "hello world"
        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(pixel_values=pixel_values, labels=labels)
        >>> loss = outputs.loss

        >>> # inference (generation)
        >>> generated_ids = model.generate(pixel_values)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        """ ----------------- Modification Start ----------------- """
        # retrieve control_outputs or compute them
        if control_outputs is None:
            control_outputs = self._forward_control(control_pixel_values, control_input_ids, control_attention_mask)
        else:
            pass
            # NOTE: nothing todo here, but need to add this case in `generate()`
            # UPDATE: Done

        # project control_outputs to decoder hidden size
        control_outputs = self.control_to_dec_proj(control_outputs)

        # now only two dimensions, batch and hidden size, need to add sequence length (1)
        if control_outputs.dim() == 2:
            control_outputs = control_outputs.unsqueeze(1)
        # concat encoder and output hidden states
        encoder_hidden_states = torch.cat([encoder_hidden_states, control_outputs], dim=1)
        """ ----------------- Modification End ----------------- """

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            """ ----------------- Modification Start ----------------- """
            if hasattr(self, "mask_tensor"):
                #### this part is for the balance loss
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
                # balance mask
                lambda_mask = self.mask_tensor.to(loss.device)
                lambda_mask = lambda_mask.repeat(loss.size(0) // lambda_mask.size(0))
                loss = loss * lambda_mask
                # mean, ignoring (-100)
                loss = loss.sum() / (labels != -100).sum()
            else:
                #### original method
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            """ ----------------- Modification End ----------------- """

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def prepare_inputs_for_generation(self, input_ids, control_outputs=None, **kwargs):
        ret_dict = super().prepare_inputs_for_generation(input_ids, **kwargs)
        ret_dict["control_outputs"] = control_outputs
        return ret_dict


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        control_outputs = self._forward_control(**kwargs)
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            control_outputs=control_outputs,
            **kwargs
        )
