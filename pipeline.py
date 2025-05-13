# Copyright 2025 Tencent InstantX Team. All rights reserved.
#

from PIL import Image
from einops import rearrange
import torch
from diffusers.pipelines.pipeline_utils import is_accelerate_available, is_accelerate_version
from diffusers.pipelines.flux.pipeline_flux import *
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor

from models.attn_processor import FluxIPAttnProcessor
from models.resampler import CrossLayerCrossScaleProjector
from models.utils import flux_load_lora


# TODO
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class InstantCharacterFluxPipeline(FluxPipeline):
    _exclude_layer_from_cpu_offload = []

    @torch.inference_mode()
    def encode_siglip_image_emb(self, siglip_image, device, dtype):
        siglip_image = siglip_image.to(device, dtype=dtype)
        res = self.siglip_image_encoder(siglip_image, output_hidden_states=True)

        siglip_image_embeds = res.last_hidden_state

        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        
        return siglip_image_embeds, siglip_image_shallow_embeds


    @torch.inference_mode()
    def encode_dinov2_image_emb(self, dinov2_image, device, dtype):
        dinov2_image = dinov2_image.to(device, dtype=dtype)
        res = self.dino_image_encoder_2(dinov2_image, output_hidden_states=True)

        dinov2_image_embeds = res.last_hidden_state[:, 1:]

        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)

        return dinov2_image_embeds, dinov2_image_shallow_embeds


    @torch.inference_mode()
    def encode_image_emb(self, siglip_image, device, dtype):
        object_image_pil = siglip_image
        object_image_pil_low_res = [object_image_pil.resize((384, 384))]
        object_image_pil_high_res = object_image_pil.resize((768, 768))
        object_image_pil_high_res = [
            object_image_pil_high_res.crop((0, 0, 384, 384)),
            object_image_pil_high_res.crop((384, 0, 768, 384)),
            object_image_pil_high_res.crop((0, 384, 384, 768)),
            object_image_pil_high_res.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res)

        siglip_image_embeds = self.encode_siglip_image_emb(
            self.siglip_image_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )
        dinov2_image_embeds = self.encode_dinov2_image_emb(
            self.dino_image_processor_2(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )

        image_embeds_low_res_deep = torch.cat([siglip_image_embeds[0], dinov2_image_embeds[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_image_embeds[1], dinov2_image_embeds[1]], dim=2)

        siglip_image_high_res = self.siglip_image_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        siglip_image_high_res = siglip_image_high_res[None]
        siglip_image_high_res = rearrange(siglip_image_high_res, 'b n c h w -> (b n) c h w')
        siglip_image_high_res_embeds = self.encode_siglip_image_emb(siglip_image_high_res, device, dtype)
        siglip_image_high_res_deep = rearrange(siglip_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        dinov2_image_high_res = self.dino_image_processor_2(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        dinov2_image_high_res = dinov2_image_high_res[None]
        dinov2_image_high_res = rearrange(dinov2_image_high_res, 'b n c h w -> (b n) c h w')
        dinov2_image_high_res_embeds = self.encode_dinov2_image_emb(dinov2_image_high_res, device, dtype)
        dinov2_image_high_res_deep = rearrange(dinov2_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

        image_embeds_dict = dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow,
            image_embeds_low_res_deep=image_embeds_low_res_deep,
            image_embeds_high_res_deep=image_embeds_high_res_deep,
        )
        return image_embeds_dict


    @torch.inference_mode()
    def init_ccp_and_attn_processor(self, *args, **kwargs):
        subject_ip_adapter_path = kwargs['subject_ip_adapter_path']
        nb_token = kwargs['nb_token']
        device=kwargs.get('nb_token', torch.device('cuda'))
        
        state_dict = torch.load(subject_ip_adapter_path, map_location="cpu")
        dtype = self.transformer.dtype


        print(f"=> init attn processor")
        attn_procs = {}
        for idx_attn, (name, v) in enumerate(self.transformer.attn_processors.items()):
            layer = FluxIPAttnProcessor(
                hidden_size=self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads,
                ip_hidden_states_dim=self.text_encoder_2.config.d_model,
            ).to(device, dtype=dtype)
            attn_procs[name] = layer
        self.transformer.set_attn_processor(attn_procs)
        tmp_ip_layers = torch.nn.ModuleList(self.transformer.attn_processors.values())
        key_name = tmp_ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
        print(f"=> load attn processor: {key_name}")

        print(f"=> init project")
        image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536,
            num_attention_heads=42,
            attention_head_dim=64,
            cross_attention_dim=1152 + 1536,
            num_layers=4,
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=nb_token,
            embedding_dim=1152 + 1536,
            output_dim=4096,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        image_proj_model.eval()
        image_proj_model.to(device, dtype=dtype)

        key_name = image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        print(f"=> load project: {key_name}")
        self.subject_image_proj_model = image_proj_model


    @torch.inference_mode()
    def init_adapter(
        self, 
        image_encoder_path=None, 
        image_encoder_2_path=None, 
        subject_ipadapter_cfg=None,
        device=torch.device('cuda')
    ):
        dtype = self.transformer.dtype

        # image encoder
        print(f"=> loading image_encoder_1: {image_encoder_path}")
        image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
        image_processor = SiglipImageProcessor.from_pretrained(image_encoder_path)
        image_encoder.eval()
        image_encoder.to(device, dtype=dtype)
        self.siglip_image_encoder = image_encoder
        self.siglip_image_processor = image_processor

        # image encoder 2
        print(f"=> loading image_encoder_2: {image_encoder_2_path}")
        image_encoder_2 = AutoModel.from_pretrained(image_encoder_2_path)
        image_processor_2 = AutoImageProcessor.from_pretrained(image_encoder_2_path)
        image_encoder_2.eval()
        image_encoder_2.to(device, dtype=dtype)
        image_processor_2.crop_size = dict(height=384, width=384)
        image_processor_2.size = dict(shortest_edge=384)
        self.dino_image_encoder_2 = image_encoder_2
        self.dino_image_processor_2 = image_processor_2

        # ccp and adapter
        self.init_ccp_and_attn_processor(**subject_ipadapter_cfg, device=device)


    @torch.inference_mode()
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        subject_image: Image.Image = None,
        subject_scale: float = 0.8,

    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 3.1 Prepare subject emb
        if subject_image is not None:
            subject_image = subject_image.resize((max(subject_image.size), max(subject_image.size)))
            subject_image_embeds_dict = self.encode_image_emb(subject_image, device, dtype)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)


                # subject adapter
                if subject_image is not None:
                    subject_image_prompt_embeds = self.subject_image_proj_model(
                        low_res_shallow=subject_image_embeds_dict['image_embeds_low_res_shallow'],
                        low_res_deep=subject_image_embeds_dict['image_embeds_low_res_deep'],
                        high_res_deep=subject_image_embeds_dict['image_embeds_high_res_deep'],
                        timesteps=timestep.to(dtype=latents.dtype), 
                        need_temb=True
                    )[0]
                    self._joint_attention_kwargs['emb_dict'] = dict(
                        length_encoder_hidden_states=prompt_embeds.shape[1]
                    )
                    self._joint_attention_kwargs['subject_emb_dict'] = dict(
                        ip_hidden_states=subject_image_prompt_embeds,
                        scale=subject_scale,
                    )
    
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


    def with_style_lora(self, lora_file_path, lora_weight=1.0, trigger='', *args, **kwargs):
        flux_load_lora(self, lora_file_path, lora_weight)
        kwargs['prompt'] = f"{trigger}, {kwargs['prompt']}"
        res = self.__call__(*args, **kwargs)
        flux_load_lora(self, lora_file_path, -lora_weight)
        return res

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using 🤗 Accelerate, significantly reducing memory usage. When called, the state
        dicts of all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are saved to CPU
        and then moved to `torch.device('meta')` and loaded to GPU only when their specific submodule has its `forward`
        method called. Offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        self._maybe_raise_error_if_group_offload_active(raise_error=True)

        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")
        self.remove_all_hooks()

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_sequential_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_sequential_cpu_offload()`."
            )

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        exclude_models = [module_name.split('.')[0] for module_name in self._exclude_layer_from_cpu_offload]

        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue
            
            if name in exclude_models:
                for layer_name, layer in model.named_children():
                    if '.'.join([name, layer_name]) in self._exclude_layer_from_cpu_offload:
                        layer.to(device)
                    else:
                        offload_buffers = len(layer._parameters) > 0
                        cpu_offload(layer, device, offload_buffers=offload_buffers)
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                # make sure to offload buffers if not all high level weights
                # are of type nn.Module
                offload_buffers = len(model._parameters) > 0
                cpu_offload(model, device, offload_buffers=offload_buffers)