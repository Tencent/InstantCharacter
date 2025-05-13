# !pip install transformers accelerate diffusers huggingface_cli
import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline

# Step 1 Load base model and adapter

ip_adapter_path = 'checkpoints/instantcharacter_ip-adapter.bin'
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
seed = 123456

pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

# Step 1.1, To manually configure the CPU offload mode.
# You may selectively designate which layers to employ the offload hook based on the available VRAM capacity of your GPU.
# The following configuration can reach about 22GB of VRAM usage on NVIDIA L20 (Ada arch)

pipe.to("cpu")
pipe._exclude_from_cpu_offload.extend([
    # 'vae',
    'text_encoder',
    # 'text_encoder_2',
])
pipe._exclude_layer_from_cpu_offload.extend([
    "transformer.pos_embed",
    "transformer.time_text_embed",
    "transformer.context_embedder",
    "transformer.x_embedder",
    "transformer.transformer_blocks",
    # "transformer.single_transformer_blocks",
    "transformer.norm_out",
    "transformer.proj_out",
])
pipe.enable_sequential_cpu_offload()

pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
    device=torch.device('cuda')
)

# Step 1.2 Optional inference acceleration
# You can set the TORCHINDUCTOR_CACHE_DIR in production environment.

torch._dynamo.reset()
torch._dynamo.config.cache_size_limit = 1024
torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

for layer in pipe.transformer.attn_processors.values():
    layer = torch.compile(
        layer,
        fullgraph=True,
        dynamic=True,
        mode="max-autotune",
        backend='inductor'
    )
pipe.transformer.single_transformer_blocks.compile(
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)
pipe.transformer.transformer_blocks.compile(
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)
pipe.vae = torch.compile(
    pipe.vae,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)
pipe.text_encoder = torch.compile(
    pipe.text_encoder,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)


# Step 2 Load reference image
ref_image_path = 'assets/girl.jpg'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')

# Step 3 Inference without style
prompt = "A girl is playing a guitar in street"

# warm up for torch.compile
image = pipe(
        prompt=prompt, 
        num_inference_steps=28,
        guidance_scale=3.5,
        subject_image=ref_image,
        subject_scale=0.9,
        generator=torch.manual_seed(seed),
    ).images[0]

image = pipe(
        prompt=prompt, 
        num_inference_steps=28,
        guidance_scale=3.5,
        subject_image=ref_image,
        subject_scale=0.9,
        generator=torch.manual_seed(seed),
    ).images[0]

image.save("flux_instantcharacter.png")
