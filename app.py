import torch
import random
import numpy as np
from PIL import Image

import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

from pipeline import InstantCharacterFluxPipeline

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# pre-trained weights
ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
birefnet_path = 'ZhengPeng7/BiRefNet'
makoto_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai", filename="Makoto_Shinkai_style.safetensors")
ghibli_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Ghibli", filename="ghibli_style.safetensors")

# init InstantCharacter pipeline
pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to(device)

# load InstantCharacter
pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
)

# load matting model
birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
birefnet.to('cuda')
birefnet.eval()
birefnet_transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def remove_bkg(subject_image):

    def infer_matting(img_pil):
        input_images = birefnet_transform_image(img_pil).unsqueeze(0).to('cuda')

        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img_pil.size)
        mask = np.array(mask)
        mask = mask[..., None]
        return mask

    def get_bbox_from_mask(mask, th=128):
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1

        sample = np.max(mask, axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x1 = idx
                break
        
        sample = np.max(mask[:, ::-1], axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x2 = width - 1 - idx
                break

        sample = np.max(mask, axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y1 = idx
                break

        sample = np.max(mask[::-1], axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y2 = height - 1 - idx
                break

        x1 = np.clip(x1, 0, width-1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height-1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width-1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height-1).round().astype(np.int32)

        return [x1, y1, x2, y2]

    def pad_to_square(image, pad_value = 255, random = False):
        '''
            image: np.array [h, w, 3]
        '''
        H,W = image.shape[0], image.shape[1]
        if H == W:
            return image

        padd = abs(H - W)
        if random:
            padd_1 = int(np.random.randint(0,padd))
        else:
            padd_1 = int(padd / 2)
        padd_2 = padd - padd_1

        if H > W:
            pad_param = ((0,0),(padd_1,padd_2),(0,0))
        else:
            pad_param = ((padd_1,padd_2),(0,0),(0,0))

        image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
        return image

    salient_object_mask = infer_matting(subject_image)[..., 0]
    x1, y1, x2, y2 = get_bbox_from_mask(salient_object_mask)
    subject_image = np.array(subject_image)
    salient_object_mask[salient_object_mask > 128] = 255
    salient_object_mask[salient_object_mask < 128] = 0
    sample_mask = np.concatenate([salient_object_mask[..., None]]*3, axis=2)
    obj_image = sample_mask / 255 * subject_image + (1 - sample_mask / 255) * 255
    crop_obj_image = obj_image[y1:y2, x1:x2]
    crop_pad_obj_image = pad_to_square(crop_obj_image, 255)
    subject_image = Image.fromarray(crop_pad_obj_image.astype(np.uint8))
    return subject_image


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def get_example():
    case = [
        [
            "./assets/girl.jpg",
            "A girl is playing a guitar in street",
            0.9,
            'Makoto Shinkai style',
        ],
        [
            "./assets/boy.jpg",
            "A boy is riding a bike in snow",
            0.9,
            'Makoto Shinkai style',
        ],
    ]
    return case

def run_for_examples(source_image, prompt, scale, style_mode):

    return create_image(
        input_image=source_image,
        prompt=prompt,
        scale=scale,
        guidance_scale=3.5,
        num_inference_steps=28,
        seed=123456,
        style_mode=style_mode,
    )

def create_image(input_image,
                 prompt,
                 scale, 
                 guidance_scale,
                 num_inference_steps,
                 seed,
                 style_mode=None):
    
    input_image = remove_bkg(input_image)

    if style_mode is None:
        images = pipe(
            prompt=prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
            subject_image=input_image,
            subject_scale=scale,
            generator=torch.manual_seed(seed),
        ).images
    else:
        if style_mode == 'Makoto Shinkai style':
            lora_file_path = makoto_style_lora_path
            trigger = 'Makoto Shinkai style'
        elif style_mode == 'Ghibli style':
            lora_file_path = ghibli_style_lora_path
            trigger = 'ghibli style'

        images = pipe.with_style_lora(
            lora_file_path=lora_file_path,
            trigger=trigger,
            prompt=prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
            subject_image=input_image,
            subject_scale=scale,
            generator=torch.manual_seed(seed),
        ).images

    
    return images

# Description
title = r"""
<h1 align="center">InstantCharacter : Personalize Any Characters with a Scalable Diffusion Transformer Framework</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://instantcharacter.github.io/' target='_blank'><b>InstantCharacter : Personalize Any Characters with a Scalable Diffusion Transformer Framework</b></a>.<br>
How to use:<br>
1. Upload a character image, removing background would be preferred.
2. Enter a text prompt to describe what you hope the chracter does.
3. Click the <b>Submit</b> button to begin customization.
4. Share your customized photo with your friends and enjoy! 😊
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
TBD
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue.
"""

block = gr.Blocks(css="footer {visibility: hidden}").queue(max_size=10, api_open=False)
with block:
    
    # description
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                
                with gr.Row():
                    with gr.Column():
                        image_pil = gr.Image(label="Source Image", type='pil')
                
                prompt = gr.Textbox(label="Prompt", value="a character is riding a bike in snow")
                
                scale = gr.Slider(minimum=0, maximum=1.5, step=0.01,value=1.0, label="Scale")
                style_mode = gr.Dropdown(label='Style', choices=[None, 'Makoto Shinkai style', 'Ghibli style'], value='Makoto Shinkai style')
                
                with gr.Accordion(open=False, label="Advanced Options"):
                    guidance_scale = gr.Slider(minimum=1,maximum=7.0, step=0.01,value=3.5, label="guidance scale")
                    num_inference_steps = gr.Slider(minimum=5,maximum=50.0, step=1.0,value=28, label="num inference steps")
                    seed = gr.Slider(minimum=-1000000, maximum=1000000, value=123456, step=1, label="Seed Value")
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                generate_button = gr.Button("Generate Image")
                
            with gr.Column():
                generated_image = gr.Gallery(label="Generated Image")

        generate_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=create_image,
            inputs=[image_pil,
                    prompt,
                    scale, 
                    guidance_scale,
                    num_inference_steps,
                    seed,
                    style_mode,
                    ], 
            outputs=[generated_image])
    
    gr.Examples(
        examples=get_example(),
        inputs=[image_pil, prompt, scale, style_mode],
        fn=run_for_examples,
        outputs=[generated_image],
        cache_examples=True,
    )
    
    gr.Markdown(article)

block.launch(server_name="0.0.0.0", server_port=80)
