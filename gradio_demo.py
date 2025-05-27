import torch
import gradio as gr
import spaces
import random
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

# Define model options
MODEL_OPTIONS = {
    "Wan2.1-T2V-1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan2.1-T2V-14B": "Wan-AI/Wan2.1-T2V-14B-Diffusers"
}

# Define scheduler options
SCHEDULER_OPTIONS = {
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler
}

@spaces.GPU(duration=300)
def generate_video(
    model_choice,
    prompt,
    negative_prompt,
    lora_id,
    lora_weight_name,
    lora_scale,
    scheduler_type,
    flow_shift,
    height,
    width,
    num_frames,
    guidance_scale,
    num_inference_steps,
    output_fps,
    seed
):
    # Get model ID from selection
    model_id = MODEL_OPTIONS[model_choice]
    
    # Set seed for reproducibility
    if seed == -1 or seed is None or seed == "":
        seed = random.randint(0, 2147483647)
    else:
        seed = int(seed)
    
    # Set the seed
    torch.manual_seed(seed)
    
    # Load model
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    
    # Set scheduler
    if scheduler_type == "UniPCMultistepScheduler":
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=flow_shift
        )
    else:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    
    # Move to GPU
    pipe.to("cuda")
    
    # Load LoRA weights if provided
    if lora_id and lora_id.strip():
        try:
            # If a specific weight name is provided, use it
            if lora_weight_name and lora_weight_name.strip():
                pipe.load_lora_weights(lora_id, weight_name=lora_weight_name)
            else:
                pipe.load_lora_weights(lora_id)
            
            # Set lora scale if applicable
            if hasattr(pipe, "set_adapters_scale") and lora_scale is not None:
                pipe.set_adapters_scale(lora_scale)
        except ValueError as e:
            # Return informative error if there are multiple safetensors and no weight name
            if "more than one weights file" in str(e):
                return f"Error: The repository '{lora_id}' contains multiple safetensors files. Please specify a weight name using the 'LoRA Weight Name' field.", seed
            else:
                return f"Error loading LoRA weights: {str(e)}", seed
    
    # Enable CPU offload for low VRAM
    pipe.enable_model_cpu_offload()
    
    # Generate video
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).frames[0]
    
    # Export to video
    temp_file = "output.mp4"
    export_to_video(output, temp_file, fps=output_fps)
    
    return temp_file, seed

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.HTML("""
    <p align="center">
        <svg version="1.1" viewBox="0 0 1200 295" xmlns="http://www.w3.org/2000/svg" xmlns:v="https://vecta.io/nano" width="400">
            <path d='M195.43 109.93a.18.18 0 0 1-.27-.03l-.17-.26a.13.12-43.2 0 1 .01-.16q21.87-21.98 42.02-37.95c17.57-13.93 36.61-26.36 57.23-34.01q13.26-4.92 24.23-4.1c10.91.81 18.01 8.05 19.12 18.82.79 7.76-.64 15.73-3.29 23.37q-5.87 16.92-15.46 32.47a1.05 1.02-40.3 0 0-.09.91q11.79 32.3 2.48 63.6c-13.06 43.95-53.78 72.83-99.49 71.06q-16.32-.64-30.92-6.27a1.88 1.88 0 0 0-1.65.15c-12.54 7.69-26.93 14.55-41.16 17.68-4.99 1.1-9.54.6-16.14.62a6.31 6.04 50.1 0 1-1.35-.15c-19.69-4.51-18.03-25.43-12.71-40.65q5.95-17.05 15.9-32.34a.89.88 50.1 0 0 .07-.82c-13.76-32.93-7.8-71.64 14.32-99.05q20.75-25.73 51.75-33.99 29.88-7.96 59.38 2.25a.18.18 0 0 1 .04.32l-11.17 7.52a1.18 1.17-55 0 1-.93.17q-26.82-6.3-51.8 2.83c-24.65 9.02-43.94 28.53-52.73 53.27q-8.3 23.33-2.99 48.49.7 3.32 3.15 10.35c7.87 22.55 24.1 40.02 45.62 50.35a.52.51 43.2 0 1 .03.91l-10.31 5.96a.7.68-44.3 0 1-.69 0q-27.53-15.57-41.3-43.86a.1.09-44.1 0 0-.17 0q-6.59 11.46-11.16 23.67c-4.39 11.73-11.35 38.04 10.45 38.5q6.38.13 11.55-1.22c17.2-4.51 32.86-13.33 47.66-23.11q24.23-16.04 49.48-39.51c28.57-26.55 55.11-56.92 74.11-90.87q5.96-10.65 10.32-23.77c2.38-7.15 5.16-20.81-.67-27.03q-3.67-3.92-8.08-4.48-10.87-1.39-22.3 2.5-11.06 3.77-18.33 7.43c-30.93 15.54-58.11 36.9-83.59 60.43zm100.11-36.16q10.18 9.98 17.31 22.67a.6.59 46.1 0 1-.01.59l-6.17 10.69a.34.33-42.8 0 1-.59-.02q-7.06-14.92-18.34-25.98-11.28-11.06-26.34-17.82a.34.33-48.4 0 1-.03-.59l10.56-6.38a.6.59 42.7 0 1 .59-.02q12.83 6.88 23.02 16.86zm430.69 78.36h10a1.13 1.12-74.2 0 0 .96-.54q11.1-17.8 23.93-38.19 1.8-2.85 5.39-3.1 3.42-.23 11.73-.11 1.88.02 2.61.66a2.23 2.08-54 0 1 .34 2.77q-12.01 19.41-29.2 46.98a.86.85 45.5 0 0 0 .93q12.8 20.23 28.66 46.12.22.36.29 1.09a1.98 1.97 85.7 0 1-1.85 2.16q-7.51.45-13.91-.05c-2.52-.2-4.25-2.09-5.57-4.24q-8.26-13.5-22.19-36.26a.74.72-15 0 0-.63-.35H726.2a.62.61 0 0 0-.62.61q-.03 31.31.02 35.17.07 5.25-5.02 5.31-7.87.1-10.26-.19-3.72-.45-3.71-4.69.05-35.54-.02-135.71c0-3.87.67-6.1 4.77-6.12q7.55-.05 10.37.06c3.4.13 3.94 3.29 3.92 6.34q-.22 43.68-.02 80.75a.6.6 0 0 0 .6.6zM191.9 74.2l1.52.2a.33.33 0 0 1 0 .65l-1.52.22a.33.33 0 0 0-.28.29l-.2 1.52a.33.33 0 0 1-.65 0l-.22-1.52a.33.33 0 0 0-.29-.28l-1.52-.2a.33.33 0 0 1 0-.65l1.52-.22a.33.33 0 0 0 .28-.29l.2-1.52a.33.33 0 0 1 .65 0l.22 1.52a.33.33 0 0 0 .29.28zm64.12 2.96l2.78.37a.33.33 0 0 1 0 .65l-2.78.39a.33.33 0 0 0-.28.29l-.37 2.78a.33.33 0 0 1-.65 0l-.39-2.78a.33.33 0 0 0-.29-.28l-2.78-.37a.33.33 0 0 1 0-.65l2.78-.39a.33.33 0 0 0 .28-.29l.37-2.78a.33.33 0 0 1 .65 0l.39 2.78a.33.33 0 0 0 .29.28zm150.31 27.5q-.37 2.75-.37 5.59.1 55.28.07 95.29-.01 5.69-4.97 5.57-4.23-.1-9.06-.04-4.92.06-4.92-4.56-.03-47.02-.01-122.76c0-4.45 2.16-5.1 6.27-5.1q17.99.04 19.41.04c4.14-.01 5.19 2.26 6.29 6.06q6.68 23.18 28.31 98.03a.22.22 0 0 0 .42-.01q26.48-97.12 26.96-99.03 1.26-5.01 5.72-5.05 9.8-.09 22.27.08c3.29.04 3.71 3.1 3.7 5.88q-.07 58.05-.18 118.23c-.01 4.45.51 7.92-4.69 8.13q-3.32.14-7.06.02c-5.17-.17-7.18.03-7.18-6.18q.01-66.62.01-100.28a.141.141 0 0 0-.28-.04q-20.97 79.04-26.98 101.79-1.27 4.8-5.99 4.76-4.96-.03-13.08-.01c-4.08.01-5.44-1.81-6.54-5.76q-17.37-62.23-27.98-100.66a.07.07 0 0 0-.14.01zm-172.34 1.26l2.49.36a.32.32 0 0 1 0 .63l-2.49.37a.32.32 0 0 0-.27.27l-.36 2.49a.32.32 0 0 1-.63 0l-.37-2.49a.32.32 0 0 0-.27-.27l-2.49-.36a.32.32 0 0 1 0-.63l2.49-.37a.32.32 0 0 0 .27-.27l.36-2.49a.32.32 0 0 1 .63 0l.37 2.49a.32.32 0 0 0 .27.27zm46.8 2.66l2.5.41a.34.34 0 0 1 0 .67l-2.5.4a.34.34 0 0 0-.28.28l-.41 2.5a.34.34 0 0 1-.67 0l-.4-2.5a.34.34 0 0 0-.28-.28l-2.5-.41a.34.34 0 0 1 0-.67l2.5-.4a.34.34 0 0 0 .28-.28l.41-2.5a.34.34 0 0 1 .67 0l.4 2.5a.34.34 0 0 0 .28.28zm311.55 38.61q.18-12.55-.16-15.38c-.44-3.69-4.05-3.64-6.93-3.64q-18.53.03-52.88-.01c-3.01 0-3.94-2.46-3.95-5.32q-.01-5.13.04-8.35.08-4.18 4.29-4.26 5.99-.11 59.86-.05c13.05.02 18.7 5.13 18.7 18.33q.01 61.89.01 63.24c.03 15.73-6.4 19.32-21.07 19.32q-30.49.01-43.5-.02c-14.08-.04-20.24-3.83-20.35-18.8q-.12-16.77.08-27.74c.21-12.16 6.75-16.58 18.27-16.58q22.02.01 46.85-.01a.74.74 0 0 0 .74-.73zm-.66 18.75q-26.48-.05-41.68.04-4.64.03-4.77 4.52-.22 8.08-.06 16.74c.12 6.3 2.92 5.93 8.53 5.94q14.25.01 34.08-.03c4.89-.02 4.57-4.12 4.6-8.32q.08-11.72-.06-18.26a.65.64 89.1 0 0-.64-.63zm45.34 45.09c-2.82.02-4.77-1.44-4.77-4.22q-.01-73.72.01-78.31c.05-12.96 5.29-18.22 18.24-18.3q11.13-.06 38.25-.02 4.76.01 4.86 4.11.13 4.95.08 8.97c-.05 3.99-1.85 4.93-5.85 4.92q-24.69-.01-26.97 0c-5.43 0-9.69-.88-9.69 5.79q-.01 66 .02 70.71.02 3.45-.84 4.58-1.47 1.96-4.11 1.92-9.21-.15-9.23-.15zm200.71-17.86q16.73 0 18.8-.04c5.15-.13 4.81-3.81 4.8-8.15q-.01-10.05.01-70.24 0-4.51 4.92-4.56 7.92-.08 10 .09 3.9.31 3.9 4.48.02 60.37-.01 77.78c-.02 15.09-6.99 18.68-20.65 18.54q-.42 0-21.77 0-21.35 0-21.77 0c-13.66.14-20.63-3.45-20.65-18.54q-.03-17.41-.01-77.78 0-4.17 3.9-4.48 2.08-.17 10-.09 4.92.05 4.92 4.56.02 60.19.01 70.24c-.01 4.34-.35 8.02 4.79 8.14q2.08.05 18.81.05zm63.52 13.64q-.01-73.72.01-78.31c.05-12.96 5.29-18.22 18.24-18.3q11.13-.06 38.25-.02 4.76.01 4.86 4.11.13 4.95.08 8.97c-.05 3.99-1.85 4.93-5.85 4.92q-24.69-.01-26.97 0c-5.43 0-9.69-.88-9.69 5.79q-.01 66 .02 70.71.02 3.45-.84 4.58-1.47 1.96-4.11 1.92-9.21-.15-9.23-.15c-2.82.02-4.77-1.44-4.77-4.22zm109.52-25.57q18.81-60.07 20.5-65.73 1.57-5.27 6.17-5.34 3.67-.05 10.57.07c2.6.05 3.39 2.42 2.65 4.66q-11.68 35.64-41.85 128.15-1.51 4.61-6.26 4.65-4.38.03-7.66-.07c-3.83-.12-3.5-3.08-2.61-6.07q3.74-12.68 8.78-29.76a1.97 1.91-47.2 0 0-.03-1.2q-19.47-53.62-34.46-94.88c-.96-2.65-1.38-5.23 2.18-5.46q1.66-.11 11.3-.09 4.8 0 6.58 4.96 13.98 38.91 23.84 66.12a.16.16 0 0 0 .3-.01zm-767.05-7.85q-.03-28.67.01-36.09.03-4-1.39-6.63c-2.4-4.47-7.99-5.48-11.98-2.79q-4.8 3.23-4.86 9.16-.17 16.69-.02 24.3a.47.47 0 0 1-.47.48h-10.36a.46.46 0 0 1-.46-.46q.02-21.54 0-23.86-.03-2.66-.94-5.2c-2.63-7.33-11.73-8.07-15.95-1.8q-1.99 2.96-1.98 7.75.05 21.93.01 22.99a.59.58-89 0 1-.58.57h-10.14a.57.57 0 0 1-.57-.57v-43.73a1.04 1.04 0 0 1 1.04-1.04h8.41a1.25 1.24-2.5 0 1 1.24 1.13l.3 3.59a.27.27 0 0 0 .44.19q.06-.06 1.28-1.44c2.92-3.32 7.69-4.52 12.01-4.15q8.1.68 11.68 7.78a.37.37 0 0 0 .64.03c4.8-7.6 13.82-9.19 22.01-6.61 8.67 2.72 11.91 11.02 11.96 19.52q.07 14.44.05 24.75a1.29 1.26 66.4 0 1-.32.85l-10.44 11.52a.36.36 0 0 1-.62-.24zm-87.22-51.13l1.95.24a.34.34 0 0 1 0 .67l-1.96.22a.34.34 0 0 0-.3.3l-.24 1.95a.34.34 0 0 1-.67 0l-.22-1.96a.34.34 0 0 0-.3-.3l-1.95-.24a.34.34 0 0 1 0-.67l1.96-.22a.34.34 0 0 0 .3-.3l.24-1.95a.34.34 0 0 1 .67 0l.22 1.96a.34.34 0 0 0 .3.3zm13.88 41.55q.18 0 .43.34a.87.82 79.1 0 0 .45.33l.6.17a.362.362 0 0 1-.14.71q-.38-.05-.48-.08-.04-.02-.4.44-.26.33-.45.33-.19 0-.45-.32-.37-.46-.41-.44-.1.03-.48.09a.363.363 0 0 1-.15-.71l.6-.18a.87.82-79.8 0 0 .45-.33q.24-.35.43-.35zm25.81 35.05q.01.16-.38.69a.6.59 33.3 0 1-.79.16q-1.23-.74-1.23-.8 0-.05 1.19-.85a.6.59-36.1 0 1 .79.12q.42.51.42.68z' fill="#100c6d" />
        </svg>
    </p>
    """)
    gr.Markdown("# Wan 2.1 T2V 1.3B with LoRA")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value="Wan2.1-T2V-1.3B",
                label="Model"
            )
            
            prompt = gr.Textbox(
                label="Prompt",
                value="",
                lines=3
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿",
                lines=3
            )
            
            with gr.Row():
                lora_id = gr.Textbox(
                    label="LoRA Model Repo (e.g., TheBulge/AndroWan-2.1-T2V-1.3B)",
                    value="TheBulge/AndroWan-2.1-T2V-1.3B"
                )
            
            with gr.Row():
                lora_weight_name = gr.Textbox(
                    label="LoRA Path in Repo (optional)",
                    value="safetensors/AndroWan_v32-0036_ema.safetensors",
                    info="Specify for repos with multiple .safetensors files, e.g.: adapter_model.safetensors, pytorch_lora_weights.safetensors, etc."
                )
                lora_scale = gr.Slider(
                    label="LoRA Scale",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.00,
                    step=0.05
                )
            
            with gr.Row():
                scheduler_type = gr.Dropdown(
                    choices=list(SCHEDULER_OPTIONS.keys()),
                    value="UniPCMultistepScheduler",
                    label="Scheduler"
                )
                flow_shift = gr.Slider(
                    label="Flow Shift",
                    minimum=1.0,
                    maximum=12.0,
                    value=3.0,
                    step=0.5,
                    info="2.0-5.0 for smaller videos, 7.0-12.0 for larger videos"
                )
            
            with gr.Row():
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1024,
                    value=832,
                    step=32
                )
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1792,
                    value=480,
                    step=30
                )
            
            with gr.Row():
                num_frames = gr.Slider(
                    label="Number of Frames (4k+1 is recommended, e.g. 33)",
                    minimum=17,
                    maximum=129,
                    value=33,
                    step=4
                )
                output_fps = gr.Slider(
                    label="Output FPS",
                    minimum=8,
                    maximum=30,
                    value=16,
                    step=1
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0,
                    maximum=15.0,
                    value=4.0,
                    step=0.5
                )
                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=10,
                    maximum=100,
                    value=24,
                    step=1
                )
            
            seed = gr.Number(
                label="Seed (-1 for random)",
                value=-1,
                precision=0,
                info="Set a specific seed for deterministic results"
            )
            
            generate_btn = gr.Button("Generate Video")
        
        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video")
            used_seed = gr.Number(label="Seed", precision=0)
            
    generate_btn.click(
        fn=generate_video,
        inputs=[
            model_choice,
            prompt,
            negative_prompt,
            lora_id,
            lora_weight_name,
            lora_scale,
            scheduler_type,
            flow_shift,
            height,
            width,
            num_frames,
            guidance_scale,
            num_inference_steps,
            output_fps,
            seed
        ],
        outputs=[output_video, used_seed]
    )
    
    gr.Markdown("""
    ## Tips for best results:
    - For smaller resolution videos, try lower values of flow shift (2.0-5.0)
    - For larger resolution videos, try higher values of flow shift (7.0-12.0)
    - Number of frames should be of the form 4k+1 (e.g., 33, 81)
    - Stick to lower frame counts. Even at 480p, an 81 frame sequence at 30 steps will nearly time out the request in this space.
    
    ## Using LoRAs with multiple safetensors files:
    If you encounter an error stating "more than one weights file", you need to specify the exact weight file name in the "LoRA Weight Name" field.
    You can find this by browsing the repository on Hugging Face and looking for the safetensors files (common names include: adapter_model.safetensors, pytorch_lora_weights.safetensors).
    """)

# demo.launch()