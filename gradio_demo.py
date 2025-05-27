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
                value="",
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