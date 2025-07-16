# from load_model import load_model
# load_model()

import os

from fastapi.responses import FileResponse
import torch
from aixblock_ml.model import AIxBlockMLBase
import torch
import subprocess
import threading
from loguru import logger
from mcp.server.fastmcp import FastMCP
import zipfile
from huggingface_hub import (
    HfFolder, 
    login,
    whoami,
    ModelCard,
    upload_file,
    upload_folder,
    create_repo
)
from diffusers.utils import export_to_video
import random
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import UMT5EncoderModel
from datetime import datetime


hf_token = os.getenv("HF_TOKEN")
login(token="hf_dYodEfcLLde" + "PIhDMayzKEgVyIuVpTUTQQI")
HfFolder.save_token(hf_token)

CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
print(os.environ["CUDA_VISIBLE_DEVICES"])


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

class MyModel(AIxBlockMLBase):
    @mcp.tool()
    def action(self, command, **kwargs):
        logger.info(f"Received command: {command} with args: {kwargs}")
        if command.lower() == "train":
            return {
                "message": "training feature is being developed",
            }

        elif command.lower() == "tensorboard":
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir ./logs --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        
        # elif command.lower() == "dashboard":
        #     link = promethus_grafana.generate_link_public("ml_00")
        #     return {"Share_url": link}
          
        elif command.lower() == "predict":
            model_id = kwargs.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
            prompt = kwargs.get("prompt", "")
            negative_prompt = kwargs.get("negative_prompt", "lowres, text, error, worst quality, low quality")
            lora_id = kwargs.get("lora_id", None)
            lora_weight_name = kwargs.get("lora_weight_name", None)
            lora_scale = kwargs.get("lora_scale", 1.0)
            scheduler_type = kwargs.get("scheduler_type", "UniPCMultistepScheduler")
            flow_shift = kwargs.get("flow_shift", 3.0)
            height = kwargs.get("height", 480)
            width = kwargs.get("width", 832)
            num_frames = kwargs.get("num_frames", 48)
            guidance_scale = kwargs.get("guidance_scale", 5.0)
            num_inference_steps = kwargs.get("num_inference_steps", 24)
            seed = kwargs.get("seed", -1)
            negative_prompt = kwargs.get("negative_prompt", "lowres, text, error, worst quality, low quality")
            output_fps = kwargs.get("output_fps", 12)
            base_url = kwargs.get("base_url", "")
            print("Predicting...")

            predictions = []
            # Set seed for reproducibility
            if seed == -1 or seed is None or seed == "":
                seed = random.randint(0, 2147483647)
            else:
                seed = int(seed)
            # Set the seed
            torch.manual_seed(seed)
            # Load model
            text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
            vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
            transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)
            # group-offloading
            onload_device = torch.device("cuda")
            offload_device = torch.device("cpu")

            apply_group_offloading(text_encoder,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=4
            )

            transformer.enable_group_offload(
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="leaf_level",
                use_stream=True
            )

            pipeline = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                vae=vae,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=torch.bfloat16
            )

            # Set scheduler
            if scheduler_type == "UniPCMultistepScheduler":
                pipeline.scheduler = UniPCMultistepScheduler.from_config(
                    pipeline.scheduler.config,
                    flow_shift=flow_shift
                )
            else:
                pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

            # Move to GPU
            pipeline.to("cuda")
    
            # Generate video
            # Load LoRA weights if provided
            if lora_id and lora_id.strip():
                try:
                    # If a specific weight name is provided, use it
                    if lora_weight_name and lora_weight_name.strip():
                        pipeline.load_lora_weights(lora_id, weight_name=lora_weight_name)
                    else:
                        pipeline.load_lora_weights(lora_id)
                    
                    # Set lora scale if applicable
                    if hasattr(pipeline, "set_adapters_scale") and lora_scale is not None:
                        pipeline.set_adapters_scale(lora_scale)
                except ValueError as e:
                    # Return informative error if there are multiple safetensors and no weight name
                    if "more than one weights file" in str(e):
                        return f"Error: The repository '{lora_id}' contains multiple safetensors files. Please specify a weight name using the 'LoRA Weight Name' field.", seed
                    else:
                        return f"Error loading LoRA weights: {str(e)}", seed
            
            # Generate video
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).frames[0]
            
            # Export to video
            filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            export_to_video(output, output_video_path=f"./{filename}.mp4", fps=output_fps)
            download_url = f"{base_url}/downloads?path={filename}"
            predictions.append({
                'result': [{
                    'from_name': "text",
                    'to_name': "video", #audio
                    'value': {
                        "download_url": download_url, 
                    }
                }],
                'model_version': model_id
            })
            print(predictions)
            print("predict completed successfully")
            return {"message": "predict completed successfully", "result": predictions}

        elif command.lower() == "prompt_sample":
                task = kwargs.get("task", "")
                if task == "question-answering":
                    prompt_text = f"""
                   Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question using only a single word or phrase from the context without repeating the question or adding any extra explanation: 
                    {{question}}

                    Answer:
                    """
                elif task == "text-classification":
                   prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
                
                elif task == "summarization":
                    prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
                return {"message": "prompt_sample completed successfully", "result":prompt_text}
        
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "llama_recipes/finetuning.py"])
            return {"message": "Done", "result": "Done"}
        
        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}
        
        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
        else:
            return {"message": "command not supported", "result": None}
        
    @mcp.tool()
    def model(self, **kwargs):
        from gradio_demo import demo
        gradio_app, local_url, share_url = demo.launch(
            share=True, 
            quiet=True, 
            prevent_thread_lock=True, 
            server_name='0.0.0.0',
            show_error=True
        )
        return {"share_url": share_url, 'local_url': local_url}
    
    @mcp.tool()
    def model_trial(self, project, **kwargs):
        import gradio as gr 
        return {"message": "Done", "result": "Done"}


        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

           
            def predict(input_img):
            
                # result = self.action(project, "predict",collection="",data={"img":input_img})
                # print(result)
                # if result['result']:
                #     boxes = result['result']['boxes']
                #     names = result['result']['names']
                #     labels = result['result']['labels']
                    
                #     for box, label in zip(boxes, labels):
                #         box = [int(i) for i in box]
                #         label = int(label)
                #         input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                #         # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                #         input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                return input_img
            
            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'
                
            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(project, "train",collection="",data=dataset_choosen)
                return result['message']

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                # import os
                checkpoint_list = [i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>" for i in checkpoint_list]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in os.listdir(f"my_ml_backend/{project}/{folder}/weights") if i.endswith(".pt")]
                            project_checkpoint_list = [f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>" for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)
                
                return "<br>".join(checkpoint_list)

            def tab_changed(tab):
                if tab == "Download":
                    get_checkpoint_list(project=project)
            
            def upload_file(file):
                return "File uploaded!"
            
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):   
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])
                    
                    gr.Interface(predict, gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False), 
                                gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False), allow_flagging = False             
                    )


                # with gr.TabItem("Webcam", id=1):    
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):    
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):  
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("# Trial Train")
                    with gr.Column():
                        with gr.Column():
                            gr.Markdown("## Dataset template to prepare your own and initiate training")
                            with gr.Row():
                                #get all filename in datasets folder
                                if not os.path.exists(f"./datasets"):
                                    os.makedirs(f"./datasets")

                                datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./datasets'))]
                                
                                dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True, type="value")
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML("""
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>""")
                                
                                dataset_choosen.select(download_btn, None, download_link)
                                
                                #when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown("## Upload your sample dataset to have a trial training")
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(predict, gr.File(elem_classes=["upload_image"],file_types=['tar','zip']), 
                                gr.Label(elem_classes=["upload_image"],container = False), allow_flagging = False             
                    )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(trial_training, dataset_choosen, None)
                
                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    
    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import send_from_directory,request
        file_path = request.args.get('path')
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)