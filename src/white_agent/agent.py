"""White agent implementation (Qwen2.5-VL)."""
import os
import logging
import torch
import uvicorn
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Try to import Qwen2.5-VL, fallback to AutoModel
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    try:
        from transformers import Qwen25VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as QwenVLModel

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from src.my_util import parse_tags

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="video_understanding",
        name="Egocentric Video Understanding",
        description="Understands and answers questions about egocentric procedural errors in videos.",
        tags=["multimodal", "video", "procedural-understanding", "procedural error detection"],
        examples=[],
    )
    card = AgentCard(
        name="qwen2.5_vl_video_agent",
        description="Video understanding agent powered by Qwen2.5-VL-7B-Instruct",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


def load_video_segment(video_path: str, start_sec: float, end_sec: float, num_frames: int = 8):
    """Load video segment and uniformly sample num_frames frames, return as list of PIL Images.
    
    If both start_sec and end_sec are negative, sample from the entire video.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps() or 30.0

    # Determine frame indices
    if start_sec < 0 and end_sec < 0:
        logger.info("🌍 Global sampling mode: sampling entire video")
        start_frame = 0
        end_frame = total_frames - 1
    else:
      
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        if end_frame <= start_frame:
            end_frame = min(start_frame + num_frames, total_frames - 1)

    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    indices = np.clip(indices, 0, total_frames - 1)
    frames = vr.get_batch(indices).asnumpy()
    
    del vr  # release
    
    # Convert to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    logger.info(f"📹 Sampled frames: {start_frame} -> {end_frame} (indices: {indices.tolist()[:3]}...{indices.tolist()[-3:]})")
    
    return pil_frames


class Qwen25VLWhiteAgentExecutor(AgentExecutor):
    """Qwen2.5-VL White Agent Executor"""
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        self.model_path = model_path
        self.ctx_id_to_messages = {}
        
        logger.info(f"Loading Qwen2.5-VL-7B-Instruct from {model_path} on {device}...")
        
        # Load model
        self.model = QwenVLModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        logger.info("✅ Qwen2.5-VL-7B-Instruct loaded successfully.")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        
        # Parse tags
        tags = parse_tags(user_input)
        video_path = tags.get("video_path", "")
        question = tags.get("question", "")
        start_ts = tags.get("start_ts", "")
        end_ts = tags.get("end_ts", "")
        
        # timestamp
        try:
            start_sec = float(start_ts) if start_ts else 0.0
            end_sec = float(end_ts) if end_ts else 30.0
        except ValueError:
            start_sec, end_sec = 0.0, 30.0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Time range: {start_sec}s - {end_sec}s")
        logger.info(f"Question: {question[:100]}...")
        
        if not os.path.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            logger.error(error_msg)
            await event_queue.enqueue_event(
                new_agent_text_message(error_msg, context_id=context.context_id)
            )
            return
        
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Load video frames
            video_frames = load_video_segment(video_path, start_sec, end_sec, num_frames=8)
            logger.info(f"Loaded {len(video_frames)} frames")
            
            # Construct messages for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_frames,  # List of PIL Images
                            "fps": 1.0,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            # Prepare for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            logger.info("Generating answer...")
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            logger.info(f"Generated answer: {output_text[:100]}...")
            
            # answer back
            await event_queue.enqueue_event(
                new_agent_text_message(output_text, context_id=context.context_id)
            )
            
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(
    model_path=None,
    device="cuda:0",
    host="localhost",
    port=9002
):
    """
    Run White Agent (Qwen2.5-VL-7B-Instruct)
    
    Args:
        model_path: model's path
        device: device (e.g., "cuda:0")
        host: host address
        port: port number
    """
    if model_path is None:
        model_path = "/home/junlong_li/Qwen2.5-VL-7B-Instruct"
    
    print(f"Starting Qwen2.5-VL white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)
    
    executor = Qwen25VLWhiteAgentExecutor(model_path, device)
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
