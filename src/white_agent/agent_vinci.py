"""White agent implementation - 视频多模态模型代理 (Vinci-8B)."""
import os
import logging
import torch
import uvicorn
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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
        name="Video Understanding",
        description="Understands and answers questions about egocentric videos",
        tags=["multimodal", "video", "procedural-understanding"],
        examples=[],
    )
    card = AgentCard(
        name="vinci_video_agent",
        description="Video understanding agent powered by Vinci-8B-base",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


# ========== 图像预处理 ==========
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    from torchvision import transforms as T
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=224, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video_segment(video_path: str, start_sec: float, end_sec: float, num_frames: int = 24):
    """Load video segment and uniformly sample num_frames frames."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps() or 30.0

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)
    if end_frame <= start_frame:
        end_frame = min(start_frame + num_frames, total_frames - 1)

    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    indices = np.clip(indices, 0, total_frames - 1)
    frames = vr.get_batch(indices).asnumpy()
    
    del vr  # 释放资源
    
    return frames


class VinciWhiteAgentExecutor(AgentExecutor):
    """Vinci White Agent Executor"""
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        self.model_path = model_path
        self.ctx_id_to_messages = {}
        
        logger.info(f"Loading Vinci-8B from {model_path} on {device}...")
        
        # Build transform
        self.transform = build_transform(input_size=448)
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("✅ Vinci-8B loaded successfully.")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        
        # 解析输入
        tags = parse_tags(user_input)
        video_path = tags.get("video_path", "")
        question = tags.get("question", "")
        start_ts = tags.get("start_ts", "")
        end_ts = tags.get("end_ts", "")
        
        # 转换时间戳
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
            frames = load_video_segment(video_path, start_sec, end_sec, num_frames=24)
            logger.info(f"Loaded {len(frames)} frames")
            
            # Preprocess frames
            pixel_values_list, num_patches_list = [], []
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame).convert('RGB')
                tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
                pixel_values = [self.transform(tile) for tile in tiles]
                pixel_values = torch.stack(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
            
            # Check patch consistency
            if len(set(num_patches_list)) > 1:
                logger.warning(f"Inconsistent patches: {num_patches_list}, using minimum")
                min_patches = min(num_patches_list)
                pixel_values_list = [pv[:min_patches] for pv in pixel_values_list]
                num_patches_list = [min_patches] * len(num_patches_list)
            
            pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(self.device)
            logger.info(f"Final tensor shape: {pixel_values.shape}, total patches: {sum(num_patches_list)}")

            # Build prompt
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            full_question = video_prefix + question

            # Generate response
            generation_config = dict(
                num_beams=1,
                max_new_tokens=128,
                do_sample=False,
            )
            
            logger.info("Generating answer...")
            with torch.inference_mode():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )
            
            output_text = response.strip()
            logger.info(f"Generated answer: {output_text[:100]}...")
            
            # 发送回答
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
    port=9003
):
    """
    启动 White Agent (Vinci-8B)
    
    Args:
        model_path: 模型路径
        device: 设备 (如 "cuda:0")
        host: 主机地址
        port: 端口号
    """
    if model_path is None:
        model_path = "Vinci-8B-base"
    
    print(f"Starting Vinci-8B white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)
    
    executor = VinciWhiteAgentExecutor(model_path, device)
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)