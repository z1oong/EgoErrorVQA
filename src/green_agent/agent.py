"""Green agent"""
import os
import json
import random
import logging
import tomllib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import parse_tags, my_a2a

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Multiple-choice
# ==============================
PROCEDURAL_ERROR_GUIDANCE = """
You are an expert evaluator of procedural tasks in instructional videos.  
Your task is to determine whether the current step contains a procedural error, based on the full task procedure and the expected sequence of actions.

Below are the error types with clear definitions and examples:

1. Wrong object
   - Using an incorrect tool, material, or component. Misuse of equipment. Wrong preparation of materials.
   - Example: Using a red liquid instead of a yellow liquid; picking up the wrong screw.

2. Wrong action
   - Performing the correct step in an incorrect manner. Working in the wrong way or moving. Doing in wrong position. Measurement Error. Incorrect usage of temperature. Incorrect time for cooking. Motor error.
   - Example: Stirring instead of shaking; cutting with scissors instead of tearing.

3. Wrong order
   - Executing a step before or after its correct position in the sequence. Previous one is mistake, this action is also an ordering mistake but is caused by the preceding ordering mistakes in the context.
   - Example: Adding sugar after stirring coffee; connecting wires before powering off; Avocado is added after topping the leaves.

4. Omission
   - Skipping a necessary step entirely.
   - Example: Forgetting to coat the ramekin with spray; not connecting the battery before testing.

5. Unintended and unnecessary action
   - Performing an extra step that is not part of the procedure. Searching for an item. The action shouldn't have happened(this detach action is unnecessary).
   - Example: Wiping a clean tool; re-measuring already measured ingredients; grasping wrong objects and releasing them without using.

6. Correct wrong action
   - Realizing a prior mistake and fixing it (this is acceptable, but should be noted as a correction).
   - Example: Removing a wrongly placed part and replacing it correctly.

7. Equipment failure
   - A tool or material fails or malfunctions during the task (not user error).
   - Example: A cup breaks; a switch doesn't respond despite correct action.

8. Others
   - Any error that doesn't fit the above categories like slowness in movement.

Instructions:
- Compare the observed action against the full task procedure and step description.
- If the action is correct, output ONLY: correct
- If there is an error, output ONLY ONE of the error type names above (e.g., Wrong object, Wrong action).
"""

CHOICE_OPTIONS = [
    "correct",
    "Wrong object",
    "Wrong action",
    "Wrong order",
    "Correct wrong action",
    "Unintended and unnecessary action",
    "Others",
    "Equipment failure",
    "Omission"
]


def clean_answer(answer: str) -> str:
    """Clean model output to match options"""
    answer = answer.strip().strip('.,!?;:"')
    for opt in CHOICE_OPTIONS:
        if opt.lower() == answer.lower():
            return opt
    return answer


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


# ==============================
# Judge model
# ==============================
class QwenJudge:
    def __init__(self, model_path: str, device: str = "cuda:1"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        ).eval()

    def evaluate_answer(self, question: str, model_answer: str, ground_truth: str) -> dict:
        prompt = f"""You are an expert evaluator. Score the model's answer on accuracy and completeness, evaluate the quality of the model's answer by comparing it with the ground truth.
Question: {question}
Ground Truth: {ground_truth}
Model's Answer: {model_answer}

Rate the answer on a scale of 0 to 5:
5 = Perfect match, 4 = Minor error, 3 = Partially correct, 2 = Mostly wrong, 1 = Wrong, 0 = No response

Output ONLY: {{"score": int, "reason": "brief explanation"}}"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Only record the assistant's reply part (remove system and user messages)
        if "assistant" in response:
            clean_response = response.split("assistant")[-1].strip()
            logger.info(f"🔍 Qwen Judge Response: [{clean_response}]")
        else:
            logger.info(f"🔍 RAW Qwen output: [{response}]")
        
        try:
            import re
            
            if "assistant" in response:
                json_part = response.split("assistant")[-1].strip()
            else:
                json_part = response
                       
            json_match = re.search(r'\{.*\}', json_part, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)               
                json_str = json_str.rstrip(' ,\n\t\r]')
                result = json.loads(json_str)
                return {
                    "score": int(result.get("score", 0)),
                    "reason": str(result.get("reason", "No reason"))
                }
            else:
                raise ValueError("No JSON object found")

        except Exception as e:
            logger.warning(f"Qwen judge parse error: {e}")
        return {"score": 0, "reason": "Parse failed"}

    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class DeepSeekJudge:
    def __init__(self, model_path: str, device: str = "cuda:1"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        ).eval()

    def evaluate_answer(self, question: str, model_answer: str, ground_truth: str) -> dict:
        prompt = f"""You are an expert evaluator. Score the model's answer on accuracy and completeness, evaluate the quality of the model's answer by comparing it with the ground truth.
Question: {question}
Ground Truth: {ground_truth}
Model's Answer: {model_answer}

Scoring (0-5):
5 = Perfect, 4 = Minor issues, 3 = Partially correct, 2 = Mostly wrong, 1 = Wrong, 0 = No response

Respond ONLY with a valid JSON: {{"score": int, "reason": "brief explanation"}}"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            import re
            
            all_json_matches = list(re.finditer(r'\{[^{}]*\}', response))
            
            if all_json_matches:               
                for match in reversed(all_json_matches):
                    json_str = match.group(0).strip()
                    try:
                        result = json.loads(json_str)
                       
                        if "score" in result and isinstance(result.get("score"), int):
                            logger.info(f"🔍 DeepSeek Judge Response: [{json_str}]")
                            return {
                                "score": int(result["score"]),
                                "reason": str(result.get("reason", "No reason provided"))
                            }
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
            
            raise ValueError("No valid JSON with score found")

        except Exception as e:
            logger.warning(f"DeepSeek parse error: {e}")
            
            import re
            score_match = re.search(r'"score"\s*:\s*(\d+)', response)
            if score_match:
                return {"score": int(score_match.group(1)), "reason": "Fallback"}
            num_match = re.search(r'\b([0-5])\b', response)
            if num_match:
                return {"score": int(num_match.group(1)), "reason": "Fallback"}
            return {"score": 0, "reason": "Parse failed"}

    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


# ==============================
# Video path resolution
# ==============================
def get_video_path(video_root: str, video_subdir: str, sample: dict, dataset_name: str = None) -> str:
    
    
    VIDEO_PATH_PATTERNS = {
        "captaincook4d": [
            "{video_root}/{video_subdir}/{video_id}_360p.mp4",
        ],
        "epic_tent": [
            "{video_root}/{video_subdir}/0{video_id}.tent.gopro.MP4",
        ],
        "assembly101": [
            "{video_root}/{video_subdir}/{video_id}/HMC_84355350_mono10bit.mp4",
            "{video_root}/{video_subdir}/{video_id}/HMC_21110305_mono10bit.mp4",
        ],
        "egooops": [
            "{video_root}/{video_subdir}/{task_id}/{video_id}.MP4",
        ]
    }
    
    task_id = sample.get("task_id") or os.path.basename(sample.get("task", ""))
    video_id = sample.get("video_id", "")
    subject_id = sample.get("subject_id", "")
    recording_id = sample.get("recording_id", "")
    
    path_vars = {
        "video_root": video_root,
        "video_subdir": video_subdir,
        "video_id": video_id,
        "task_id": task_id,
        "subject_id": subject_id,
        "recording_id": recording_id,
    }
    
    candidates = []
    
    if dataset_name and dataset_name in VIDEO_PATH_PATTERNS:
        for pattern in VIDEO_PATH_PATTERNS[dataset_name]:
            path = pattern.format(**path_vars)
            candidates.append(path)
    else:
        candidates = [os.path.join(video_root, video_subdir, f"{video_id}.mp4")]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    if dataset_name == "assembly101":
        video_dir = os.path.join(video_root, video_subdir, video_id)
        if os.path.isdir(video_dir):
            for file in os.listdir(video_dir):
                if file.startswith("HMC_") and file.endswith(".mp4"):
                    return os.path.join(video_dir, file)
    
    return candidates[0] if candidates else os.path.join(video_root, video_subdir, f"{video_id}.mp4")


async def evaluate_with_white_agent_multiple_choice(
    white_agent_url: str,
    eval_config: dict,
    context_id: str = None
):
    """Run Multiple-Choice VQA"""
    
    # Create reusable A2A client
    logger.info(f"🔗 Creating reusable A2A client for {white_agent_url}...")
    white_client, white_httpx_client = await my_a2a.create_a2a_client(white_agent_url)
    logger.info("✅ A2A client created, will reuse connection for all requests")
    
    try:
        
        video_root = eval_config["video_root"]
        json_root = eval_config["json_root"]
        procedure_json_path = eval_config["procedure_json_path"]
        datasets_to_eval = eval_config.get("datasets", ["captaincook4d", "epic_tent", "assembly101", "egooops"])
        
        DATASETS = {
            "captaincook4d": "captaincook4d_answer.json",
            "epic_tent": "epic_tent_answer_sec.json",
            "assembly101": "assembly101_answers.json",
            "egooops": "egooops_answer.json"
        }
        
        VIDEO_SUBDIRS = {
            "captaincook4d": "captaincook4d",
            "epic_tent": "epic-tent",
            "assembly101": "assembly101",
            "egooops": "egooops"
        }
        
        # Load procedure.json
        with open(procedure_json_path, 'r', encoding='utf-8') as f:
            task_procedures = json.load(f)
        
        logger.info(f"Loaded task procedures from {procedure_json_path}")
        
        output_dir = "/home/junlong_li/egoerrorbench/results"
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        dataset_metrics = {}
        
        for dataset_name in datasets_to_eval:
            if dataset_name not in DATASETS:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
                
            json_file = DATASETS[dataset_name]
            json_path = os.path.join(json_root, json_file)
            
            if not os.path.exists(json_path):
                logger.warning(f"JSON file not found: {json_path}")
                continue
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                samples = list(data.values()) if data else []
            else:
                logger.error(f"Unexpected data format in {dataset_name}: {type(data)}")
                continue
            
            logger.info(f"Processing {dataset_name}: {len(samples)} samples")
            
            results = []
            correct_count = 0
            total_count = 0
            tp = fp = tn = fn = 0
            video_subdir = VIDEO_SUBDIRS.get(dataset_name, dataset_name)
            
            for idx, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    logger.warning(f"Skipping invalid sample in {dataset_name}: {type(sample)}")
                    continue
                
                try:
                    video_path = get_video_path(video_root, video_subdir, sample, dataset_name)
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video not found: {video_path}")
                    
                    # Get procedure
                    task_id = sample.get("task_id", "unknown")
                    procedure = task_procedures.get(task_id, "No procedure available.")
                    
                    original_desc = sample.get("action_annotation", "")
                    question = (
                        f"{PROCEDURAL_ERROR_GUIDANCE}\n\n"
                        f"Task Procedure: {procedure}\n\n"
                        f"Current Step: \"{original_desc}\"\n\n"
                    )
                    
                    # Get timestamps
                    start_time = sample.get("start_time", "")
                    end_time = sample.get("end_time", "")
                    
                    # Construct message to send to white agent
                    message_to_white = f"""<video_path>{video_path}</video_path>
                                            <question>{question}</question>
                                            <start_ts>{start_time}</start_ts>
                                            <end_ts>{end_time}</end_ts>"""
                    
                    logger.info(f"Sending to white agent for {dataset_name} sample {idx+1}/{len(samples)}...")
                    
                    # Send message
                    white_response = await my_a2a.send_message_with_client(
                        white_client, message_to_white, context_id=context_id
                    )
                    
                    from a2a.types import SendMessageSuccessResponse, Message
                    res_root = white_response.root
                    if not isinstance(res_root, SendMessageSuccessResponse):
                        raise ValueError(f"Unexpected response type: {type(res_root)}")
                    
                    res_result = res_root.result
                    if not isinstance(res_result, Message):
                        raise ValueError(f"Unexpected result type: {type(res_result)}")
                    
                    text_parts = get_text_parts(res_result.parts)
                    if len(text_parts) < 1:
                        raise ValueError("No text parts in response")
                    
                    raw_answer = text_parts[0].strip()
                    cleaned_answer = clean_answer(raw_answer)
                    ground_truth = sample.get("close_end_answer")
                    
                    if isinstance(ground_truth, list):
                        is_correct = any(cleaned_answer.lower() == gt.lower() for gt in ground_truth)
                    else:
                        is_correct = (cleaned_answer.lower() == ground_truth.lower())
                    
                    if is_correct:
                        correct_count += 1
                    
                    # Calculate confusion matrix
                    if isinstance(ground_truth, list):
                        actual_is_error = not any(gt.lower() == 'correct' for gt in ground_truth)
                    else:
                        actual_is_error = ground_truth.lower() != 'correct'
                    
                    pred_is_error = cleaned_answer.lower() != 'correct'
                    
                    if pred_is_error and actual_is_error:
                        if is_correct:
                            tp += 1
                        else:
                            fp += 1
                    elif pred_is_error and not actual_is_error:
                        fp += 1
                    elif not pred_is_error and not actual_is_error:
                        tn += 1
                    elif not pred_is_error and actual_is_error:
                        fn += 1
                    
                    total_count += 1
                    
                    # Sample ID is unique

                    sample_id = sample.get('sample_id')
                    if sample_id is None:
                        vid = sample.get('video_id', 'unknown')
                        start_t = sample.get('start_time', idx)
                        sample_id = f"{vid}_{start_t}"
                    
                    logger.info("=" * 80)
                    logger.info(f"📊 Sample {idx+1}/{len(samples)} - {dataset_name} - {sample_id}")
                    logger.info(f"❓ Question: {original_desc}")
                    logger.info(f"✅ Ground Truth: {ground_truth}")
                    logger.info(f"🤖 Model Answer: {cleaned_answer}")
                    logger.info(f"✓ Correct: {is_correct}")
                    logger.info("=" * 80)
                    
                    results.append({
                        "sample_id": f"{sample_id}_{dataset_name}",
                        "dataset": dataset_name,
                        "task_id": task_id,
                        "action_annotation": original_desc,
                        "ground_truth": ground_truth,
                        "white_agent_raw_answer": raw_answer,
                        "white_agent_cleaned_answer": cleaned_answer,
                        "is_correct": is_correct
                    })
                    
                except Exception as e:
                    logger.error(f"Error on {dataset_name} sample {idx}: {e}")
                    results.append({
                        "sample_id": f"unknown_{dataset_name}_{idx}",
                        "error": str(e),
                        "is_correct": False
                    })
                    total_count += 1
            
            # Calculate metrics
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            dataset_metrics[dataset_name] = {
                "total_samples": total_count,
                "correct_count": correct_count,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
            
            all_results[dataset_name] = results
            
            # Save dataset results
            dataset_file = os.path.join(output_dir, f"{dataset_name}_multiple_choice_results.json")
            with open(dataset_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ {dataset_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Calculate overall metrics
        total_samples = sum(m["total_samples"] for m in dataset_metrics.values())
        total_correct = sum(m["correct_count"] for m in dataset_metrics.values())
        total_tp = sum(m["tp"] for m in dataset_metrics.values())
        total_fp = sum(m["fp"] for m in dataset_metrics.values())
        total_tn = sum(m["tn"] for m in dataset_metrics.values())
        total_fn = sum(m["fn"] for m in dataset_metrics.values())
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        simple_avg_accuracy = sum(m["accuracy"] for m in dataset_metrics.values()) / len(dataset_metrics) if dataset_metrics else 0.0
        simple_avg_precision = sum(m["precision"] for m in dataset_metrics.values()) / len(dataset_metrics) if dataset_metrics else 0.0
        simple_avg_recall = sum(m["recall"] for m in dataset_metrics.values()) / len(dataset_metrics) if dataset_metrics else 0.0
        simple_avg_f1 = sum(m["f1"] for m in dataset_metrics.values()) / len(dataset_metrics) if dataset_metrics else 0.0
        
        final_report = {
            "summary": {
                "dataset_metrics": dataset_metrics,
                "overall_accuracy": round(overall_accuracy, 4),
                "overall_precision": round(overall_precision, 4),
                "overall_recall": round(overall_recall, 4),
                "overall_f1": round(overall_f1, 4),
                "total_samples": total_samples,
                "simple_average_accuracy": round(simple_avg_accuracy, 4),
                "simple_average_precision": round(simple_avg_precision, 4),
                "simple_average_recall": round(simple_avg_recall, 4),
                "simple_average_f1": round(simple_avg_f1, 4)
            },
            "detailed_results": all_results
        }
        
        # Save final report
        final_path = os.path.join(output_dir, "final_multiple_choice_combined_report.json")
        with open(final_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info("\n")
        logger.info("=" * 100)
        logger.info("                    🏆 MULTIPLE-CHOICE EVALUATION COMPLETE 🏆")
        logger.info("=" * 100)
        logger.info(f"📊 Total Datasets Evaluated: {len(dataset_metrics)}")
        logger.info(f"📈 Total Samples: {total_samples}")
        logger.info(f"✅ Total Correct: {total_correct}")
        logger.info("-" * 100)
        
        for dataset_name, metrics in dataset_metrics.items():
            logger.info(f"  📁 {dataset_name.upper():20} | Acc: {metrics['accuracy']:.4f} | "
                       f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        
        logger.info("-" * 100)
        logger.info(f"⭐ Overall Accuracy: {overall_accuracy:.4f}")
        logger.info(f"⭐ Overall Precision: {overall_precision:.4f}")
        logger.info(f"⭐ Overall Recall: {overall_recall:.4f}")
        logger.info(f"⭐ Overall F1: {overall_f1:.4f}")
        logger.info(f"💾 Final Report Saved: {final_path}")
        logger.info("=" * 100)
        logger.info("🎉" * 50)
        logger.info("\n")
        
        return final_report
    
    finally:
        
        logger.info("🔌 Closing A2A client connection...")
        await white_httpx_client.aclose()
        logger.info("✅ Connection closed")


async def evaluate_with_white_agent(
    white_agent_url: str,
    eval_config: dict,
    context_id: str = None
):
    """Open-End QA task with dual judge"""
    
    
    logger.info(f"🔗 Creating reusable A2A client for {white_agent_url}...")
    white_client, white_httpx_client = await my_a2a.create_a2a_client(white_agent_url)
    logger.info("✅ A2A client created, will reuse connection for all requests")
    
    try:
        
        video_root = eval_config["video_root"]
        json_root = eval_config["json_root"]
        procedure_json_path = eval_config["procedure_json_path"]
        datasets_to_eval = eval_config.get("datasets", ["captaincook4d", "epic_tent", "assembly101", "egooops"])
        judge_model_path = eval_config.get("judge_model_path", "/home/junlong_li/Qwen2.5-7B-Instruct")
        deepseek_model_path = eval_config.get("deepseek_model_path", "/home/junlong_li/deepseek-llm-7b-chat")
        judge_device = eval_config.get("judge_device", "cuda:1")
        
        DATASETS = {
            "captaincook4d": "captaincook4d_qa_pairs_updated.json",
            "epic_tent": "epic_tent_qa_pair.json",
            "assembly101": "assembly101_qa_pairs_updated.json",
            "egooops": "egooops_qa_pair.json"
        }
        
        VIDEO_SUBDIRS = {
            "captaincook4d": "captaincook4d",
            "epic_tent": "epic-tent",
            "assembly101": "assembly101",
            "egooops": "egooops"
        }
        
        # Load procedure.json
        with open(procedure_json_path, 'r', encoding='utf-8') as f:
            task_procedures = json.load(f)
        
        logger.info(f"Loaded task procedures from {procedure_json_path}")
        
        output_dir = "/home/junlong_li/egoerrorbench/results"
        os.makedirs(output_dir, exist_ok=True)
        
        all_qwen_results = {}
        all_deepseek_results = {}
        
        # ==============================
        # Stage 1: Qwen evaluation on all datasets
        # ==============================
        logger.info("=== STAGE 1: Qwen Evaluation on All Datasets ===")
        qwen_judge = QwenJudge(judge_model_path, device=judge_device)
        
        for dataset_name in datasets_to_eval:
            if dataset_name not in DATASETS:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
                
            json_file = DATASETS[dataset_name]
            json_path = os.path.join(json_root, json_file)
            
            if not os.path.exists(json_path):
                logger.warning(f"JSON file not found: {json_path}")
                continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"Processing {dataset_name}: {len(samples)} samples")
        
        qwen_results = []
        video_subdir = VIDEO_SUBDIRS.get(dataset_name, dataset_name)
        
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                logger.warning(f"Skipping invalid sample in {dataset_name}: {type(sample)}")
                continue
            
            try:
                video_path = get_video_path(video_root, video_subdir, sample, dataset_name)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
                
                if "qa_pairs" not in sample or len(sample["qa_pairs"]) == 0:
                    raise ValueError("Missing 'qa_pairs'")
                
                # Get procedure
                task_id = sample.get("task_id", "unknown")
                procedure = task_procedures.get(task_id, "No procedure available.")
                
                # Randomly select a question from qa_pairs
                qa_pair = random.choice(sample["qa_pairs"])
                original_question = qa_pair["question"]
                ground_truth = qa_pair["answer"]
                
                # Combine question: procedure + original question
                full_question = f"Task Procedure: {procedure}\n\nQuestion: {original_question}"
                
                # Get timestamps
                start_time = sample.get("start_time", "")
                end_time = sample.get("end_time", "")
                
                # Construct message to send to white agent
                message_to_white = f"""<video_path>{video_path}</video_path>
                                        <question>{full_question}</question>
                                        <start_ts>{start_time}</start_ts>
                                        <end_ts>{end_time}</end_ts>"""
                
                logger.info(f"Sending to white agent for {dataset_name} sample {idx}...")
                
                # Send message to white agent
                max_retries = 3
                model_answer = None
                for retry in range(max_retries):
                    try:
                        white_response = await my_a2a.send_message_with_client(
                            white_client, message_to_white, context_id=context_id
                        )
                        
                        from a2a.types import SendMessageSuccessResponse, Message
                        res_root = white_response.root
                        if not isinstance(res_root, SendMessageSuccessResponse):
                            raise ValueError(f"Unexpected response type: {type(res_root)}")
                        
                        res_result = res_root.result
                        if not isinstance(res_result, Message):
                            raise ValueError(f"Unexpected result type: {type(res_result)}")
                        
                        text_parts = get_text_parts(res_result.parts)
                        if len(text_parts) < 1:
                            raise ValueError("No text parts in response")
                        
                        model_answer = text_parts[0].strip()
                        logger.info(f"White agent response: {model_answer[:100]}...")
                        break
                        
                    except Exception as retry_e:
                        logger.warning(f"Retry {retry+1}/{max_retries} failed for sample {idx}: {retry_e}")
                        if retry == max_retries - 1:
                            raise
                        import asyncio
                        await asyncio.sleep(2)  
                
                if model_answer is None:
                    raise ValueError("Failed to get answer from white agent after retries")
                
                # Use Qwen for evaluation
                qwen_result = qwen_judge.evaluate_answer(original_question, model_answer, ground_truth)
                
                sample_id = sample.get('sample_id')
                if sample_id is None:
                    vid = sample.get('video_id', 'unknown')
                    start_t = sample.get('start_time', idx)
                    sample_id = f"{vid}_{start_t}"
                
                # Detailed evaluation results
                logger.info("=" * 80)
                logger.info(f"📊 Sample {idx+1}/{len(samples)} - {dataset_name} - {sample_id}")
                logger.info("-" * 80)
                logger.info(f"❓ Question: {original_question}")
                logger.info(f"✅ Ground Truth: {ground_truth}")
                logger.info(f"🤖 Model Answer: {model_answer}")
                logger.info(f"🎯 Qwen Score: {qwen_result['score']}/5")
                logger.info(f"💭 Qwen Reason: {qwen_result['reason']}")
                logger.info("=" * 80)
                
                qwen_results.append({
                    "sample_id": f"{sample_id}_{dataset_name}",
                    "dataset": dataset_name,
                    "video_path": video_path,
                    "question": original_question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "qwen_score": qwen_result["score"],
                    "qwen_reason": qwen_result["reason"]
                })
                
            except Exception as e:
                logger.error(f"Qwen error on {dataset_name} sample {idx}: {e}")
                qwen_results.append({
                    "sample_id": f"unknown_{dataset_name}_{idx}",
                    "dataset": dataset_name,
                    "error": str(e),
                    "qwen_score": 0,
                    "qwen_reason": "Error in Qwen stage"
                })
        
            all_qwen_results[dataset_name] = qwen_results
            logger.info(f"✅ Qwen evaluation complete for {dataset_name}: {len(qwen_results)} samples")
        
        # Release Qwen
        logger.info("🧹 Cleaning up Qwen judge...")
        qwen_judge.cleanup()
        del qwen_judge
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("✅ Qwen stage complete for all datasets. Memory released.")
        
        # ==============================
        # Stage 2: DeepSeek evaluation on all datasets
        # ==============================
        logger.info("=== STAGE 2: DeepSeek Evaluation on All Datasets ===")
        deepseek_judge = DeepSeekJudge(deepseek_model_path, device=judge_device)
        
        for dataset_name in datasets_to_eval:
            if dataset_name not in all_qwen_results:
                continue
            
            qwen_results = all_qwen_results[dataset_name]
            deepseek_results = []
            
            for qwen_res in qwen_results:
                if "error" in qwen_res:
                    deepseek_results.append({
                        "sample_id": qwen_res["sample_id"],
                        "deepseek_score": 0,
                        "deepseek_reason": "Skipped due to Qwen error"
                    })
                    continue
                
                try:
                    deepseek_result = deepseek_judge.evaluate_answer(
                        qwen_res["question"],
                        qwen_res["model_answer"],
                        qwen_res["ground_truth"]
                    )
                    
                    # Detailed printing of DeepSeek evaluation results
                    logger.info("=" * 80)
                    logger.info(f"📊 DeepSeek Evaluation - {qwen_res['sample_id']}")
                    logger.info("-" * 80)
                    logger.info(f"❓ Question: {qwen_res['question']}")
                    logger.info(f"🤖 Model Answer: {qwen_res['model_answer']}")
                    logger.info(f"🎯 DeepSeek Score: {deepseek_result['score']}/5")
                    logger.info(f"💭 DeepSeek Reason: {deepseek_result['reason']}")
                    logger.info("=" * 80)
                    
                    deepseek_results.append({
                        "sample_id": qwen_res["sample_id"],
                        "deepseek_score": deepseek_result["score"],
                        "deepseek_reason": deepseek_result["reason"]
                    })
                except Exception as e:
                    logger.error(f"DeepSeek error on {qwen_res['sample_id']}: {e}")
                    deepseek_results.append({
                        "sample_id": qwen_res["sample_id"],
                        "deepseek_score": 0,
                        "deepseek_reason": "Error in DeepSeek stage"
                    })
            
            all_deepseek_results[dataset_name] = deepseek_results
            logger.info(f"✅ DeepSeek evaluation complete for {dataset_name}: {len(deepseek_results)} samples")
        
        # Release DeepSeek
        logger.info("🧹 Cleaning up DeepSeek judge...")
        deepseek_judge.cleanup()
        del deepseek_judge
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ DeepSeek stage complete for all datasets. Memory released.")
        
        # ==============================
        # Stage 3: Merge results + calculate metrics
        # ==============================
        all_final_results = {}
        dataset_metrics = {}
        
        for dataset_name in datasets_to_eval:
            if dataset_name not in all_qwen_results or dataset_name not in all_deepseek_results:
                continue
            
            qwen_results = all_qwen_results[dataset_name]
            deepseek_results = all_deepseek_results[dataset_name]
            
            final_results = []
            total_avg = 0.0
            successful_count = 0
            
            for q_res, d_res in zip(qwen_results, deepseek_results):
                if "error" in q_res or "error" in d_res:
                    final_results.append({**q_res, **d_res})
                    continue
                
                avg_score = (q_res["qwen_score"] + d_res["deepseek_score"]) / 2.0
                
                # Detailed printing of final merged results
                logger.info("🎯" * 40)
                logger.info(f"📋 FINAL RESULT - {q_res['sample_id']}")
                logger.info("-" * 80)
                logger.info(f"❓ Question: {q_res['question']}")
                logger.info(f"✅ Ground Truth: {q_res['ground_truth']}")
                logger.info(f"🤖 Model Answer: {q_res['model_answer']}")
                logger.info(f"📊 Qwen Score: {q_res['qwen_score']}/5 - {q_res['qwen_reason']}")
                logger.info(f"📊 DeepSeek Score: {d_res['deepseek_score']}/5 - {d_res['deepseek_reason']}")
                logger.info(f"⭐ Final Average Score: {avg_score:.2f}/5")
                logger.info("🎯" * 40)
                
                final_results.append({
                    **q_res,
                    **d_res,
                    "final_avg_score": avg_score
                })
                total_avg += avg_score
                successful_count += 1
            
            final_avg = total_avg / successful_count if successful_count > 0 else 0.0
            
            dataset_metrics[dataset_name] = {
                "total_samples": len(qwen_results),
                "successful_evaluations": successful_count,
                "final_average_score": round(final_avg, 2)
            }
            
            all_final_results[dataset_name] = final_results
            
            # Save results for each dataset
            dataset_file = os.path.join(output_dir, f"{dataset_name}_dual_judge_results.json")
            with open(dataset_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            # ===== Dataset summary statistics =====
            logger.info("=" * 100)
            logger.info(f"📊 DATASET SUMMARY: {dataset_name.upper()}")
            logger.info("=" * 100)
            logger.info(f"📈 Total Samples: {len(qwen_results)}")
            logger.info(f"✅ Successful Evaluations: {successful_count}")
            logger.info(f"❌ Failed Evaluations: {len(qwen_results) - successful_count}")
            logger.info(f"⭐ Average Score: {final_avg:.2f}/5")
            logger.info(f"💾 Results saved to: {dataset_file}")
            logger.info("=" * 100)
        
        # Calculate overall metrics
        total_samples = sum(m["total_samples"] for m in dataset_metrics.values())
        total_successful = sum(m["successful_evaluations"] for m in dataset_metrics.values())
        
        # Weighted average score
        weighted_avg = 0.0
        for dataset_name, metrics in dataset_metrics.items():
            if metrics["successful_evaluations"] > 0:
                weighted_avg += metrics["final_average_score"] * metrics["successful_evaluations"]
        overall_avg = weighted_avg / total_successful if total_successful > 0 else 0.0
        
        # Simple average score
        simple_avg = sum(m["final_average_score"] for m in dataset_metrics.values()) / len(dataset_metrics) if dataset_metrics else 0.0
        
        final_report = {
            "summary": {
                "dataset_metrics": dataset_metrics,
                "overall_average_score": round(overall_avg, 2),
                "simple_average_score": round(simple_avg, 2),
                "total_samples": total_samples,
                "total_successful_evaluations": total_successful
            },
            "detailed_results": all_final_results
        }
        
        # Save final report
        final_path = os.path.join(output_dir, "final_dual_judge_combined_report.json")
        with open(final_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Overall evaluation complete summary
        logger.info("\n")
        logger.info("🎉" * 50)
        logger.info("=" * 100)
        logger.info("                        🏆 OVERALL EVALUATION COMPLETE 🏆")
        logger.info("=" * 100)
        logger.info(f"📊 Total Datasets Evaluated: {len(dataset_metrics)}")
        logger.info(f"📈 Total Samples: {total_samples}")
        logger.info(f"✅ Total Successful: {total_successful}")
        logger.info(f"❌ Total Failed: {total_samples - total_successful}")
        logger.info("-" * 100)
        
        # Detailed results for each dataset
        for dataset_name, metrics in dataset_metrics.items():
            success_rate = (metrics["successful_evaluations"] / metrics["total_samples"] * 100) if metrics["total_samples"] > 0 else 0
            logger.info(f"  📁 {dataset_name.upper():20} | Samples: {metrics['total_samples']:3} | "
                       f"Success: {metrics['successful_evaluations']:3} ({success_rate:5.1f}%) | "
                       f"Avg Score: {metrics['final_average_score']:.2f}/5")
        
        logger.info("-" * 100)
        logger.info(f"⭐ Overall Weighted Average Score: {overall_avg:.2f}/5")
        logger.info(f"⭐ Simple Average Score: {simple_avg:.2f}/5")
        logger.info(f"💾 Final Report Saved: {final_path}")
        logger.info("=" * 100)
        logger.info("🎉" * 50)
        logger.info("\n")
        
        return final_report
    
    finally:
       
        logger.info("🔌 Closing A2A client connection...")
        await white_httpx_client.aclose()
        logger.info("✅ Connection closed")


# ==============================
# Green Agent Executor
# ==============================
class EgoErrorVQAGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Parse task
        logger.info("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        eval_config_str = tags["eval_config"]
        eval_config = json.loads(eval_config_str)
        
        # Get task type, default to open-end
        task_type = eval_config.get("task_type", "open-end")
        
        logger.info(f"Green agent: Starting evaluation (task_type={task_type})...")
        
        # Execute different evaluations based on task type
        if task_type == "multiple-choice":
            final_report = await evaluate_with_white_agent_multiple_choice(
                white_agent_url,
                eval_config,
                context_id=context.context_id
            )
        else:
            # Default to Open-End evaluation
            final_report = await evaluate_with_white_agent(
                white_agent_url,
                eval_config,
                context_id=context.context_id
            )
        
        # Send results
        summary_text = f"""Evaluation Complete!

Overall Average Score: {final_report['summary']['overall_average_score']}
Simple Average Score: {final_report['summary']['simple_average_score']}
Total Samples: {final_report['summary']['total_samples']}
Successful Evaluations: {final_report['summary']['total_successful_evaluations']}

Dataset Metrics:
{json.dumps(final_report['summary']['dataset_metrics'], indent=2)}

Full results saved to: /home/junlong_li/egoerrorbench/results/
"""
        
        await event_queue.enqueue_event(
            new_agent_text_message(summary_text, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def start_green_agent(agent_name="egoerrorvqa_green_agent", host="localhost", port=9001):
    print("Starting EgoErrorVQA green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    request_handler = DefaultRequestHandler(
        agent_executor=EgoErrorVQAGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
