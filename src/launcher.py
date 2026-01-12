"""Launcher module."""

import multiprocessing
import json
import signal
import sys
import atexit
import logging
import time
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.my_util import my_a2a

logger = logging.getLogger(__name__)

# Global list of active processes for cleanup
_active_processes = []


def cleanup_processes():
    """Clean up all active processes"""
    global _active_processes
    logger.info("🧹 Cleaning up all processes...")
    
    for proc_name, proc in _active_processes:
        if proc and proc.is_alive():
            logger.info(f"  Terminating {proc_name}...")
            proc.terminate()
            try:
                proc.join(timeout=5)  # Wait up to 5 seconds
                if proc.is_alive():
                    logger.warning(f"  Force killing {proc_name}...")
                    proc.kill()
                    proc.join()
                logger.info(f"  ✅ {proc_name} stopped")
            except Exception as e:
                logger.error(f"  Error stopping {proc_name}: {e}")
    
    _active_processes.clear()
    logger.info("✅ All processes cleaned up")


def signal_handler(signum, frame):
    """Handle termination signals"""
    signal_name = signal.Signals(signum).name
    logger.warning(f"⚠️  Received signal {signal_name} ({signum}), shutting down gracefully...")
    cleanup_processes()
    sys.exit(0)


# Register signal handlers and exit cleanup
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill
atexit.register(cleanup_processes)             # Clean up on normal exit

async def launch_evaluation(
    white_model_path=None,
    white_device="cuda:0",
    datasets=None,
    task_type="open-end"
):
    """
    Launch the full evaluation with green and white agents.
    
    Args:
        white_model_path: Path to the Qwen2.5-VL model
        white_device: Device used by the White agent
        datasets: List of datasets to evaluate
        task_type: Evaluation task type ("open-end" or "multiple-choice")
    """
    global _active_processes
    
    if datasets is None:
        datasets = ["captaincook4d", "epic_tent", "assembly101", "egooops"]
    
    p_green = None
    p_white = None
    
    try:
        # 启动 green agent
        print("🚀 Launching green agent...")
        green_address = ("localhost", 9001)
        green_url = f"http://{green_address[0]}:{green_address[1]}"
        p_green = multiprocessing.Process(
            target=start_green_agent, args=("egoerrorvqa_green_agent", *green_address),
            daemon=False  # 不使用守护进程，以便正确清理
        )
        p_green.start()
        _active_processes.append(("Green Agent", p_green))
        
        if not await my_a2a.wait_agent_ready(green_url, timeout=30):
            raise RuntimeError("Green agent not ready in time")
        print("✅ Green agent is ready.")

        # Launch white agent
        print("🚀 Launching white agent (Qwen2.5-VL)...")
        white_address = ("localhost", 9002)
        white_url = f"http://{white_address[0]}:{white_address[1]}"
        p_white = multiprocessing.Process(
            target=start_white_agent,
            kwargs={
                "model_path": white_model_path,
                "device": white_device,
                "host": white_address[0],
                "port": white_address[1]
            },
            daemon=False
        )
        p_white.start()
        _active_processes.append(("White Agent", p_white))
        
        if not await my_a2a.wait_agent_ready(white_url, timeout=60):
            raise RuntimeError("White agent not ready in time")
        print("✅ White agent is ready.")

        
        print("📤 Sending task description to green agent...")
        eval_config = {
            "video_root": "/home/junlong_li/EgoProceBench_video",
            "json_root": "/home/junlong_li/EgoProceBench_video/json",
            "procedure_json_path": "/home/junlong_li/EgoProceBench_video/json/procedure.json",
            "datasets": datasets,
            "task_type": task_type,
            "judge_model_path": "/home/junlong_li/Qwen2.5-7B-Instruct",
            "deepseek_model_path": "/home/junlong_li/deepseek-llm-7b-chat",
            "judge_device": "cuda:1"
        }
        
        task_text = f"""
Your task is to evaluate a video understanding agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>
You should use the following evaluation configuration:
<eval_config>
{json.dumps(eval_config, indent=2)}
</eval_config>
        """
        
        print("📋 Task description:")
        print(task_text)
        print("📤 Sending...")
        response = await my_a2a.send_message(green_url, task_text)
        print("✅ Response from green agent:")
        print(response)

        print("\n🎉 Evaluation complete!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Error during evaluation: {e}")
        raise
    finally:
        # Clean up processes
        print("\n🛑 Shutting down agents...")
        cleanup_processes()
        print("👋 Shutdown complete.")


def launch_white_agent_only(
    white_model_path=None,
    white_device="cuda:0",
    port=9002
):
    """
    Launch only the white agent (for standalone testing)
    
    Args:
        white_model_path: Path to the Qwen2.5-VL model
        white_device: Device
        port: Port number
    """
    print("Launching white agent (Qwen2.5-VL) only...")
    start_white_agent(
        model_path=white_model_path,
        device=white_device,
        host="localhost",
        port=port
    )
