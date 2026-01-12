"""Main entry point for EgoErrorVQA evaluation."""
import asyncio
import argparse
import sys
import logging
from src.launcher import launch_evaluation, launch_white_agent_only

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="EgoErrorVQA - Video Understanding Evaluation")
    parser.add_argument(
        "command",
        choices=["launch", "white-only"],
        help="Command to run: 'launch' for full evaluation, 'white-only' to start white agent only"
    )
    parser.add_argument(
        "--white-model-path",
        default=None,
        help="Path to Qwen2.5-VL model (default: /home/junlong_li/Qwen2.5-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--white-device",
        default="cuda:0",
        help="Device for white agent (e.g., cuda:0)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate (default: all)"
    )
    parser.add_argument(
        "--task-type",
        choices=["open-end", "multiple-choice"],
        default="open-end",
        help="Evaluation task type: 'open-end' for open-end QA with dual judges, 'multiple-choice' for multiple-choice QA."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9002,
        help="Port for white agent (white-only mode)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "launch":
            asyncio.run(launch_evaluation(
                white_model_path=args.white_model_path,
                white_device=args.white_device,
                datasets=args.datasets,
                task_type=args.task_type
            ))
        elif args.command == "white-only":
            launch_white_agent_only(
                white_model_path=args.white_model_path,
                white_device=args.white_device,
                port=args.port
            )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user (Ctrl+C), exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
