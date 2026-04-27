#!/usr/bin/env python3
"""
Convenient script to start vLLM server with common configurations.

Usage:
    python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --port 8000
    python start_vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.1 --gpu-util 0.9
    python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization awq
"""

import argparse
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM server with convenient options"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model name from Hugging Face (default: Llama-2-7b-chat)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run vLLM server on (default: 8000)",
    )
    parser.add_argument(
        "--gpu-util",
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        dest="gpu_memory_utilization",
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", "bitsandbytes", "none"],
        default="none",
        help="Quantization method (default: none)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "half", "bfloat16", "float"],
        default="auto",
        help="Data type (default: auto)",
    )
    parser.add_argument(
        "--disable-log-requests",
        action="store_true",
        help="Disable logging requests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    args = parser.parse_args()

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--host",
        "0.0.0.0",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    if args.quantization != "none":
        cmd.extend(["--quantization", args.quantization])

    if args.tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])

    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    if args.dtype != "auto":
        cmd.extend(["--dtype", args.dtype])

    if args.disable_log_requests:
        cmd.append("--disable-log-requests")

    cmd.extend(["--seed", str(args.seed)])

    print(f"Starting vLLM server with command:")
    print(" ".join(cmd))
    print()

    # Run the server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting vLLM server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
