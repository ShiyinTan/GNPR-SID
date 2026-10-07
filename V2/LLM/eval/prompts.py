"""Prompt strings aligned with LLaMA-Factory chat templates.

Training joins ``instruction`` and ``input`` with a newline, then wraps that
user message in the template named in the training yaml. Evaluation has to use
the same template.
"""

from __future__ import annotations

QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

TEMPLATES = ("qwen", "qwen3_nothink", "llama3", "glm4")


def user_content(instruction: str, user_input: str) -> str:
    parts = [part.strip() for part in (instruction, user_input) if part and part.strip()]
    return "\n".join(parts)


def build_prompt(instruction: str, user_input: str, template: str) -> str:
    content = user_content(instruction, user_input)
    if template == "qwen":
        return (
            "<|im_start|>system\n"
            f"{QWEN_SYSTEM}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if template == "qwen3_nothink":
        return (
            "<|im_start|>user\n"
            f"{content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if template == "llama3":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if template == "glm4":
        return f"[gMASK] <|user|>\n{content}<|assistant|>"
    known = ", ".join(TEMPLATES)
    raise ValueError(f"Unknown template {template!r}. Choose one of: {known}")
