"""Run the Android advisory prompts against the same pinned model on a host."""
import argparse
from pathlib import Path
import subprocess

from executorch.extension.llm.custom_ops import custom_ops  # Register Llama kernels.
from executorch.kernels import quantized  # Register quantized kernels.
from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("classes", type=Path, help="Directory containing compiled AdvicePromptCheck")
parser.add_argument("--models", type=Path, default=Path("models"))
parser.add_argument("--java", default="java")
args = parser.parse_args()
runner = TextLLMRunner(str(args.models / "llama.pte"), str(args.models / "tokenizer.model"))
config = GenerationConfig(echo=False, max_new_tokens=6, temperature=0, num_bos=1)
for fixture, expected in [("line", "B"), ("clean", "A"), ("shifted", "AB"), ("compact", "AB"), ("still-clean", "AB")]:
    prompt = subprocess.check_output(
        [args.java, "-cp", str(args.classes), "io.surfaceguard.android.AdvicePromptCheck", fixture],
        text=True,
    )
    tokens = []
    runner.reset()
    runner.generate(prompt, config, tokens.append)
    answer = subprocess.check_output(
        [args.java, "-cp", str(args.classes), "io.surfaceguard.android.AdvicePromptCheck", "parse", "".join(tokens)],
        text=True,
    )
    print(f"{fixture}: {answer!r} (expected {expected})", flush=True)
    if not answer or answer not in expected:
        raise SystemExit(1)
