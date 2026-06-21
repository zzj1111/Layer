# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time

import fire
import numpy as np
import vllm
from jinja2 import Template

from datasets import load_from_disk
from verl.utils.reward_score.oat_math_grader import (
    answer_tag_reward_fn,
    boxed_reward_fn,
)


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


# The following two templates are used to evaluate baselines from other projects.
def apply_prime_zero_template(question: str):
    """https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME-Zero"""
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    return f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"


def apply_open_reasoner_zero_template(question: str):
    "https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/e008f6d95f0b9a0e992f6b8bac912515b50a4634/playground/zero_setting_base.py"
    prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""
    prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""
    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=question)
    prompt_template = Template(prompt_template_jinja)
    return prompt_template.render(bos_token="", prompt=prompt_instruction)


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime", "amc", "math", "minerva", "olympiad_bench"],
    template: str = "qwen_math",
    dataset_name: str = "./datasets/evaluation_suite",
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 3000,
    max_model_len: int = 4096,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 1,
    n_samples_overrides: str = "",  # per-task n, e.g. "aime:8,aime25:8" (other tasks use n_samples)
    max_test: int = 999999,
    save: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    shard: str = "0/1",  # "i/N" -> this proc handles every N-th prompt starting at offset i
    save_dir: str = "/tmp/eval_results",
):

    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=2,
        seed=int(time.time_ns()),
    )

    model = vllm.LLM(
        model_name,
        swap_space=32,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    shard_idx, shard_total = (int(x) for x in shard.split("/"))
    assert 0 <= shard_idx < shard_total, f"bad shard {shard}"

    if "prime" in model_name.lower():
        template = "prime-zero"
    if "open-reasoner-zero" in model_name.lower():
        template = "open-reasoner-zero"

    if "instruct" in model_name.lower() and "instruct" not in template:
        input(
            f"{model_name}\n{template}\ninstruct model but not instruct template! continue?"
        )

    print("Using template:", template)
    if template in ["qwen_math", "no"]:
        math_reward_fn = boxed_reward_fn
        # sampling_params.stop = [
        #     "</s>",
        #     "<|im_end|>",
        #     "<|endoftext|>",
        #     "<|im_start|>",
        #     "\nUser:",
        # ]
        # sampling_params.stop_token_ids = [151645, 151643]
        if template == "qwen_math":
            apply_template = apply_qwen_math_template
        else:
            apply_template = lambda x: x

    elif template == "r1":
        math_reward_fn = answer_tag_reward_fn
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True
        apply_template = apply_r1_template
    elif template == "prime-zero":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_prime_zero_template
    elif template == "open-reasoner-zero":
        from verl.utils.reward_score.oat_math_grader import answer_tag_reward_fn_for_orz

        math_reward_fn = answer_tag_reward_fn_for_orz
        apply_template = apply_open_reasoner_zero_template
    elif template == "llama-instruct":

        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n",
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

    elif template == "r1d":  # r1-distill
        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )

    elif template == "r1d_box":  # r1-distill + Skywork-OR1's appended boxed instruction (paper-exact eval)
        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # exact string Skywork-OR1 appends in or1_data/eval/aime*.parquet (no system prompt)
        SKYWORK_EVAL_INSTRUCTION = " Let's think step by step and output the final answer within \\boxed{}."

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [{"content": question + SKYWORK_EVAL_INSTRUCTION, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )

    else:
        raise ValueError

    # per-task n_samples overrides, e.g. "aime:8,aime25:8" -> those tasks use n=8; others use n_samples.
    n_overrides = {}
    for tok in n_samples_overrides.split(","):
        tok = tok.strip()
        if tok and ":" in tok:
            t, v = tok.rsplit(":", 1)
            n_overrides[t.strip()] = int(v.strip())
    if n_overrides:
        print(f"per-task n_samples overrides: {n_overrides} (default n={n_samples})")

    results = {}
    avg_lens = {}
    max_lens = {}
    formatted = {}
    to_be_saved = []
    per_task_raw = {}  # task -> list[(orig_idx, reward, length)] for DP merging
    for task_name, dataset in load_from_disk(dataset_name).items():
        if task_name not in tasks:
            continue
        prompts_all = dataset["problem"][:max_test]
        targets_all = dataset["answer"][:max_test]
        # data-parallel shard: stride slicing keeps shards roughly balanced
        prompts = prompts_all[shard_idx::shard_total]
        targets = targets_all[shard_idx::shard_total]
        orig_idx = list(range(shard_idx, len(prompts_all), shard_total))

        prompts = list(map(apply_template, prompts))
        task_n = n_overrides.get(task_name, n_samples)
        task_sampling_params = vllm.SamplingParams(
            n=task_n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, logprobs=2, seed=int(time.time_ns()),
        )
        print(f"inference for {task_name}  (n={task_n})")
        outputs = model.generate(prompts, task_sampling_params)
        batch_scores = []
        batch_formatted = []
        batch_lengths = []
        for k in range(len(outputs)):
            output = outputs[k]
            gt_repeated = [targets[k]] * task_n
            rewards, infos = [], []
            for model_output, gt in zip([o.text for o in output.outputs], gt_repeated):
                info, r = math_reward_fn(model_output, gt, fast=False)
                rewards.append(r)
                infos.append(info)
            rewards = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            batch_scores.append(rewards.mean())

            if infos[0] is not {}:
                batch_formatted.append(np.array([i["formatted"] for i in infos]).sum())

            to_be_saved.append(
                {
                    "task_name": task_name,
                    "prompt": output.prompt,
                    "gt": gt_repeated,
                    "model_output": [o.text for o in output.outputs],
                    "model_output_token_ids": [o.token_ids for o in output.outputs],
                    "reward": [r for r in rewards],
                }
            )

        results[task_name] = float(np.mean(batch_scores)) if batch_scores else 0.0
        avg_lens[task_name] = float(np.mean(batch_lengths)) if batch_lengths else 0.0
        if batch_formatted:
            formatted[task_name] = float(np.mean(batch_formatted))
        max_lens[task_name] = int(np.max(batch_lengths)) if batch_lengths else 0
        per_task_raw[task_name] = [
            {"orig_idx": int(orig_idx[i]),
             "reward": float(batch_scores[i]),
             "lengths": [int(x) for x in batch_lengths[i]]}
            for i in range(len(batch_scores))
        ]

    print(results)
    print("avg:", np.mean(list(results.values())))
    print("avg_lens:", avg_lens)
    print("max_lens:", max_lens)
    print("formatted:", formatted)

    # save per-shard raw rewards for DP merging (always, even shard_total=1)
    import os
    os.makedirs(save_dir, exist_ok=True)
    safe_model = model_name.replace("/", "_").strip("_")
    shard_fn = f"{save_dir}/raw_{safe_model}_shard{shard_idx}of{shard_total}.json"
    json.dump({"per_task_raw": per_task_raw,
               "results_local": results,
               "avg_lens_local": avg_lens,
               "max_lens_local": max_lens,
               "formatted_local": formatted,
               "model_name": model_name,
               "shard": shard,
               "template": template}, open(shard_fn, "w"), indent=2)
    print(f"[shard {shard}] saved raw -> {shard_fn}")

    if save:
        fn = "model_eval_out_" + model_name.replace("/", "_") + str(int(time.time()))
        fn = f"{fn}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json"
        print(f"saving model outputs at {fn}")
        json.dump(
            to_be_saved,
            open(
                fn,
                "w",
            ),
            indent=4,
        )


if __name__ == "__main__":
    fire.Fire(main)
