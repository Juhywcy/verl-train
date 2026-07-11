import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def normalize_answer(text: str) -> str:
    s = text.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.strip().strip(".")
    return s


def extract_final_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    last_num = re.findall(r"-?\d+(?:\.\d+)?", text)
    if last_num:
        return last_num[-1].strip()
    return None


def build_verifier_prompt(question: str, response: str) -> str:
    return (
        "You are a verifier.\n"
        "Given a math question and a model response, identify reasoning steps in the response.\n"
        "For each step, decide if it is correct based only on the question context.\n"
        "Mark a step incorrect if it has a math error, a logical error, or is irrelevant/contradictory.\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\"steps\":[{\"text\":string,\"correct\":true|false}]}\n\n"
        f"Question: {question}\n\n"
        f"Response: {response}\n"
    )


def verify_steps(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> dict:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    json_block = _extract_json_block(raw)
    if not json_block:
        return {"steps": []}
    try:
        return json.loads(json_block)
    except json.JSONDecodeError:
        return {"steps": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to jsonl file")
    parser.add_argument("--model", required=True, help="Verifier model path")
    parser.add_argument("--output", default=None, help="Output json path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(input_path.suffix + ".hallucination_summary.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    total = 0
    correct_total = 0
    hallucination_total = 0
    correct_with_hallucination = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            doc = obj.get("doc", {})
            question = doc.get("question", "")
            target = obj.get("target", "")

            resps = obj.get("resps", [])
            response = ""
            if resps and isinstance(resps, list) and resps[0]:
                response = resps[0][0]

            extracted = extract_final_answer(response) or ""
            is_correct = normalize_answer(extracted) == normalize_answer(str(target))
            if is_correct:
                correct_total += 1

            prompt = build_verifier_prompt(question, response)
            verdict = verify_steps(model, tokenizer, prompt)
            steps = verdict.get("steps", []) if isinstance(verdict, dict) else []
            has_hallucination = any(not s.get("correct", False) for s in steps)
            if has_hallucination:
                hallucination_total += 1
                if is_correct:
                    correct_with_hallucination += 1

    summary = {
        "total": total,
        "correct_total": correct_total,
        "hallucination_total": hallucination_total,
        "correct_with_hallucination": correct_with_hallucination,
        "correct_with_hallucination_rate": (
            correct_with_hallucination / correct_total if correct_total else 0.0
        ),
        "hallucination_rate": (hallucination_total / total if total else 0.0),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
