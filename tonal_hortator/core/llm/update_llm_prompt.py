import json


def generate_prompt_from_jsonl(jsonl_path: str, output_prompt_path: str) -> None:
    with open(jsonl_path) as f:
        examples = [json.loads(line) for line in f if '"label": 1' in line]
    with open(output_prompt_path, "w") as f:
        for ex in examples:
            f.write(f"User: {ex['input']}\n")
            f.write(f"LLM: {json.dumps(ex['system_parsed'])}\n\n")
    print(f"Wrote prompt to {output_prompt_path}")


if __name__ == "__main__":
    generate_prompt_from_jsonl("playlist_training_data.jsonl", "llm_prompt.txt")
