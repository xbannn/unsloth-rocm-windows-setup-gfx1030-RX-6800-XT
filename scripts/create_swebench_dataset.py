import json, os, re
from datasets import load_dataset
from tqdm import tqdm

OUTPUT = r"C:\Users\redwan\Desktop\swebench_pro_formatted.jsonl"

def parse_diff_for_files(patch):
    """Extract file paths from a git diff."""
    files = re.findall(r'^diff --git a/(.*?) b/', patch, re.MULTILINE)
    return list(set(files))

def parse_diff_for_summary(patch):
    """Generate a brief summary of what the diff changes."""
    files = parse_diff_for_files(patch)
    added = len(re.findall(r'^\+', patch, re.MULTILINE))
    removed = len(re.findall(r'^\-', patch, re.MULTILINE))

    # Identify function/class names being changed
    funcs = re.findall(r'^\+.*def (\w+)\(', patch, re.MULTILINE) or \
            re.findall(r'^\+.*function (\w+)\(', patch, re.MULTILINE) or \
            re.findall(r'^\+.*func (\w+)\(', patch, re.MULTILINE)
    classes = re.findall(r'^\+.*class (\w+)', patch, re.MULTILINE)

    parts = []
    if files:
        parts.append(f"Modifies {len(files)} file(s): {', '.join(files[:5])}")
        if len(files) > 5:
            parts[-1] += f" and {len(files)-5} more"
    if classes:
        parts.append(f"affecting class(es): {', '.join(classes[:3])}")
    if funcs:
        parts.append(f"function(s): {', '.join(funcs[:3])}")
    parts.append(f"({added} additions, {removed} deletions)")
    return " ".join(parts)

print("Loading SWE-bench_Pro...")
ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
print(f"Total: {len(ds)} samples")

with open(OUTPUT, "w", encoding="utf-8") as f:
    for i, s in enumerate(tqdm(ds, desc="Formatting")):
        problem = s["problem_statement"]
        patch = s["patch"]
        repo = s["repo"]
        lang = s["repo_language"]

        summary = parse_diff_for_summary(patch)
        reasoning = f"The bug is in {repo} ({lang}). {summary}."

        user_msg = f"Fix this bug:\n\n{problem}"
        asst_msg = f"<think>\n{reasoning}\n</think>\n\n{patch}"

        row = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(ds)} samples to {OUTPUT}")
print(f"File size: {os.path.getsize(OUTPUT)/1e6:.1f} MB")
