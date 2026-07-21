import os, torch, json, time, sys
os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_ALLOW_CPU"] = "1"

from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm

device = "cuda"
REPORT = r"C:\Users\redwan\Desktop\swebench_agents_before.json"

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=r"C:\Users\redwan\.cache\huggingface\hub\models--InternScience--Agents-A1-4B",
    max_seq_length=4096, dtype=torch.float16, load_in_4bit=False,
)
print("Model loaded. Setting up inference...")
FastLanguageModel.for_inference(model)
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
ds = ds.select(range(10))
print(f"Dataset: {len(ds)} samples")

results = []
for i, s in enumerate(tqdm(ds, desc="Benchmarking")):
    prompt = s["problem_statement"]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer.tokenizer.encode(text), device=device).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=1024, temperature=0.6, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
    gen_time = time.time() - t0

    # Only decode new tokens (skip prompt)
    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    results.append({
        "index": i,
        "repo": s["repo"],
        "instance_id": str(s["instance_id"])[:50],
        "problem_length": len(s["problem_statement"]),
        "expected_patch_length": len(s["patch"]),
        "generated_length": len(response),
        "execution_time": round(gen_time, 2),
        "generated_output_start": response[:500],
        "expected_patch_start": s["patch"][:200],
    })
    print(f"\nSample {i} done in {gen_time:.1f}s. Output: {len(response)} chars")
    print(f"Response preview: {response[:200]}")

report = {
    "model": "InternScience/Agents-A1-4B (before training)",
    "dataset": "ScaleAI/SWE-bench_Pro (first 50 samples)",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total": len(results),
    "results": results,
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nDone. Report saved to: {REPORT}")
