import torch, os, json, time
os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_ALLOW_CPU"] = "1"
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm

device = "cuda"
REPORT = r"C:\Users\redwan\Desktop\swebench_agents_before.json"

dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
dataset = dataset.select(range(50))  # 50 samples for benchmark

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=r"C:\Users\redwan\.cache\huggingface\hub\models--InternScience--Agents-A1-4B",
    max_seq_length=4096, dtype=torch.float16, load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

results = []
for i, s in enumerate(tqdm(dataset, desc="Benchmarking SWE-bench Pro")):
    prompt = s["problem_statement"]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer.tokenizer.encode(text), device=device).unsqueeze(0)
    
    t0 = time.time()
    outputs = model.generate(input_ids, max_new_tokens=1024, temperature=0.6, top_p=0.95)
    gen_time = time.time() - t0
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = generated.split("assistant", 1)[-1].strip() if "assistant" in generated else generated
    
    expected_patch = s["patch"]
    repo = s["repo"]
    
    results.append({
        "index": i,
        "repo": repo,
        "instance_id": s["instance_id"][:50],
        "problem_length": len(s["problem_statement"]),
        "expected_patch_length": len(expected_patch),
        "generated_length": len(generated),
        "execution_time": round(gen_time, 2),
        "generated_output": generated[:500],
        "expected_patch_start": expected_patch[:200],
    })

report = {
    "model": "InternScience/Agents-A1-4B (before training)",
    "dataset": "ScaleAI/SWE-bench_Pro (first 50 samples)",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total": len(results),
    "results": results,
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nBenchmark complete. Report saved to: {REPORT}")
print(f"Samples tested: {len(results)}")
