from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch, seaborn as sns, matplotlib.pyplot as plt, einops

MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"

cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
cfg.output_router_logits = True  # <-- the key switch
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, config=cfg, torch_dtype="auto", device_map="auto", trust_remote_code=True
).eval()

print("hello", cfg)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

@torch.no_grad()
def run_and_collect(texts: list[str]):
    batch = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model(**batch, use_cache=False, return_dict=True)

    # argmax → chosen expert id for every token
    chosen = [logits.argmax(-1).cpu() for logits in out.router_logits]
    return chosen, batch["input_ids"].cpu()

sent = "Mixture-of-experts models dynamically route tokens to specialised subnetworks."
chosen, ids = run_and_collect([sent])

layers, seq_len = len(chosen), chosen[0].shape[1]
heat = torch.stack(chosen).squeeze(1)  # (layers, seq_len)

plt.figure(figsize=(seq_len / 1.2, layers / 1.3))
sns.heatmap(
    heat,
    cmap="viridis",
    xticklabels=[tok.decode(t) for t in ids[0]],
    yticklabels=[f"L{i}" for i in range(layers)],
    cbar=True,
)
plt.title("DeepSeek-MoE — chosen expert per token per layer")
plt.show()