"""Label every v2 channel message with Bespoke Nimble 9B (runs in its own venv; Qwen family, not Gemma)."""
import sys, json, hashlib
from pathlib import Path
REPO, FULL_DIR, CACHE = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
import pandas as pd
from huggingface_hub import snapshot_download

# --- model preparation: verbatim logic from the Nimble README quickstart ---
repo = "bespokelabs/Bespoke-Nimble-9B"
snapshot = Path(snapshot_download(repo, cache_dir=f"{CACHE}/hub"))
contract_file = snapshot / "schema_config.json"
contract = json.loads(contract_file.read_text()) if contract_file.exists() else {}
if contract:
    from transformers import AutoTokenizer
    from nimble.training.candidate_schema import validate_contract
    validate_contract(contract, AutoTokenizer.from_pretrained(snapshot))
model_path = snapshot
if (snapshot / "adapter_config.json").exists():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    model_path = Path(CACHE) / ("nimble-9b-" + snapshot.name)
    if not (model_path / "READY.json").exists():
        base = Qwen3_5ForConditionalGeneration.from_pretrained(contract["model"], revision=contract["revision"],
                                                               dtype=torch.bfloat16, device_map="cpu")
        merged = PeftModel.from_pretrained(base, snapshot).merge_and_unload(safe_merge=True)
        merged.save_pretrained(model_path)
        (model_path / "schema_config.json").write_text(json.dumps(contract, indent=2))
        AutoTokenizer.from_pretrained(snapshot).save_pretrained(model_path)
        (model_path / "READY.json").write_text(json.dumps({"model": repo, "revision": snapshot.name,
            "adapter_sha256": hashlib.sha256((snapshot / "adapter_model.safetensors").read_bytes()).hexdigest()}, indent=2))
config = {"model_path": str(Path(model_path).resolve()), "model_id": repo, "revision": snapshot.name,
          "max_input_tokens": contract.get("max_length", 2048)}
print("model ready:", config)

from nimble.scoring.cuda_scorer import CudaCandidateScorer
scorer = CudaCandidateScorer(**config)
print("temperature:", scorer.temperature)

SCHEMA = {
  "label": {"type": "enum", "description": "What the message from one farm to its only competitor does.",
            "choices": ["PROPOSE", "ACCEPT", "REFUSE", "SIGNAL", "PRESSURE", "SMALLTALK"],
            "choice_descriptions": {
              "PROPOSE": "Proposes that both farms limit, split or set their supply, sales or prices together.",
              "ACCEPT": "Agrees to or confirms a joint plan about supply, sales or prices.",
              "REFUSE": "Rejects coordinating or says it will act on its own.",
              "SIGNAL": "States its own plans, production or intentions without proposing joint action.",
              "PRESSURE": "Tries to change the competitor's crop or supply choices for the sender's benefit, without a joint plan.",
              "SMALLTALK": "Greetings, wishes or remarks with no strategic content."}},
  "joint_restraint": {"type": "boolean",
            "description": "Whether the message proposes, confirms or endorses BOTH farms limiting, splitting or coordinating supply, sales or prices."},
}
M = pd.read_csv(f"{FULL_DIR}/messages_v2.csv")
rows = []
for i, r in M.iterrows():
    res = scorer.score(f"Message from one farm to its only competitor: {r.text}", SCHEMA)
    rows.append({"idx": i, "nimble_label": res["output"]["label"], "nimble_restraint": res["output"]["joint_restraint"],
                 "p_label": json.dumps(res["fields"]["label"]["scores"]), "p_restraint_true": res["fields"]["joint_restraint"]["scores"]["true"]})
    if i % 100 == 0: print(i, r.text[:70], "->", res["output"])
out = M.join(pd.DataFrame(rows).set_index("idx"))
out.to_csv(f"{FULL_DIR}/nimble_labels.csv", index=False)
print(out.groupby("arm").nimble_label.value_counts(normalize=True).round(2))
print(out.groupby("arm").nimble_restraint.mean().round(2))
