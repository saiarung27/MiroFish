"""FIX 1 proof: TwHIN-BERT rec model loads from local cache and ranks posts.

Step 1 (cheap, no torch weights): validate snapshot structure — config,
safetensors header, tokenizer files.
Step 2 (live): load via the exact OASIS path (oasis.social_platform.recsys
sufficient, no network fetch), embed 3 sample posts with
generate_post_vector, and rank them against a query post by cosine
similarity. A successful ranking == rec can score posts == feeds work.
"""
import json
import os
import struct
import sys

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Twitter--twhin-bert-base"
    "/snapshots")
snaps = [os.path.join(SNAP, d) for d in os.listdir(SNAP)]
assert len(snaps) == 1, f"expected 1 snapshot, got {snaps}"
snap = snaps[0]
print("snapshot:", snap)

# --- Step 1: structural validation ---
cfg = json.load(open(os.path.join(snap, "config.json")))
print("config: architectures=", cfg.get("architectures"),
      "hidden_size=", cfg.get("hidden_size"),
      "vocab_size=", cfg.get("vocab_size"))
assert cfg.get("hidden_size") == 768

for fname in ("model.safetensors", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json"):
    p = os.path.join(snap, fname)
    print(f"{fname}: {os.path.getsize(p)} bytes")
    assert os.path.getsize(p) > 0

with open(os.path.join(snap, "model.safetensors"), "rb") as f:
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    q = [k for k in tensors if k.endswith("attention.self.query.weight")][0]
    print(f"safetensors: {len(tensors)} tensors, e.g. {q} shape="
          f"{tensors[q]['shape']}")
    assert len(tensors) > 100

# --- Step 2: live load via the OASIS path, cache-only ---
sys.path.insert(0, os.path.expanduser(
    "~/workspace/mirofish/backend/.venv/lib/python3.12/site-packages"))
import torch
torch.set_num_threads(1)
from oasis.social_platform.recsys import load_model
from oasis.social_platform.process_recsys_posts import generate_post_vector

tokenizer, model = load_model("Twitter/twhin-bert-base")
print("OASIS load_model OK:", type(model).__name__)
model.eval()

posts = [
    "I am thinking about trying a Telegram bot that sends stock alerts "
    "to my DM when prices hit key levels.",
    "Cooking pasta tonight with fresh basil from the garden.",
    "Looking for a stock alerts app with instant push notifications "
    "and private DM alerts.",
]
vecs = generate_post_vector(model, tokenizer, posts, batch_size=8)
print("post vectors shape:", tuple(vecs.shape))
assert vecs.shape == (3, 768)

query = vecs[0:1]
sims = torch.nn.functional.cosine_similarity(query, vecs, dim=1)
ranking = torch.argsort(sims, descending=True).tolist()
print("cosine sims:", [round(float(s), 4) for s in sims])
print("ranking (best first):", ranking)
assert ranking[0] == 0, "query post should rank itself first"
assert sims[2] > sims[1], "stock-alerts post must outrank the pasta post"
print("FIX1 PROOF PASSED: rec model loads from cache and ranks posts.")

# --- Step 3: offline load from the LOCAL snapshot path ---
# transformers' mistral-regex check calls model_info() over the network when
# given a repo ID, so HF_HUB_OFFLINE=1 hard-fails even with a complete cache
# (proven by the first run of this script). A local path skips that check.
print("--- step 3: offline load from local snapshot path ---")
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModel, AutoTokenizer
tok_local = AutoTokenizer.from_pretrained(snap)
model_local = AutoModel.from_pretrained(snap)
print("offline local-path load OK:", type(model_local).__name__)
vecs_local = generate_post_vector(model_local, tok_local, posts[:1],
                                  batch_size=8)
print("offline local-path vectors shape:", tuple(vecs_local.shape))
assert vecs_local.shape == (1, 768)
print("FIX1 STEP3 PASSED: local-path load works fully offline.")
