# OTel 2.0 31B — Local MLX Setup

What I ran to get the GSMA Open Telco leaderboard model working offline on Apple silicon.

This is a record of one working path, **not a recommended one**. Versions and choices below
are what happened to work on this machine. If you know a better way, or a version that
handles any of this more cleanly, I'd like to hear it — open an issue.

Machine: Apple M5 Max, 48 GB unified memory, macOS.
Result: 18.3 GB peak, ~28 tok/s generation.

---

## Provenance

```
Model:      farbodtavakkoli/OTel-2.0-LLM-31B-IT
Commit SHA: c5533461bc117759cd998bb3919d21e803425352
License:    Apache-2.0
Params:     32,106,632,252
Base:       Gemma 4 31B-IT
Quant:      4-bit MLX, 4.501 bits/weight
```

Weights are updated weekly upstream, so the SHA matters.

```bash
python3 -c "
from huggingface_hub import HfApi
print(HfApi().model_info('farbodtavakkoli/OTel-2.0-LLM-31B-IT').sha)
"
```

---

## 1. Environment

```bash
mkdir -p ~/telco-eval && cd ~/telco-eval
uv venv --python 3.12
source .venv/bin/activate
uv pip install git+https://github.com/ml-explore/mlx-lm huggingface_hub
hf auth login
```

I installed mlx-lm from git rather than PyPI because the release at the time did not
include `gemma4_text`.

```bash
python3 -c "import mlx.core as mx; print(mx.default_device())"
python3 -c "from mlx_lm.models import gemma4_text; print('gemma4 ok')"
```

Got `Device(gpu, 0)` and `gemma4 ok`.

---

## 2. Weight file count

```bash
python3 -c "
from huggingface_hub import list_repo_files
fs = list_repo_files('farbodtavakkoli/OTel-2.0-LLM-31B-IT')
print(sum(f.endswith('.safetensors') for f in fs), 'weight files')
"
```

Got 14.

---

## 3. Strip incompatible tensors

The checkpoint ships six tensors that mlx-lm's `gemma4_text` does not expect, so loading
fails until they are removed.

I confirmed `lm_head.weight` is tied before dropping it:

```bash
python3 -c "
from huggingface_hub import hf_hub_download
import json
c = json.load(open(hf_hub_download('farbodtavakkoli/OTel-2.0-LLM-31B-IT','config.json')))
print('tie_word_embeddings =', c.get('tie_word_embeddings'))
"
```

I did **not** verify the other five — `embed_scale` and the four rotary `inv_freq`
buffers. They should be recomputed from config at load time, but I have not checked
whether this checkpoint ships default values for them. This is the step I am least
sure about.

```bash
cat > ~/telco-eval/strip.py << 'EOF'
import json, os, shutil, pathlib
import mlx.core as mx
from huggingface_hub import snapshot_download

DROP = {
 "lm_head.weight",
 "model.embed_tokens.embed_scale",
 "model.rotary_emb.full_attention_inv_freq",
 "model.rotary_emb.full_attention_original_inv_freq",
 "model.rotary_emb.sliding_attention_inv_freq",
 "model.rotary_emb.sliding_attention_original_inv_freq",
}

src = pathlib.Path(snapshot_download("farbodtavakkoli/OTel-2.0-LLM-31B-IT"))
dst = pathlib.Path.home() / "telco-eval" / "otel-31b-clean"
dst.mkdir(parents=True, exist_ok=True)

for f in src.iterdir():
    if f.suffix == ".safetensors" or f.name.endswith(".index.json"):
        continue
    if f.is_file():
        shutil.copy2(f, dst / f.name)

idx = json.load(open(src / "model.safetensors.index.json"))
wm = {k: v for k, v in idx["weight_map"].items() if k not in DROP}

total = 0
for shard in sorted(set(idx["weight_map"].values())):
    w = mx.load(str(src / shard))
    keep = {k: v for k, v in w.items() if k not in DROP}
    if len(w) != len(keep):
        print(f"{shard}: dropped {len(w)-len(keep)}")
        mx.save_safetensors(str(dst / shard), keep, metadata={"format": "pt"})
    else:
        os.link(src / shard, dst / shard)
    total += sum(v.nbytes for v in keep.values())
    del w, keep

json.dump({"metadata": {"total_size": total}, "weight_map": wm},
          open(dst / "model.safetensors.index.json", "w"), indent=2)
print("clean model at:", dst)
EOF

python3 ~/telco-eval/strip.py
```

Unmodified shards are hardlinked rather than copied, so this costs a few GB instead of
another 64.

Output: 5 dropped from shard 1, 1 dropped from shard 2.

---

## 4. Quantize

```bash
mlx_lm.convert --hf-path ~/telco-eval/otel-31b-clean -q --q-bits 4 --mlx-path ~/telco-eval/otel-31b-q4
```

Output: `Quantized model with 4.501 bits per weight.`

---

## 5. Clean up

```bash
hf cache rm model/farbodtavakkoli/OTel-2.0-LLM-31B-IT
rm -rf ~/telco-eval/otel-31b-clean
```

Reclaimed ~70 GB. Cache subcommand names differ across `huggingface_hub` versions —
`hf cache --help` if this errors.

---

## 6. Run

```bash
cd ~/telco-eval && source .venv/bin/activate
mlx_lm.chat --model ~/telco-eval/otel-31b-q4 --max-tokens 1000 --temp 0.0 --seed 0
```

`q` exit · `r` reset · `h` help

Two things I hit here. Thinking mode is on by default — the chat template injects a
system turn containing `<|think|>`, and the raw reasoning trace prints to the terminal,
which reads as broken output. It also consumes the token budget, so 1000 is too low
unless thinking is off. And `mlx_lm.chat` keeps history within a session, so `r` between
unrelated questions.

Gemma 4 turn delimiters are `<|turn>` and `<turn|>`, not `<start_of_turn>` /
`<end_of_turn>`. They look wrong but are correct:

```bash
python3 -c "
from mlx_lm import load
import pathlib
t = load(str(pathlib.Path.home()/'telco-eval/otel-31b-q4'))[1]
print('<turn|> ->', t.encode('<turn|>', add_special_tokens=False))
print('eos:', t.eos_token, t.eos_token_id)
"
```

Got `<turn|> -> [106]` and `eos: <turn|> 106`.

---

## Resuming after restart

```bash
cd ~/telco-eval
source .venv/bin/activate
mlx_lm.chat --model ~/telco-eval/otel-31b-q4
```

The HF login persists in `~/.cache/huggingface`, not the venv.

---

## Links

- Model — https://huggingface.co/farbodtavakkoli/OTel-2.0-LLM-31B-IT
- Leaderboard — https://huggingface.co/spaces/GSMA/open-telco-leaderboard
- Benchmark data — https://huggingface.co/datasets/GSMA/ot-lite
- Remote conversion fallback — https://huggingface.co/spaces/mlx-community/mlx-my-repo
