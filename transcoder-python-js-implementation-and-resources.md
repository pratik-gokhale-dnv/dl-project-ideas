# Python <-> JavaScript Code Translation: 3-Week Implementation Plan and Resources

Team: 3 people | Duration: about 3 weeks | Hardware: MacBook Pro M4 (MPS/MLX), Colab T4 as fallback

> Link status: only `nuprl/MultiPL-E` was confirmed through the GitHub API while writing this. All other links come from memory of well-known projects. Click through and verify each one (existence, license, language coverage) before relying on it. Training times are estimates. Benchmark your own machine in week 1.

See also: [transcoder-python-js-plan.md](transcoder-python-js-plan.md) (original semester plan).

---

## 1. Scope (cut for 3 weeks)

| Keep | Drop or make stretch |
|---|---|
| Function-level translation, Py->JS and JS->Py | Back-translation (stretch only) |
| One fine-tuned small model | Full-size QLoRA on 7B |
| Few-shot baseline | Go or class-level translation |
| Execution-based eval on HumanEval-X / MultiPL-E | RL, repair loops |
| Error taxonomy of about 50 failures | |

Research questions: (1) does fine-tuning beat few-shot prompting, (2) which errors dominate, (3) LoRA vs full fine-tune of a small model.

## 2. Models

PyTorch's MPS backend does the training (`torch.device("mps")`).

| Role | Model | Size | Notes |
|---|---|---|---|
| Primary (full fine-tune) | `Salesforce/codet5p-220m` | 220M | Encoder-decoder, suits translation. Fits in 16GB. |
| Safe drop-in | `Salesforce/codet5-base` | 220M | Use if `trust_remote_code` or loading issues appear with CodeT5+. |
| Primary alt (LoRA) | `Qwen/Qwen2.5-Coder-0.5B-Instruct`, `1.5B-Instruct` | 0.5-1.5B | Decoder-only, strong for its size. |
| Baseline (no training) | `Qwen2.5-Coder-1.5B/3B-Instruct` with few-shot prompts | 1.5-3B | Run via `mlx-lm`, `llama.cpp` or Ollama. |
| Optional larger | `bigcode/starcoder2-3b`, `deepseek-ai/deepseek-coder-1.3b-instruct` | 1.3-3B | Extra baselines if time allows. |

Notes:
- `bitsandbytes` (QLoRA) is not supported on Mac. Use plain LoRA via `peft`, or use MLX.
- MLX is the fastest route on Apple Silicon.

### Model links
- CodeT5+ 220M: https://huggingface.co/Salesforce/codet5p-220m
- CodeT5+ 770M: https://huggingface.co/Salesforce/codet5p-770m
- CodeT5 base: https://huggingface.co/Salesforce/codet5-base
- CodeT5 small: https://huggingface.co/Salesforce/codet5-small
- CodeT5+ GitHub: https://github.com/salesforce/CodeT5
- CodeBERT: https://huggingface.co/microsoft/codebert-base
- GraphCodeBERT: https://huggingface.co/microsoft/graphcodebert-base
- UniXcoder: https://huggingface.co/microsoft/unixcoder-base
- Qwen2.5-Coder collection: https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f
- Qwen2.5-Coder 0.5B Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
- Qwen2.5-Coder 1.5B Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- Qwen2.5-Coder 3B Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct
- Qwen2.5-Coder GitHub: https://github.com/QwenLM/Qwen2.5-Coder
- StarCoder2-3B: https://huggingface.co/bigcode/starcoder2-3b
- StarCoder2 GitHub: https://github.com/bigcode-project/starcoder2
- DeepSeek-Coder 1.3B Instruct: https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct
- Code Llama 7B (too large for comfortable Mac training): https://huggingface.co/codellama/CodeLlama-7b-hf
- MLX community pre-converted models: https://huggingface.co/mlx-community

## 3. Dataset

### 3.1 Target format (JSONL)
```json
{"id": "cn_p00001_17", "problem_id": "p00001", "src_lang": "python", "tgt_lang": "javascript",
 "src": "def add(a,b):\n    return a+b", "tgt": "function add(a,b){ return a+b; }",
 "source_dataset": "codenet", "tests": null}
```
Swap src and tgt to get the other direction. Prompt for CodeT5: `Translate Python to JavaScript:\n{src}`. Use a chat template for Qwen.

### 3.2 Sources (priority order)
1. XLCoST: snippet and program-level parallel data. Check that Python and JavaScript are both present.
2. CodeNet: pair solutions by problem ID, then keep pairs whose outputs match.
3. Rosetta Code: small but cleanly parallel.
4. HumanEval-X / MultiPL-E: TEST SET ONLY. Never train on them.
5. Optional: The Stack v2 small subsets for back-translation.

Realistic target: 10k-30k pairs after filtering.

### 3.3 Preparation pipeline (Person A)
Libraries: `datasets`, `tree-sitter`, `tree-sitter-python`, `tree-sitter-javascript`, `datasketch`, `pandas`.
1. Download with `datasets.load_dataset(...)`.
2. Extract functions with tree-sitter (`function_definition` in Python; `function_declaration` and arrow functions in JS).
3. Filter: parses without errors (`root_node.has_error == False`), 512 tokens or fewer, standard library only (`sys.stdlib_module_names`, no non-built-in `require()`), no I/O or network.
4. Deduplicate: exact hash, then MinHash LSH (Jaccard about 0.8).
5. Split by problem ID (90/5/5). Near-dedup against HumanEval prompts to avoid leakage.
6. Optional verification: run both versions on the same inputs and keep matching pairs.

Outputs: `data/train.jsonl`, `val.jsonl`, `test.jsonl`, `stats.md`.

### Dataset and benchmark links
- XLCoST GitHub: https://github.com/reddy-lab-code-research/XLCoST
- XLCoST paper: https://arxiv.org/abs/2206.08474
- XLCoST on HF (check): https://huggingface.co/datasets/codeparrot/xlcost-text-to-code
- Project CodeNet GitHub: https://github.com/IBM/Project_CodeNet
- CodeNet paper: https://arxiv.org/abs/2105.12655
- Rosetta Code (site): https://rosettacode.org
- Rosetta Code on HF (check): https://huggingface.co/datasets/christopher/rosetta-code
- HumanEval-X (CodeGeeX) on HF: https://huggingface.co/datasets/THUDM/humaneval-x
- CodeGeeX GitHub: https://github.com/THUDM/CodeGeeX
- MultiPL-E GitHub (verified): https://github.com/nuprl/MultiPL-E
- MultiPL-E on HF: https://huggingface.co/datasets/nuprl/MultiPL-E
- MultiPL-E paper: https://arxiv.org/abs/2208.08227
- OpenAI HumanEval: https://github.com/openai/human-eval
- AVATAR (Java<->Python): https://github.com/wasiahmad/AVATAR
- CodeXGLUE (code-to-code): https://github.com/microsoft/CodeXGLUE
- CodeTransOcean (multilingual translation benchmark): https://github.com/WeixiangYAN/CodeTransOcean
- The Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2
- The Stack v1: https://huggingface.co/datasets/bigcode/the-stack
- TransCoder GeeksforGeeks test data: inside https://github.com/facebookresearch/TransCoder
- TransCoder-test-X (multilingual): https://github.com/facebookresearch/CodeGen
- CoST / GeeksforGeeks parallel data: https://github.com/reddy-lab-code-research/XLCoST

## 4. Training (Person B)

### 4.1 CodeT5+ 220M (full fine-tune, HF Trainer)
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
tok = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
args = Seq2SeqTrainingArguments(
  output_dir="out/codet5", per_device_train_batch_size=8, gradient_accumulation_steps=4,
  learning_rate=5e-5, num_train_epochs=3, warmup_ratio=0.05, weight_decay=0.01,
  eval_strategy="epoch", save_strategy="epoch", predict_with_generate=True,
  generation_max_length=512, report_to="wandb")
```
- Source and target max length of 512. Mask pad tokens in labels with `-100`.
- Estimated time: about 20k pairs x 3 epochs takes roughly 3-6 hours on an M4 Pro, 8+ hours on a base M4. Reduce max length to 384 or use 10k pairs if slow.
- If MPS raises op errors: `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### 4.2 Qwen2.5-Coder LoRA (MLX)
```bash
pip install mlx-lm
mlx_lm.lora --model Qwen/Qwen2.5-Coder-1.5B-Instruct --train \
  --data ./data_mlx --batch-size 4 --iters 1500 --num-layers 16 --learning-rate 1e-4
mlx_lm.generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter-path adapters --prompt "..."
```
- Data: `data_mlx/{train,valid,test}.jsonl` with `{"messages":[{"role":"user",...},{"role":"assistant",...}]}`.
- Estimated time: 0.5B takes about 30-60 min for 1,500 iterations. 1.5B takes about 1.5-3 hours.
- HF alternative: `peft.LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"])` with `trl.SFTTrainer`. It is slower on MPS.

### 4.3 Few-shot baseline
3-5 fixed example pairs in the prompt, greedy decoding. Run via `mlx_lm` or Ollama (`ollama run qwen2.5-coder:3b`). Also evaluate the same models without fine-tuning (base vs fine-tuned is the cleanest comparison).

### Tooling links
- Hugging Face Transformers: https://github.com/huggingface/transformers
- Transformers docs: https://huggingface.co/docs/transformers
- Seq2SeqTrainer docs: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer
- Datasets: https://github.com/huggingface/datasets
- PEFT (LoRA): https://github.com/huggingface/peft
- TRL (SFTTrainer): https://github.com/huggingface/trl
- Accelerate: https://github.com/huggingface/accelerate
- PyTorch MPS notes: https://pytorch.org/docs/stable/notes/mps.html
- MLX: https://github.com/ml-explore/mlx
- mlx-lm: https://github.com/ml-explore/mlx-lm
- mlx-lm LoRA guide: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Ollama: https://ollama.com
- Ollama Qwen2.5-Coder: https://ollama.com/library/qwen2.5-coder
- Weights & Biases: https://wandb.ai
- tree-sitter: https://github.com/tree-sitter/tree-sitter
- py-tree-sitter: https://github.com/tree-sitter/py-tree-sitter
- tree-sitter-python: https://github.com/tree-sitter/tree-sitter-python
- tree-sitter-javascript: https://github.com/tree-sitter/tree-sitter-javascript
- datasketch (MinHash LSH): https://github.com/ekzhu/datasketch
- Docker Desktop: https://www.docker.com/products/docker-desktop/
- Node.js: https://nodejs.org

## 5. Evaluation (Person C)

### 5.1 Primary: execution-based
1. Test sets: HumanEval-X (Python and JS, 164 problems with tests), MultiPL-E `humaneval-js`, plus 50-100 held-out verified problems from your own test split.
2. Py->JS: translate the Python canonical solution, plug it into the JS test file, run with Node. JS->Py: the reverse.
3. Sandbox: `subprocess.run([...], timeout=10, capture_output=True)` inside Docker with `--network none --memory 512m --cpus 1`. Lighter option: temp dir plus `resource` limits.
4. Metrics:
   - Computational accuracy (pass@1, greedy). This is the headline number.
   - pass@k (k=5 or 10, temperature 0.8) with the unbiased estimator from the Codex paper (optional).
   - Syntax validity rate via tree-sitter.

### 5.2 Secondary: text metrics (weak, report but do not grade on them)
- BLEU: `sacrebleu`
- CodeBLEU: `codebleu` package, `calc_codebleu(refs, hyps, lang="javascript")`. Verify Python and JS support.
- Exact match.

### 5.3 Error analysis
Sample about 50 failures per model. Label as: syntax error, wrong API mapping, off-by-one or negative indexing, integer division, type coercion/equality, string/array method mismatch, sort default, missing import or helper, other.

### 5.4 Results table
| Model | Direction | Syntax valid % | pass@1 | CodeBLEU | BLEU |
|---|---|---|---|---|---|
| Qwen 1.5B few-shot | Py->JS | | | | |
| Qwen 1.5B + LoRA | Py->JS | | | | |
| CodeT5+ 220M FT | Py->JS | | | | |
| (same three) | JS->Py | | | | |

### Evaluation links
- CodeBLEU paper: https://arxiv.org/abs/2009.10297
- CodeBLEU implementation: https://github.com/k4black/codebleu
- CodeBLEU (original, Microsoft): https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU
- sacrebleu: https://github.com/mjpost/sacrebleu
- Codex paper (pass@k): https://arxiv.org/abs/2107.03374
- bigcode-evaluation-harness (supports HumanEval-X and MultiPL-E): https://github.com/bigcode-project/bigcode-evaluation-harness
- Hugging Face `evaluate` code_eval metric: https://huggingface.co/spaces/evaluate-metric/code_eval
- Hugging Face `evaluate`: https://github.com/huggingface/evaluate
- EvalPlus (stronger HumanEval tests): https://github.com/evalplus/evalplus

## 6. Three-week timeline

| Week | Person A (Data) | Person B (Modeling) | Person C (Evaluation) |
|---|---|---|---|
| 1 | Download XLCoST, CodeNet, Rosetta. Build tree-sitter extraction. First small `train.jsonl` by Wed. | Set up env, benchmark MPS speed, smoke test CodeT5 on 500 pairs. Set up MLX. | Build the Node + Python harness. Load HumanEval-X. Verify canonical solutions pass. |
| 2 | Finish filtering, dedup, splits, stats. Hand off data mid-week. Optional output-match verification. | Train CodeT5+ both directions. Train Qwen LoRA. Log to W&B. | Few-shot baselines. Wire in BLEU, CodeBLEU, syntax checks. Test the harness on first checkpoints. |
| 3 | Help with error labelling. Write the data section. | Final runs, optional ablation, produce generations. | Full eval, error taxonomy, plots. All: report, slides, cleanup, README. |

Hard rule: data freezes by mid-week 2. If CodeNet pairing runs late, use XLCoST and Rosetta only.

## 7. Repo layout
```
├── data/ (raw/, processed/)
├── src/data/ (extract.py, filter.py, dedup.py, split.py)
├── src/train/ (train_codet5.py, prepare_mlx.py)
├── src/eval/ (run_tests.py, sandbox.py, metrics.py, error_analysis.ipynb)
├── configs/  scripts/  README.md  requirements.txt
```
Libraries: `torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `mlx-lm`, `tree-sitter`, `datasketch`, `sacrebleu`, `codebleu`, `wandb`. Also Node.js LTS and Python 3.10+.

## 8. Risks
- MPS instability or slowness: keep a Colab T4 fallback, use sequence length 384.
- Little clean parallel data: lean on XLCoST and the output-match filter, accept a smaller dataset.
- HumanEval contamination: Qwen and CodeT5+ may have seen these problems. State this as a limitation.
- Harness bugs: test on canonical solutions in week 1.

## 9. Related work: papers and similar projects

### Papers
- TransCoder (Roziere et al., 2020): https://arxiv.org/abs/2006.03511
- TransCoder-IR (Szafraniec et al.): https://arxiv.org/abs/2207.03578
- DOBF (Roziere et al.): https://arxiv.org/abs/2102.07492
- CodeT5 (Wang et al., 2021): https://arxiv.org/abs/2109.00859
- CodeT5+ (Wang et al., 2023): https://arxiv.org/abs/2305.07922
- CodeBERT: https://arxiv.org/abs/2002.08155
- GraphCodeBERT: https://arxiv.org/abs/2009.08366
- CodeXGLUE: https://arxiv.org/abs/2102.04664
- AVATAR: https://arxiv.org/abs/2108.11590
- CodeGeeX / HumanEval-X: https://arxiv.org/abs/2303.17568
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- StarCoder: https://arxiv.org/abs/2305.06161
- StarCoder 2 and The Stack v2: https://arxiv.org/abs/2402.19173
- Qwen2.5-Coder technical report: https://arxiv.org/abs/2409.12186
- Lost in Translation (study of bugs in LLM code translation): https://arxiv.org/abs/2308.03109
- CodeTransOcean: https://arxiv.org/abs/2310.04951

### Repos doing similar things
- facebookresearch/TransCoder (original unsupervised translator): https://github.com/facebookresearch/TransCoder
- facebookresearch/CodeGen (TransCoder successor code): https://github.com/facebookresearch/CodeGen
- salesforce/CodeT5 (includes code translation fine-tuning scripts): https://github.com/salesforce/CodeT5
- microsoft/CodeXGLUE (code-to-code translation baselines, Java<->C#): https://github.com/microsoft/CodeXGLUE
- reddy-lab-code-research/XLCoST (multilingual parallel data and baselines): https://github.com/reddy-lab-code-research/XLCoST
- wasiahmad/AVATAR: https://github.com/wasiahmad/AVATAR
- IBM/Project_CodeNet: https://github.com/IBM/Project_CodeNet
- nuprl/MultiPL-E (verified): https://github.com/nuprl/MultiPL-E
- THUDM/CodeGeeX (includes translation task and HumanEval-X): https://github.com/THUDM/CodeGeeX
- WeixiangYAN/CodeTransOcean: https://github.com/WeixiangYAN/CodeTransOcean
- bigcode-project/bigcode-evaluation-harness: https://github.com/bigcode-project/bigcode-evaluation-harness
- Lost in Translation replication package: https://github.com/Intelligent-CAT-Lab/PLTranslationEmpirical
- facebookresearch/TransCoder-IR: https://github.com/facebookresearch/CodeGen/tree/main/codegen_sources/IR_tools

Tip: to find more, search GitHub for topics such as `code-translation`, `program-translation`, `transcoder`, `code-to-code`: https://github.com/topics/code-translation

## 10. Reading order for week 1
1. TransCoder abstract and Section 3 (concepts only).
2. CodeT5 / CodeT5+ model cards on Hugging Face.
3. XLCoST dataset README.
4. MultiPL-E README (to learn how its harness runs JS tests).
5. mlx-lm LORA.md.
