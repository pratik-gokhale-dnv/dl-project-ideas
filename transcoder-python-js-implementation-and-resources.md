# Python <-> JavaScript Code Translation: 3-Week Implementation Plan and Resources

Team: 3 people | Duration: about 3 weeks | Hardware: MacBook Pro M4 (MPS/MLX), Colab T4 as fallback

> Link status: only `nuprl/MultiPL-E` was confirmed through the GitHub API while writing this. All other links come from memory of well-known projects. Click through and verify each one (existence, license, language coverage) before relying on it. Training times are estimates. Benchmark your own machine in week 1.

See also: [transcoder-python-js-plan.md](transcoder-python-js-plan.md) (original semester plan) and [project-proposal-python-js-code-translation.md](project-proposal-python-js-code-translation.md) (sample course proposal).

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
| 1 | Download XLCoST, CodeNet, Rosetta. Build tree-sitter extraction. First small `train.jsonl` by Wed. | Set up env, benchmark MPS speed, smoke test CodeT5 on 500 pairs. Set up MLX. | Build the Node + Python execution harness. Load HumanEval-X. Verify canonical solutions pass. |
| 2 | Finish filtering, dedup, splits, stats. Hand off data mid-week. Optional output-match verification. | Train CodeT5+ both directions. Train Qwen LoRA. Log to W&B. | Few-shot baselines. Wire in BLEU, CodeBLEU, syntax checks. Test the harness on first checkpoints. |
| 3 | Help with error labelling. Write the data section. | Final runs, optional ablation, produce generations. | Full eval, error taxonomy, plots. All: report, slides, cleanup, README. |

Hard rule: data freezes by mid-week 2. If CodeNet pairing runs late, use XLCoST and Rosetta only.

(The course guidelines require every member to do technical work. Rotate roles so each person trains at least one configuration and contributes to the harness or data pipeline.)

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

---

## 11. Proposed problem statements (aligned with the course guidelines)

The course guidelines (section 1.1) say a project must investigate deep learning architectures or learning methods, not just run a pretrained model. Each statement below names a DL question, a dataset, a method and an evaluation.

| Option | Problem statement | Main DL angle | Risk |
|---|---|---|---|
| **A (recommended)** | How do architecture and adaptation choices affect Python <-> JS translation? Compare CodeT5+ 220M (encoder-decoder, full fine-tune) with Qwen2.5-Coder 0.5B (decoder-only, LoRA). Ablate LoRA rank (4, 8, 16, 32) and which layers get adapters (attention only vs attention + MLP). | Architecture, parameter-efficient adaptation | Low-moderate |
| **B** | How much parallel data is needed before fine-tuning beats few-shot prompting, and does the model generalize to unseen problem types? Train on 1%, 5%, 10%, 50%, 100% of the data. | Data efficiency, generalization, learning curves | Low |
| **C** | Semantic-gap augmentation: does adding training examples for Python/JS differences (integer division, negative indexing, default sort, equality) fix the failures that dominate the error analysis? | Targeted augmentation, per-category analysis | Moderate |
| **D** | Does training on output-verified pairs (CodeNet pairs whose outputs match on shared inputs) beat training on unverified pairs? | Data quality vs quantity | Moderate-high (needs a working sandbox early) |
| **E** | How do loss and decoding choices (cross-entropy vs label smoothing; greedy, beam, sampling) change syntax validity and functional correctness? | Objective and decoding | Low value alone; use as an add-on |

**Recommendation:** Option A plus the data-size ablation from Option B. It gives a clear architectural comparison, adaptation ablations and learning curves, and it lets each team member own a technical piece (CodeT5+ training, LoRA ablations, harness and analysis).

Hypotheses to state in the report (examples):
- H1: Encoder-decoder CodeT5+ at 220M matches or beats a 0.5B decoder-only LoRA model on XLCoST text metrics because it is pretrained for code-to-code tasks, but the larger decoder-only model wins on execution accuracy for HumanEval-X because of its broader pretraining.
- H2: LoRA performance saturates at a small rank (about 8 to 16); higher ranks add cost without gains.
- H3: Adapting MLP layers in addition to attention helps on idiom-heavy cases (API mapping), not on syntax.
- H4: Fine-tuning beats few-shot prompting only beyond some data size, and the crossover point can be read from the learning curve.

## 12. How the evaluation will work in detail

### 12.1 What we are targeting
We want to know how well a small fine-tuned model translates a Python function to JavaScript and back, and why it succeeds or fails. We use two kinds of data for two different questions:

| Question | Data | Metric |
|---|---|---|
| Does the translated code look like the reference? (comparable to published baselines) | XLCoST test split, Python <-> JavaScript | BLEU, CodeBLEU, exact match |
| Does the translated code actually work? | HumanEval-X / MultiPL-E (164 problems with unit tests) | pass@1, syntax validity |
| Does it generalize to problems it never saw? | Problem-level held-out split of our own data (plus optional CodeNet pairs with verified outputs) | pass@1 or output-match rate |
| Why does it fail? | Failed translations from the runs above | Manual error taxonomy |

### 12.2 Using XLCoST (https://github.com/reddy-lab-code-research/XLCoST)
What the repository provides, based on its README:
- Parallel data in 7 languages, including Python and JavaScript, at two levels: **snippet-level** (short fragments, about 23 tokens on average) and **program-level** (whole programs, about 185 to 215 tokens on average).
- Python-JS counts (from the README tables): snippet pairs about 67K train / 3.7K val / 6.9K test (Python -> JS direction table) and program pairs about 8.2K train / 0.5K val / 0.9K test.
- Data download is a Google Drive link in the README. The translation files are in `pair_data_tok_1` (snippet translation) and `pair_data_tok_full` (program translation).
- Training and evaluation code is adapted from CodeXGLUE and PLBART, so we can reuse its data format and scoring approach.
- Published baselines are Naive Copy, CodeBERT, PLBART and CodeT5 with CodeBLEU scores. The README has these tables commented out, so check the paper (https://arxiv.org/abs/2206.08474) before quoting numbers. In the commented tables, CodeT5 snippet-level CodeBLEU for Python -> JS is about 77.8 and for JS -> Python about 73.2.

How we use it:
1. **Training data** for both models (the snippet and program splits are already separated into train/val/test).
2. **Reproduction check:** fine-tune CodeT5 on the XLCoST Python-JS pairs and confirm our CodeBLEU is close to the published number. This validates our pipeline before we add our own experiments.
3. **Naive Copy baseline:** copying the source unchanged gives a floor for CodeBLEU, which shows how much of the score comes from shared syntax.
4. **Limits:** XLCoST snippets have no unit tests, so we only measure text similarity on them. Many snippets are fragments, not complete functions. We therefore never claim functional correctness from XLCoST alone.

### 12.3 Using Project CodeNet (https://github.com/IBM/Project_CodeNet)
What the repository provides, based on its README:
- 13.9M submissions to 4,053 problems in 50+ languages. About 53.6% of submissions are marked Accepted.
- Layout: `data/<problem_id>/<Language>/<submission_id>.<ext>`, for example `data/p00001/JavaScript/s300682070.js`. Metadata is in `metadata/<problem_id>.csv` with fields `language`, `status`, `cpu_time`, `accuracy` and others.
- `derived/` has sample input/output for each problem, near-duplicate information and benchmarks. Tools for tokenizing, AST generation and graph building are in `tools/`.
- Download is about 7.8 GB: https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
- The Python800 benchmark is included, but it is for classification, not translation.

How we use it:
1. **Pair building:** for each problem that has both a Python and a JavaScript submission with status `Accepted`, pair the two solutions. Split train/val/test **by problem id** so no problem appears in two splits.
2. **Output-match verification (optional, data-quality experiment):** run both solutions on the sample inputs in `derived/` and keep only pairs whose outputs match. This yields pairs that are functionally equivalent.
3. **Held-out functional test:** for test-split problems, translate the Python solution to JavaScript, run it on the sample inputs, and compare its output with the expected output (match rate = functional accuracy on CodeNet).
4. **Limits:** most solutions are whole programs that read stdin and print to stdout, not standalone functions. We expect to extract only a subset as functions, or wrap stdin/stdout with a small harness. Sample I/O is also small, so a pass is weaker evidence than HumanEval-X's unit tests. Competitive-programming style is not typical of production code. Check the license and the anonymization notes before use.

### 12.4 Evaluation procedure (step by step)
1. **Freeze the splits:** train/val/test by problem. Run near-duplicate detection between train and HumanEval-X prompts and remove matches.
2. **Validate the harness** by running the Python and JS canonical solutions from HumanEval-X through it. All 164 must pass before any model output is scored.
3. **Generate outputs:** greedy decoding, fixed max length, same prompt template per model. Save raw generations to disk so scoring can be re-run.
4. **Score:**
   - Parse each output with tree-sitter (syntax validity).
   - Run each output against the unit tests in the Docker sandbox (no network, timeout 10 s, memory limit). Count pass or fail (pass@1).
   - Compute BLEU (`sacrebleu`) and CodeBLEU (`codebleu`) against the references on the XLCoST test split.
5. **Report per direction** (Py -> JS and JS -> Py) and per configuration.
6. **Uncertainty:** with 164 problems, one problem is about 0.6 percentage points. Report counts, bootstrap confidence intervals, and do not over-read differences smaller than a few points. If budget allows, run 2 to 3 seeds for the main configurations.
7. **Training diagnostics (required for the report):** training loss and validation loss per epoch, validation pass@1 per epoch, and the generalization gap, to discuss overfitting.
8. **Error analysis:** sample about 50 failures per main model, label each with one category (syntax error, wrong API mapping, off-by-one or negative indexing, integer vs float division, type coercion or equality, string/array method mismatch, sort default, missing import or helper, other), and report counts per category and per model.

### 12.5 How we will grade the models

| Layer | What we report | How we use it |
|---|---|---|
| Headline | pass@1 on HumanEval-X, per direction | Main comparison between configurations |
| Sanity | Syntax validity rate | Separates "can't write valid code" from "writes valid but wrong code" |
| Comparability | BLEU and CodeBLEU on XLCoST, versus the published CodeT5/PLBART/CodeBERT and Naive Copy numbers | Checks our pipeline against prior work |
| Training behavior | Loss curves, generalization gap, per-epoch pass@1 | Overfitting analysis |
| Ablation | Rank, target modules, data size, full fine-tune vs LoRA | Tests hypotheses H1 to H4 |
| Diagnosis | Error taxonomy counts | Explains why models fail |

Expected report tables:

| Model | Adaptation | Direction | Syntax valid % | pass@1 | CodeBLEU (XLCoST) | BLEU (XLCoST) |
|---|---|---|---|---|---|---|
| Naive copy | none | Py->JS | | | | |
| Qwen 0.5B few-shot | none | Py->JS | | | | |
| Qwen 0.5B | LoRA r=8 | Py->JS | | | | |
| CodeT5+ 220M | full fine-tune | Py->JS | | | | |
| (same rows) | | JS->Py | | | | |

| LoRA rank | Target modules | Trainable params | pass@1 | Val loss |
|---|---|---|---|---|
| 4 / 8 / 16 / 32 | attention only; attention + MLP | | | |

| Training data | pass@1 (fine-tuned) | pass@1 (few-shot baseline) |
|---|---|---|
| 10% / 50% / 100% | | |

### 12.6 Known evaluation limitations (state these in the report)
- HumanEval-X problems are short and may appear in pretraining data (contamination). Pretrained models may score higher than their translation ability alone would explain.
- XLCoST measures text similarity only. CodeBLEU does not guarantee correct behavior.
- CodeNet sample I/O is a weak test, and programs are stdin/stdout oriented.
- Translating a canonical solution tests function-level translation of simple algorithms, not idiomatic translation of real projects.
- Small test sets mean noisy comparisons. Use confidence intervals.
