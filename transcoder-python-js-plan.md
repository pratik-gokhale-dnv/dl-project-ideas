# Python <-> JavaScript Code Translation (TransCoder-style) - Project Plan

Team size: 3 | Duration: one semester | Compute: Colab Pro / university cluster / 1 GPU

## 1. Goal
Build a neural code translator between **Python and JavaScript** (both directions) by fine-tuning a pretrained code model, and evaluate it with both text-similarity metrics and **execution-based tests** (computational accuracy), as in the TransCoder paper (Roziere et al., 2020).

## 2. Why this scope is realistic
- The original TransCoder is *unsupervised* (cross-lingual MLM pretraining, denoising auto-encoding, back-translation) and needs huge corpora and GPU time. We do NOT reproduce that.
- We **fine-tune a pretrained code model** (supervised, with optional back-translation augmentation).
- Python and JS are both dynamically typed with similar idioms, so this is the easiest pair.

**Difficulty: moderate.** Hardest parts: building the execution test harness and getting clean parallel data.

## 3. Research questions
1. How well does a fine-tuned small model translate function-level Python <-> JS?
2. Does fine-tuning beat few-shot prompting of a larger model?
3. Does back-translation augmentation help?
4. What kinds of errors are most common (syntax, semantics, library mapping, type/semantic differences)?

## 4. Model options (pick one primary, one baseline)
| Model | Notes |
|---|---|
| CodeT5 / CodeT5+ (220M-770M) | Encoder-decoder, natural fit for translation, trains on a single GPU |
| CodeBERT / GraphCodeBERT | Encoder-only; needs a decoder, used in CodeXGLUE baselines |
| StarCoder2-3B / Qwen2.5-Coder / CodeLlama-7B + LoRA/QLoRA | Decoder-only, strongest, use parameter-efficient fine-tuning |
| Baseline: pretrained model with few-shot prompting | No training, for comparison |

Suggested: **CodeT5+ 220M (full fine-tune)** vs **a ~1-7B decoder model with LoRA** vs **few-shot baseline**.

## 5. Data sources
Verify language coverage and licenses before relying on any of these.
- **XLCoST** - parallel snippet/program-level data across several languages (includes Python and JavaScript).
- **CodeXGLUE (code-to-code)** - mainly Java <-> C#, useful for pipeline reference, not directly Python/JS.
- **AVATAR** - parallel Java/Python; reference for format and evaluation.
- **CodeNet (IBM)** - many solutions per problem across languages; can build pseudo-parallel pairs by matching problem IDs.
- **Codeforces / LeetCode / Rosetta Code / GeeksforGeeks-style problem sets** - same problem solved in multiple languages; check licensing.
- **The Stack / The Stack v2 (Hugging Face)** - large monolingual Python and JS for back-translation or pretraining.
- **TransCoder test set** (facebookresearch/TransCoder) - GeeksforGeeks functions with unit tests, check whether it covers JS or only Java/C++/Python.
- **HumanEval-X / MultiPL-E** - HumanEval problems with tests in many languages including Python and JS; excellent for **test-based evaluation**.

### Building pseudo-parallel data
1. Group solutions by problem ID (CodeNet, Rosetta Code).
2. Pair Python and JS solutions of the same problem.
3. Filter: compile/parse OK, length limits, deduplicate, remove near-duplicates.
4. Optionally verify by running both on the same inputs and keeping pairs with matching outputs.

### Back-translation augmentation (optional)
1. Train initial Py->JS and JS->Py models.
2. Translate monolingual code from The Stack.
3. Keep outputs that parse (and pass tests if available) as synthetic pairs.
4. Retrain with real + synthetic data.

## 6. Data preprocessing
- Scope: **function-level** translation (single function with standard-library use only) to keep it tractable.
- Strip comments or keep them as a controlled variable.
- Tokenization: use the model's tokenizer; check max length (e.g. 512 tokens).
- Parsing/validation: `ast` / `tree-sitter` for Python and JS.
- Deduplicate across train/val/test (hash + near-duplicate detection) to **avoid data leakage**.
- Split **by problem**, not by random sample.

## 7. Evaluation
### Automatic text metrics
- BLEU, CodeBLEU (note CodeBLEU's language support), exact match. These are weak indicators for code.

### Execution-based (primary)
- **Computational accuracy**: fraction of translations that pass the unit tests.
- **pass@k** if sampling multiple outputs.
- Test sets: HumanEval-X / MultiPL-E (Python<->JS), plus own held-out problems with tests.
- Harness: run Python with `pytest`/subprocess and JS with Node.js, **inside a sandbox** (Docker, resource and time limits, no network).

### Additional analyses
- Syntax validity rate (parse success).
- Error taxonomy on ~100 manually inspected failures: syntax, wrong API mapping, indexing/off-by-one, integer vs float division, type coercion, string/array method differences, missing imports.
- Performance by input length and by problem difficulty.

## 8. Baselines and ablations
- Few-shot prompting of a larger open model (and optionally an API model).
- Full fine-tune vs LoRA.
- With vs without back-translation data.
- Training data size scaling (10%, 50%, 100%).
- Direction: Py->JS vs JS->Py.

## 9. Python vs JavaScript semantic gotchas (for error analysis)
- `//` and `/` vs JS number division, no integer type in JS (`Math.floor`).
- `==` vs `===` and truthiness rules.
- Lists/dicts vs Arrays/Objects/Maps; slicing vs `slice`.
- Negative indexing (Python) not supported in JS.
- Big integers (Python arbitrary precision vs JS `BigInt`).
- String methods and formatting (f-strings vs template literals).
- Sorting defaults (JS sorts lexicographically by default).
- Standard library mapping (`collections`, `itertools`, `math`).

## 10. Timeline (about 12-14 weeks)
| Weeks | Milestone |
|---|---|
| 1-2 | Literature review (TransCoder, CodeT5, XLCoST, CodeBLEU), choose models, set up repo and environment |
| 3-4 | Data collection, cleaning, splits, sandboxed execution harness v1 |
| 5-6 | Baselines: few-shot prompting, evaluate on HumanEval-X |
| 7-9 | Fine-tuning experiments (CodeT5+, LoRA model) |
| 10 | Back-translation augmentation and ablations |
| 11 | Error analysis, plots, tables |
| 12-13 | Report and presentation, code cleanup, reproducibility |

## 11. Team split (3 people)
- **Person A - Data:** collection, pairing, cleaning, dedup, splits, tree-sitter validation.
- **Person B - Modeling:** training scripts, hyperparameters, LoRA, experiment tracking (Weights & Biases or TensorBoard).
- **Person C - Evaluation:** sandboxed test harness, metrics, error analysis, plots.
- Everyone: literature review, writing, weekly sync, code review.

## 12. Compute and tooling
- Hugging Face `transformers`, `datasets`, `peft`, `trl`, `accelerate`.
- `bitsandbytes` for QLoRA on a single 16-24GB GPU.
- Docker for sandboxing; Node.js LTS and Python 3.10+.
- Experiment tracking: Weights & Biases.

## 13. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Not enough clean parallel data | Pseudo-parallel pairing, back-translation, filter aggressively |
| Execution harness takes too long | Start early (weeks 3-4); reuse MultiPL-E / HumanEval-X harness ideas |
| Data leakage (pretraining or train/test overlap) | Split by problem, dedup, note that pretrained models may have seen benchmarks |
| Compute limits | Small models, LoRA/QLoRA, shorter sequences |
| Insecure code execution | Docker sandbox, timeouts, no network |

## 14. Stretch goals
- Add **Go** as a third language (harder because of static typing).
- Class- or file-level translation.
- Use compiler/runtime feedback (execution-guided repair loop).
- Reinforcement learning from test results.

## 15. Deliverables
- Cleaned dataset and split scripts
- Training and evaluation code
- Results table (BLEU, CodeBLEU, syntax validity, computational accuracy)
- Error analysis section
- Final report and presentation

## 16. References
- Roziere et al., *Unsupervised Translation of Programming Languages* (TransCoder), 2020 - https://github.com/facebookresearch/TransCoder
- Wang et al., *CodeT5 / CodeT5+*
- Zhu et al., *XLCoST*
- Ren et al., *CodeBLEU*
- Cassano et al., *MultiPL-E*; Zheng et al., *HumanEval-X (CodeGeeX)*
- Puri et al., *Project CodeNet*
- Hu et al., *LoRA*; Dettmers et al., *QLoRA*
