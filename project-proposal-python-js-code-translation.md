# Project Proposal (Draft)

## Team name, project title, and group members

- **Team name:** TBD (placeholder: *Transpilers*)
- **Project title:** How Architecture and Adaptation Choices Affect Neural Python <-> JavaScript Code Translation
- **Group members:** TBD (Member 1, Member 2, Member 3)

## Project summary

Translating code between programming languages is a common need: teams migrate codebases, and developers learn a new language by reading equivalents of programs they already know. Rule-based transpilers need hand-written rules for every language pair, and they struggle with idioms. Neural models can learn the mapping from examples, and pretrained code models make that possible with modest data. We study function-level translation between Python and JavaScript in both directions. The two languages are both dynamically typed but differ in subtle ways (integer vs float division, negative indexing, default sort order, equality rules), which makes them a good test of what a model has learned rather than just copied. Our main question is not "does a model translate code?" but "which architecture and adaptation choices matter?" We compare an encoder-decoder model with a decoder-only model, full fine-tuning with LoRA, and different LoRA configurations. Deep Learning is appropriate because the task is sequence-to-sequence learning over structured text, where pretrained transformer representations transfer well and where adaptation choices (which layers to train, how many parameters) have measurable effects.

## Approach

We will fine-tune two pretrained code models: **CodeT5+ 220M** (encoder-decoder, full fine-tune) and **Qwen2.5-Coder 0.5B** (decoder-only, LoRA). We will also run a few-shot baseline with an unadapted Qwen model. The experiments are designed to test specific choices:

1. **Architecture:** encoder-decoder vs decoder-only at a comparable parameter budget.
2. **Adaptation method:** full fine-tuning vs LoRA.
3. **LoRA configuration:** rank r in {4, 8, 16, 32}, and adapting attention projections only vs attention plus MLP layers.
4. **Data size:** training on 10%, 50% and 100% of the data to find how much parallel data is needed before fine-tuning beats few-shot prompting.

We will use existing libraries (Hugging Face `transformers`, `peft`, PyTorch with the MPS backend, `mlx-lm` as an optional fast path, `tree-sitter`) and pretrained weights. Our own contributions are: the data pipeline (function extraction, filtering, near-duplicate removal, problem-level splits), a custom training and evaluation loop that logs loss, validation loss and per-epoch execution accuracy, a sandboxed execution harness for Node.js and Python, and the analysis. To avoid treating the framework as a black box, we will write the data collator and masking (labels set to -100 on padding and prompt tokens), inspect tokenization of Python and JavaScript, plot training and validation curves to look for overfitting, and run learning-rate and rank sweeps. If time permits we will inspect cross-attention maps (CodeT5+) to see whether the model aligns equivalent constructs across languages. All three of us will do technical work: each member trains at least one configuration, and data and harness work will be shared.

## Resources, related work, and papers

- **TransCoder** (Roziere et al., 2020) introduced unsupervised program translation; we do not reproduce it but use it as context. https://arxiv.org/abs/2006.03511
- **CodeT5 / CodeT5+** (Wang et al., 2021; 2023) are pretrained encoder-decoder code models and our primary full fine-tuning model. https://arxiv.org/abs/2109.00859, https://arxiv.org/abs/2305.07922
- **Qwen2.5-Coder** is a decoder-only code model family used for the LoRA experiments and few-shot baseline. https://arxiv.org/abs/2409.12186
- **LoRA** (Hu et al., 2021) is the parameter-efficient method we ablate. https://arxiv.org/abs/2106.09685
- **XLCoST** (Zhu et al., 2022) is a parallel dataset in 7 languages that reports CodeBERT, PLBART and CodeT5 baselines, which we compare against. https://arxiv.org/abs/2206.08474, https://github.com/reddy-lab-code-research/XLCoST
- **Project CodeNet** (Puri et al., 2021) provides many solutions to the same problem across 50+ languages. https://arxiv.org/abs/2105.12655, https://github.com/IBM/Project_CodeNet
- **HumanEval-X** (Zheng et al., 2023) and **MultiPL-E** (Cassano et al., 2022) provide unit-tested problems in Python and JavaScript for execution-based evaluation. https://arxiv.org/abs/2303.17568, https://arxiv.org/abs/2208.08227
- **CodeBLEU** (Ren et al., 2020) is a text-similarity metric for code. https://arxiv.org/abs/2009.10297
- **Codex paper** (Chen et al., 2021) defines the pass@k estimator. https://arxiv.org/abs/2107.03374
- **Lost in Translation** (Pan et al., 2023) studies bugs in LLM-based code translation and informs our error taxonomy. https://arxiv.org/abs/2308.03109

Some links and dataset details are from memory and still need to be verified. We will compare our XLCoST results against the published baselines where practical, since unofficial implementations can differ.

## Datasets

1. **XLCoST** (primary training data). Python-JavaScript snippet-level pairs: about 67K train / 3.7K validation / 6.9K test. Program-level pairs: about 8.2K / 0.5K / 0.9K. Data is distributed through a Google Drive link in the repository README. Tokenized parallel files are provided. Snippets are fragments and have no unit tests, so we use XLCoST for training and text-metric comparison.
2. **Project CodeNet** (secondary training data). 13.9M submissions to 4,053 problems, with metadata CSV files (language, status, accuracy), so we can pair accepted Python and JavaScript solutions by problem ID. Per-problem sample input/output in `derived/` lets us verify that paired solutions behave the same. Solutions are full programs that read stdin, so we expect to extract only a subset as clean functions. Download is about 7.8 GB.
3. **HumanEval-X / MultiPL-E** (test only). 164 problems with unit tests in Python and JavaScript. We never train on these and we remove near-duplicates from the training data.

We will decide how much of CodeNet to use after checking XLCoST's quality and coverage. We will verify licenses and access before relying on any dataset, and report dataset composition, collection method and known biases in the report.

## Evaluation plan

- **Primary: execution-based.** Translate the HumanEval-X Python canonical solutions to JavaScript (and the reverse), run the original tests in a sandbox (Docker, no network, time and memory limits), and report **pass@1** with greedy decoding. Optionally report pass@k.
- **Syntax validity rate:** the percentage of outputs that parse with tree-sitter.
- **Text metrics on the XLCoST test split:** BLEU and CodeBLEU, compared against published CodeT5/PLBART/CodeBERT results. These are secondary because text similarity is a weak proxy for correctness.
- **Training diagnostics:** training and validation loss per epoch, the gap between them, and pass@1 per epoch to detect overfitting.
- **Ablations:** architecture, full fine-tune vs LoRA, LoRA rank and target modules, and data size (10%, 50%, 100%). Each result is reported per direction.
- **Error analysis:** about 50 failures per main model labelled by category (syntax, API mapping, indexing, integer division, type coercion, sort default, other), with a hypothesis for each major error type.

See `transcoder-python-js-implementation-and-resources.md` (sections 11 and 12) for the detailed evaluation design.

## Risks and scope

- **Core project:** CodeT5+ full fine-tune vs Qwen 0.5B LoRA, with a rank ablation, a data-size ablation, execution-based evaluation on HumanEval-X, and error analysis.
- **Stretch goals:** Qwen 1.5B, target-module ablation, attention analysis, CodeNet-derived verified pairs, and MultiPL-E.
- **Compute:** training on a MacBook (MPS/MLX) can be slow or hit unsupported operations. Fallbacks are shorter sequences (384 tokens), fewer training pairs, and a free Colab GPU.
- **Data:** XLCoST snippets may be too short or noisy. CodeNet pairing may yield few clean functions. The fallback is to train on XLCoST alone.
- **Contamination:** pretrained models may have seen HumanEval problems. We will state this as a limitation and also test on a small held-out set of our own.
- **Harness bugs:** we will validate the execution harness on the canonical solutions in week 1 before any model output exists.
- **Timeline:** about 3 weeks, so the experiment grid is deliberately small.

## Ethical implications

Code translation tools can introduce subtle bugs that go unnoticed, so users could trust incorrect or insecure translations. We will report failure modes and make clear that outputs need review and testing. Models trained on public code may reproduce licensed code or leak secrets; we use datasets with documented sources and will check licenses, remove obvious secrets, and avoid distributing training data we do not have rights to share. Automated translation could be misused to port malicious code between languages; our scope is limited to benign benchmark-style functions, and we will not train on or release malware-related data. Running generated code is a safety risk, so we execute it only in a sandbox with no network access. We use no NSFW content and no personal data beyond what is already in public code datasets (CodeNet anonymizes user IDs).
