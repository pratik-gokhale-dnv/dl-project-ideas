# Other Deep Learning Project Ideas

Each idea has a title and a short description of what can be done. Check compute needs and dataset licenses before committing.

## A. Fine-tuning existing models

1. **YOLOv8/v11 on a niche dataset** - Fine-tune on PPE detection, crop disease, traffic signs or wildlife. Compare against RT-DETR and analyze accuracy vs. speed.
2. **BERT/RoBERTa/DeBERTa domain classification or NER** - Fine-tune for legal, biomedical or code-review text. Compare model sizes and domain-adaptive pretraining.
3. **LoRA/QLoRA fine-tuning of a small LLM vs. RAG** - Fine-tune Llama/Qwen/Phi for a domain QA task and compare with retrieval-augmented generation.
4. **Segmentation with SAM or SegFormer** - Adapt to medical or satellite imagery, evaluate with IoU/Dice, test few-shot adaptation.
5. **Whisper fine-tuning for low-resource language or accent** - Measure word error rate improvement and data efficiency.
6. **Vulnerability detection with CodeBERT** - Fine-tune on code vulnerability datasets (e.g. Devign, BigVul) and analyze false positives.
7. **Code summarization or code search** - Fine-tune CodeT5 on CodeSearchNet and evaluate with BLEU and retrieval metrics.
8. **Image captioning or VQA** - Fine-tune BLIP/CLIP-based models on a custom domain.
9. **Text-to-SQL** - Fine-tune a small LLM on Spider and analyze failure types.
10. **Stable Diffusion personalization** - DreamBooth/LoRA for a custom style, evaluate with CLIP score and user study.

## B. Implementing research papers

1. **Vision Transformer (ViT) from scratch** - Compare to a CNN on CIFAR-10 and study data efficiency and augmentation.
2. **LoRA/QLoRA from scratch** - Implement low-rank adapters and benchmark vs. full fine-tuning.
3. **DDPM diffusion model** - Train on MNIST/CelebA, add classifier-free guidance, compare samplers.
4. **Mamba / state-space models** - Compare with Transformers on long-sequence tasks.
5. **nanoGPT-style language model** - Train a small GPT and study scaling laws and tokenization effects.
6. **Graph neural networks (GraphSAGE, GAT)** - Node classification or molecule property prediction.
7. **NeRF or 3D Gaussian Splatting** - Reconstruct scenes from images and compare variants.
8. **Reinforcement learning (DQN, PPO variants)** - Atari or MuJoCo, with ablations on tricks like prioritized replay.
9. **Retrieval models (DPR, ColBERT)** - Build a dense retriever and evaluate on a QA benchmark.
10. **Contrastive learning (SimCLR, MoCo, CLIP-style)** - Self-supervised pretraining on a small dataset, then linear-probe evaluation.
11. **Attention variants (FlashAttention concepts, linear attention)** - Implement and benchmark speed and memory.
12. **Neural machine translation with Transformers** - Implement "Attention Is All You Need" and train on a small language pair.

## C. Analysis-oriented projects

1. **Adversarial robustness** - FGSM/PGD attacks and adversarial training on CIFAR or an ImageNet subset.
2. **Model compression** - Pruning, quantization and distillation, reporting accuracy vs. latency and size.
3. **Reproducibility study** - Reproduce a recent paper and test whether its claims hold.
4. **Interpretability** - Grad-CAM, attention analysis or probing classifiers on a fine-tuned model.
5. **Out-of-distribution and calibration** - Evaluate uncertainty methods (temperature scaling, ensembles, MC dropout).
6. **Bias and fairness evaluation** - Audit a pretrained model and test mitigation techniques.
7. **Continual learning** - Compare methods (EWC, replay) on split benchmarks.

## D. Code-related ideas (related to the TransCoder project)

1. **Code translation for another language pair** - e.g. Python <-> Go or Java.
2. **Automatic program repair** - Fine-tune a model to fix bugs, evaluate with tests.
3. **Code comment / docstring generation** - Evaluate with BLEU and human ratings.
4. **Unit test generation** - Generate tests with an LLM and measure coverage.
5. **Natural language to Bash commands** - Fine-tune on NL2Bash and evaluate with execution or command matching.

## Tips for choosing
- Pick a project where you can show a clear experiment, ablation and error analysis.
- Prefer a well-defined dataset and metric.
- Estimate compute early, and keep a fallback smaller scope.
- Papers With Code has largely moved to Hugging Face Papers, which also lists linked code.
