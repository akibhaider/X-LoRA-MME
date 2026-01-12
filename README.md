# LoRA Multi-Model Ensemble for Code Comment Classification

Multi-model ensemble approach using LoRA fine-tuning for Code Comment Classification.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   HuggingFace Dataset (Java/Python/Pharo)                               │
│                    │                                                    │
│                    ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Unified Label Space (18 categories across all languages)       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                    │                                                    │
│     ┌──────────────┼──────────────┬──────────────┐                      │
│     ▼              ▼              ▼              ▼                      │
│ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐                    │
│ │UniXcoder│   │CodeBERT│    │GraphCB │    │CodeBERTa│                   │
│ │ +LoRA  │    │ +LoRA  │    │ +LoRA  │    │ +LoRA  │                    │
│ └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘                    │
│     │             │             │             │                         │
│     └─────────────┴──────┬──────┴─────────────┘                         │
│                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Learn Per-Label Ensemble Weights (softmax of validation F1s)   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Optimize Thresholds Per (Language, Category)                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PHASE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: "class_name | comment_text"                                    │
│                    │                                                    │
│     ┌──────────────┼──────────────┬──────────────┐                      │
│     ▼              ▼              ▼              ▼                      │
│  [Model 1]     [Model 2]     [Model 3]     [Model 4]                    │
│  Probs[18]     Probs[18]     Probs[18]     Probs[18]                    │
│     │              │             │              │                       │
│     └──────────────┴──────┬──────┴──────────────┘                       │
│                           ▼                                             │
│          Weighted Ensemble: P[i] = Σ(w[i,m] × P_m[i])                   │
│                           │                                             │
│                           ▼                                             │
│          Apply Language-Specific Thresholds                             │
│                           │                                             │
│                           ▼                                             │
│                  Binary Predictions [18]                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Base Models

| Model | Source |
|-------|--------|
| UniXcoder | `microsoft/unixcoder-base` |
| CodeBERT | `microsoft/codebert-base` |
| GraphCodeBERT | `microsoft/graphcodebert-base` |
| CodeBERTa | `huggingface/CodeBERTa-small-v1` |

### 2. LoRA Fine-Tuning

- **Rank (r):** 16
- **Alpha:** 32
- **Dropout:** 0.1
- **Target Modules:** `query`, `key`, `value`, `dense`

### 3. Training Strategy

- **Loss:** Focal Loss (γ=2.0) with class weights for imbalance
- **Optimizer:** AdamW with linear warmup scheduler
- **Epochs:** 15 with early stopping on best validation F1

### 4. Ensemble Weights

Per-label weights learned from validation F1 scores using temperature-scaled softmax:
```
w[label, model] = exp(F1[label, model] / τ) / Σ exp(F1[label, :] / τ)
```

### 5. Threshold Optimization

Per-(language, category) thresholds optimized on validation set by grid search over [0.1, 0.9].

---

## Usage

**Training:**
```bash
# Run Training.ipynb locally or on GPU
# Outputs: trained_models/*.pt + ensemble_config.pkl
```

**Evaluation:**
```bash
# Run Evaluation.ipynb on Google Colab T4
# Loads models and evaluates on test set
```

---

## References

1. [NLBSE'26 Tool Competition](https://nlbse2026.github.io/tools/)
2. [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
3. [CodeBERT](https://arxiv.org/abs/2002.08155)

