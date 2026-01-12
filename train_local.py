#!/usr/bin/env python3
"""
X-LoRA Ensemble Training Script (Local GPU)
Run this to train models locally, then use 2_Colab_Evaluation.ipynb for testing.
"""

import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'max_length': 256,
    'batch_size': 16,
    'gradient_accumulation': 2,
    'epochs': 15,
    'learning_rate': 2e-4,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'seed': 42,
}

MODEL_CONFIGS = {
    'unixcoder': {
        'name': 'microsoft/unixcoder-base',
        'hidden_size': 768,
        'target_modules': ['query', 'key', 'value', 'dense'],
    },
    'codebert': {
        'name': 'microsoft/codebert-base',
        'hidden_size': 768,
        'target_modules': ['query', 'key', 'value', 'dense'],
    },
    'graphcodebert': {
        'name': 'microsoft/graphcodebert-base',
        'hidden_size': 768,
        'target_modules': ['query', 'key', 'value', 'dense'],
    },
    'codeberta': {
        'name': 'huggingface/CodeBERTa-small-v1',
        'hidden_size': 768,
        'target_modules': ['query', 'key', 'value', 'dense'],
    },
}

LANG_LABELS = {
    'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
    'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
    'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Intent', 'Keymessages', 'Collaborators']
}

LANGUAGES = ['java', 'python', 'pharo']

# =============================================================================
# SETUP
# =============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(CONFIG['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# =============================================================================
# LOAD DATA
# =============================================================================
print('\n' + '='*70)
print('LOADING DATA FROM HUGGINGFACE')
print('='*70)
ds = load_dataset('NLBSE/nlbse26-code-comment-classification')

# Build unified label space
all_labels = []
for lang in LANGUAGES:
    all_labels.extend(LANG_LABELS[lang])
all_labels = sorted(list(set(all_labels)))
NUM_LABELS = len(all_labels)
label2idx = {label: idx for idx, label in enumerate(all_labels)}
idx2label = {idx: label for label, idx in label2idx.items()}

print(f'Labels ({NUM_LABELS}): {all_labels}')

def labels_to_unified(labels_list, lang):
    lang_labels = LANG_LABELS[lang]
    unified = [0] * NUM_LABELS
    for i, val in enumerate(labels_list):
        if val == 1:
            label_name = lang_labels[i]
            unified[label2idx[label_name]] = 1
    return unified

def prepare_data(split_name, lang):
    data = ds[split_name]
    texts = data['combo']
    labels = [labels_to_unified(l, lang) for l in data['labels']]
    return texts, labels

# Combine data
train_texts, train_labels, train_langs = [], [], []
test_texts, test_labels, test_langs = [], [], []

for lang in LANGUAGES:
    texts, labels = prepare_data(f'{lang}_train', lang)
    train_texts.extend(texts)
    train_labels.extend(labels)
    train_langs.extend([lang] * len(texts))
    
    texts, labels = prepare_data(f'{lang}_test', lang)
    test_texts.extend(texts)
    test_labels.extend(labels)
    test_langs.extend([lang] * len(texts))

print(f'Total training samples: {len(train_texts)}')
print(f'Total test samples: {len(test_texts)}')

# Validation split
train_texts, val_texts, train_labels, val_labels, train_langs, val_langs = train_test_split(
    train_texts, train_labels, train_langs, 
    test_size=0.1, random_state=CONFIG['seed'], stratify=train_langs
)
print(f'After split - Train: {len(train_texts)}, Val: {len(val_texts)}')

# =============================================================================
# DATASET CLASS
# =============================================================================
class CommentDataset(Dataset):
    def __init__(self, texts, labels, languages, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'language': self.languages[idx],
        }

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class XLoRAClassifier(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels, lora_config):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

def create_model(model_key, cfg, num_labels):
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=cfg['target_modules'],
        lora_dropout=CONFIG['lora_dropout'],
        bias='none',
        task_type='FEATURE_EXTRACTION',
    )
    model = XLoRAClassifier(cfg['name'], cfg['hidden_size'], num_labels, lora_config)
    return model.to(device)

# =============================================================================
# FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_model(model_key, cfg, train_texts, train_labels, train_langs, 
                val_texts, val_labels, val_langs, num_labels):
    print(f"\n{'='*70}")
    print(f"Training: {model_key} ({cfg['name']})")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_ds = CommentDataset(train_texts, train_labels, train_langs, tokenizer, CONFIG['max_length'])
    val_ds = CommentDataset(val_texts, val_labels, val_langs, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size']*2, shuffle=False)
    
    model = create_model(model_key, cfg, num_labels)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['gradient_accumulation']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * CONFIG['warmup_ratio']),
        num_training_steps=total_steps
    )
    
    pos_counts = np.array(train_labels).sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float).to(device)
    pos_weight = torch.clamp(pos_weight, max=10.0)
    
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / CONFIG['gradient_accumulation']
            loss.backward()
            train_loss += loss.item() * CONFIG['gradient_accumulation']
            
            if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.extend(batch['labels'].numpy())
        
        val_probs = np.array(val_probs)
        val_labels_arr = np.array(val_labels_list)
        val_preds = (val_probs > 0.5).astype(int)
        f1 = f1_score(val_labels_arr, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {train_loss/len(train_loader):.4f} - Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"Best Val F1: {best_f1:.4f}")
    return model, tokenizer, best_f1

# =============================================================================
# TRAIN ALL MODELS
# =============================================================================
print("\n" + "="*70)
print("X-LoRA FINE-TUNING FOR ALL MODELS")
print("="*70)

trained_models = {}
tokenizers = {}
val_f1_scores = {}

for model_key, cfg in MODEL_CONFIGS.items():
    model, tokenizer, best_f1 = train_model(
        model_key, cfg,
        train_texts, train_labels, train_langs,
        val_texts, val_labels, val_langs,
        NUM_LABELS
    )
    trained_models[model_key] = model
    tokenizers[model_key] = tokenizer
    val_f1_scores[model_key] = best_f1
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
for key, f1 in val_f1_scores.items():
    print(f"  {key}: Val F1 = {f1:.4f}")

# =============================================================================
# COLLECT VALIDATION PREDICTIONS
# =============================================================================
def get_predictions(model, tokenizer, texts, labels, languages, batch_size=32):
    model.eval()
    ds = CommentDataset(texts, labels, languages, tokenizer, CONFIG['max_length'])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    all_probs, all_labels, all_langs = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())
            all_langs.extend(batch['language'])
    
    return np.array(all_probs), np.array(all_labels), all_langs

print("\nCollecting validation predictions from all models...")
val_probs_all = []
for model_key in MODEL_CONFIGS.keys():
    print(f"  {model_key}...")
    model = trained_models[model_key]
    tokenizer = tokenizers[model_key]
    vp, vl, vlng = get_predictions(model, tokenizer, val_texts, val_labels, val_langs)
    val_probs_all.append(vp)

val_labels_arr = vl
val_langs_list = vlng

# =============================================================================
# LEARN ENSEMBLE WEIGHTS
# =============================================================================
def learn_ensemble_weights(model_probs, true_labels, temperature=0.3):
    num_models = len(model_probs)
    num_labels = model_probs[0].shape[1]
    weights = np.zeros((num_labels, num_models))
    
    for label_idx in range(num_labels):
        f1s = []
        for m in range(num_models):
            preds = (model_probs[m][:, label_idx] > 0.5).astype(int)
            f1 = f1_score(true_labels[:, label_idx], preds, zero_division=0)
            f1s.append(f1)
        f1s = np.array(f1s)
        
        if f1s.sum() > 0:
            exp_scores = np.exp(f1s / temperature)
            weights[label_idx] = exp_scores / exp_scores.sum()
        else:
            weights[label_idx] = np.ones(num_models) / num_models
    
    return weights

def ensemble_predict(model_probs, weights):
    num_samples, num_labels = model_probs[0].shape
    ensemble_probs = np.zeros((num_samples, num_labels))
    for label_idx in range(num_labels):
        for m, probs in enumerate(model_probs):
            ensemble_probs[:, label_idx] += weights[label_idx, m] * probs[:, label_idx]
    return ensemble_probs

print("\nLearning ensemble weights...")
weights = learn_ensemble_weights(val_probs_all, val_labels_arr)
val_ens_probs = ensemble_predict(val_probs_all, weights)
print("Done!")

# =============================================================================
# OPTIMIZE THRESHOLDS
# =============================================================================
def optimize_thresholds(ensemble_probs, true_labels, languages, all_labels):
    num_labels = ensemble_probs.shape[1]
    thresholds = {lang: np.ones(num_labels) * 0.5 for lang in LANGUAGES}
    
    print("\nOptimizing thresholds per (language, category)...")
    for label_idx, label_name in enumerate(all_labels):
        for lang in LANGUAGES:
            mask = np.array([l == lang for l in languages])
            y_true = true_labels[mask][:, label_idx]
            y_probs = ensemble_probs[mask][:, label_idx]
            
            if len(y_true) == 0 or y_true.sum() == 0:
                continue
            
            best_f1, best_t = 0, 0.5
            for t in np.arange(0.1, 0.9, 0.02):
                f1 = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            
            thresholds[lang][label_idx] = best_t
            if best_f1 > 0:
                print(f"  {lang:6s} / {label_name:25s}: t={best_t:.2f}, F1={best_f1:.4f}")
    
    return thresholds

thresholds = optimize_thresholds(val_ens_probs, val_labels_arr, val_langs_list, all_labels)

# =============================================================================
# VALIDATE ON VALIDATION SET
# =============================================================================
def apply_thresholds(probs, thresholds, languages):
    preds = np.zeros_like(probs, dtype=int)
    for i, lang in enumerate(languages):
        preds[i] = (probs[i] >= thresholds.get(lang, np.ones(probs.shape[1]) * 0.5)).astype(int)
    return preds

val_preds = apply_thresholds(val_ens_probs, thresholds, val_langs_list)
val_f1_macro = f1_score(val_labels_arr, val_preds, average='macro', zero_division=0)
val_f1_weighted = f1_score(val_labels_arr, val_preds, average='weighted', zero_division=0)

print(f"\nValidation Set Performance:")
print(f"  F1 Macro: {val_f1_macro:.4f}")
print(f"  F1 Weighted: {val_f1_weighted:.4f}")

# =============================================================================
# SAVE MODELS AND CONFIG
# =============================================================================
os.makedirs('trained_models', exist_ok=True)

for model_key, model in trained_models.items():
    torch.save(model.state_dict(), f'trained_models/{model_key}_model.pt')
    print(f"Saved: trained_models/{model_key}_model.pt")

ensemble_config = {
    'weights': weights,
    'thresholds': thresholds,
    'all_labels': all_labels,
    'label2idx': label2idx,
    'idx2label': idx2label,
    'config': CONFIG,
    'model_configs': MODEL_CONFIGS,
    'lang_labels': LANG_LABELS,
    'languages': LANGUAGES,
    'num_labels': NUM_LABELS,
    'val_f1_scores': val_f1_scores,
}

with open('trained_models/ensemble_config.pkl', 'wb') as f:
    pickle.dump(ensemble_config, f)
print(f"Saved: trained_models/ensemble_config.pkl")

print("\n" + "="*70)
print("ALL MODELS AND CONFIG SAVED!")
print("="*70)

# Check saved files
print("\nSaved files:")
for f in os.listdir('trained_models'):
    size_mb = os.path.getsize(f'trained_models/{f}') / (1024*1024)
    print(f"  {f}: {size_mb:.1f} MB")

total_size = sum(os.path.getsize(f'trained_models/{f}') for f in os.listdir('trained_models'))
print(f"\nTotal size: {total_size / (1024*1024):.1f} MB")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Upload 'trained_models/' folder to Google Drive or Colab")
print("2. Run '2_Colab_Evaluation.ipynb' on Colab T4 for final evaluation")
