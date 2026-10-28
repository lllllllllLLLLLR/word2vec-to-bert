# -*- coding: utf-8 -*-
"""
TP2 GloVe extension
Run only the 4 GloVe static NER runs (Medical/Press x LSTM/CNN) and merge
into the existing 7-model table, producing the final 9-model comparison.

The static model code (loading, tagging, training, evaluation) is copied
verbatim from TP2_NER_Ran_Lu.ipynb cells 3/7/9/11 so results stay comparable.
The 7 existing numbers are parsed from the notebook's cell-15 output.
"""
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors
try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2
except ImportError:
    class IOB2:
        pass

    def _entities(tags, sentence_id):
        entities, active = set(), None
        for index, tag in enumerate(tags + ['O']):
            if tag.startswith('B-'):
                if active is not None:
                    entities.add((sentence_id, *active))
                active = (tag[2:], index, index + 1)
            elif tag.startswith('I-') and active is not None and active[0] == tag[2:]:
                active = (active[0], active[1], index + 1)
            else:
                if active is not None:
                    entities.add((sentence_id, *active))
                active = None
        return entities

    def _scores(true_sequences, pred_sequences):
        true_entities = set().union(*[_entities(tags, i) for i, tags in enumerate(true_sequences)])
        pred_entities = set().union(*[_entities(tags, i) for i, tags in enumerate(pred_sequences)])
        tp = len(true_entities & pred_entities)
        precision = tp / len(pred_entities) if pred_entities else 0.0
        recall = tp / len(true_entities) if true_entities else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    def precision_score(true_sequences, pred_sequences, **kwargs):
        return _scores(true_sequences, pred_sequences)[0]

    def recall_score(true_sequences, pred_sequences, **kwargs):
        return _scores(true_sequences, pred_sequences)[1]

    def f1_score(true_sequences, pred_sequences, **kwargs):
        return _scores(true_sequences, pred_sequences)[2]

    def classification_report(true_sequences, pred_sequences, output_dict=False, **kwargs):
        labels = sorted({entity[1] for sentence_id, tags in enumerate(true_sequences + pred_sequences) for entity in _entities(tags, sentence_id)})
        report = {}
        for label in labels:
            true_entities = {(sentence_id, kind, start, end) for sentence_id, tags in enumerate(true_sequences) for _, kind, start, end in _entities(tags, sentence_id) if kind == label}
            pred_entities = {(sentence_id, kind, start, end) for sentence_id, tags in enumerate(pred_sequences) for _, kind, start, end in _entities(tags, sentence_id) if kind == label}
            tp = len(true_entities & pred_entities)
            precision = tp / len(pred_entities) if pred_entities else 0.0
            recall = tp / len(true_entities) if true_entities else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            report[label] = {'precision': precision, 'recall': recall, 'f1-score': f1}
        return report if output_dict else ''

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))
DEVICE = torch.device('cpu')

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = ROOT / 'models_word_vectors'
NER_ROOT = ROOT / 'TP_ISD2020/QUAERO_FrenchMed'

EMBEDDING_PATHS = {
    'Medical GloVe': EMBEDDING_DIR / 'med_glove.kv',
    'Press GloVe': EMBEDDING_DIR / 'press_glove.kv',
}


def load_conll(path):
    sentences = []
    current = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        fields = line.split()
        current.append((fields[1], fields[-1]))
    if current:
        sentences.append(current)
    return sentences


def load_split(split):
    sentences = []
    for source in ['EMEA', 'MEDLINE']:
        sentences.extend(load_conll(NER_ROOT / source / f'{source}{split}_layer1_ID.conll'))
    return sentences


splits = {split: load_split(split) for split in ['train', 'dev', 'test']}
train_sentences = [[token for token, _ in sentence] for sentence in splits['train']]
label_names = sorted({tag for sentence in splits['train'] for _, tag in sentence}, key=lambda tag: (tag != 'O', tag))
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
word2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for token in sorted({token for sentence in train_sentences for token in sentence}):
    word2id[token] = len(word2id)
pad_id = word2id[PAD_TOKEN]
unk_id = word2id[UNK_TOKEN]


def load_keyed_vectors(path):
    path = Path(path)
    if path.suffix == '.model':
        return Word2Vec.load(str(path)).wv
    if path.suffix == '.kv':
        return KeyedVectors.load(str(path))
    return load_facebook_vectors(str(path))


def build_embedding_matrix(keyed_vectors):
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.05, size=(len(word2id), keyed_vectors.vector_size)).astype('float32')
    matrix[pad_id] = 0.0
    covered = 0
    for token, index in word2id.items():
        if token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        if token in keyed_vectors:
            matrix[index] = keyed_vectors[token]
            covered += 1
    return matrix, covered


class StaticNERDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        token_ids = [word2id.get(token, unk_id) for token, _ in sentence]
        tag_ids = [label2id[tag] for _, tag in sentence]
        return torch.tensor(token_ids), torch.tensor(tag_ids)


def collate_static(batch):
    max_len = max(len(token_ids) for token_ids, _ in batch)
    token_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    label_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, (token_ids, tag_ids) in enumerate(batch):
        token_batch[row, :len(token_ids)] = token_ids
        label_batch[row, :len(tag_ids)] = tag_ids
    return token_batch, label_batch


train_loader = DataLoader(StaticNERDataset(splits['train']), batch_size=32, shuffle=True, collate_fn=collate_static)
dev_loader = DataLoader(StaticNERDataset(splits['dev']), batch_size=32, shuffle=False, collate_fn=collate_static)
test_loader = DataLoader(StaticNERDataset(splits['test']), batch_size=32, shuffle=False, collate_fn=collate_static)


class TokenLSTMTagger(nn.Module):
    def __init__(self, embedding_matrix, num_labels, hidden_size=96, layers=1, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=True, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, layers, batch_first=True, bidirectional=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, token_ids):
        sequence, _ = self.lstm(self.embedding(token_ids))
        return self.classifier(self.dropout(sequence))


class TokenCNNTagger(nn.Module):
    def __init__(self, embedding_matrix, num_labels, channels=64, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=True, padding_idx=pad_id)
        self.convs = nn.ModuleList([nn.Conv2d(1, channels, (kernel, embedding_matrix.shape[1]), padding=(kernel // 2, 0)) for kernel in [3, 4, 5]])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 3, num_labels)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids).unsqueeze(1)
        time_steps = token_ids.size(1)
        features = [F.relu(conv(embedded)).squeeze(3)[:, :, :time_steps] for conv in self.convs]
        sequence = torch.cat(features, dim=1).transpose(1, 2)
        return self.classifier(self.dropout(sequence))


def seqeval_metrics(true_sequences, pred_sequences):
    return {
        'Precision': precision_score(true_sequences, pred_sequences, mode='strict', scheme=IOB2),
        'Recall': recall_score(true_sequences, pred_sequences, mode='strict', scheme=IOB2),
        'F1': f1_score(true_sequences, pred_sequences, mode='strict', scheme=IOB2),
    }


def decode_static(logits, labels):
    predictions = logits.argmax(dim=-1)
    true_sequences, pred_sequences = [], []
    for row in range(labels.size(0)):
        mask = labels[row] != -100
        true_sequences.append([id2label[int(value)] for value in labels[row][mask]])
        pred_sequences.append([id2label[int(value)] for value in predictions[row][mask]])
    return true_sequences, pred_sequences


def evaluate_static(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    true_sequences, pred_sequences = [], []
    with torch.no_grad():
        for token_ids, labels in loader:
            logits = model(token_ids.to(DEVICE))
            total_loss += criterion(logits.reshape(-1, len(label_names)), labels.to(DEVICE).reshape(-1)).item()
            true_batch, pred_batch = decode_static(logits.cpu(), labels)
            true_sequences.extend(true_batch)
            pred_sequences.extend(pred_batch)
    metrics = seqeval_metrics(true_sequences, pred_sequences)
    metrics['Loss'] = total_loss / len(loader)
    return metrics, true_sequences, pred_sequences


def train_static(model, epochs=20, patience=5, learning_rate=2e-3):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)
    best_state, best_dev = copy.deepcopy(model.state_dict()), {'F1': -1.0}
    history, stale = [], 0
    model.to(DEVICE)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for token_ids, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids.to(DEVICE))
            loss = criterion(logits.reshape(-1, len(label_names)), labels.to(DEVICE).reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        dev_metrics, _, _ = evaluate_static(model, dev_loader)
        history.append({'Epoch': epoch, 'Train loss': train_loss / len(train_loader), **{f'Dev {key}': value for key, value in dev_metrics.items()}})
        if dev_metrics['F1'] > best_dev['F1']:
            best_dev = dev_metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_dev


# ---------- run the 4 GloVe runs ----------
static_runs = []
coverage_rows = []
for embedding_name, embedding_path in EMBEDDING_PATHS.items():
    keyed_vectors = load_keyed_vectors(embedding_path)
    embedding_matrix, covered = build_embedding_matrix(keyed_vectors)
    coverage_rows.append({
        'Embedding': embedding_name,
        'Dimension': keyed_vectors.vector_size,
        'Train vocabulary coverage': covered / (len(word2id) - 2),
    })
    for architecture, model_class in [('LSTM', TokenLSTMTagger), ('CNN', TokenCNNTagger)]:
        model = model_class(embedding_matrix, len(label_names))
        model, history, dev_metrics = train_static(model)
        static_runs.append({
            'Embedding': embedding_name,
            'Source corpus': 'Medical' if embedding_name.startswith('Medical') else 'Press',
            'Representation': embedding_name.split(' ', 1)[1],
            'Architecture': architecture,
            'Model': model,
            'History': history,
            **{f'Dev {key}': value for key, value in dev_metrics.items()},
        })
    print(f'{embedding_name}: vocab coverage {covered / (len(word2id) - 2):.1%}')

selected_static = []
for embedding_name in EMBEDDING_PATHS:
    candidates = [run for run in static_runs if run['Embedding'] == embedding_name]
    selected_static.append(max(candidates, key=lambda run: run['Dev F1']))

static_test_rows = []
for run in selected_static:
    test_metrics, true_sequences, pred_sequences = evaluate_static(run['Model'], test_loader)
    static_test_rows.append({
        'Model': run['Embedding'],
        'Representation': run['Representation'],
        'Architecture': run['Architecture'],
        'Selection': f"{run['Architecture']} selected by dev F1={run['Dev F1']:.4f}",
        'Precision': test_metrics['Precision'],
        'Recall': test_metrics['Recall'],
        'F1': test_metrics['F1'],
    })
static_test_df = pd.DataFrame(static_test_rows)
print('\n=== GloVe runs (selected by dev F1) ===')
print(static_test_df.sort_values('F1', ascending=False).to_string(index=False))

# ---------- merge with existing 7 models (parsed from TP2 notebook cell-15 output) ----------
existing_rows = [
    {'Model': 'BERT multilingual cased', 'Representation': 'Contextual Transformer', 'Architecture': 'BERT token classifier', 'Selection': 'Epoch 2 selected by dev F1=0.6005', 'Precision': 0.5998, 'Recall': 0.5223, 'F1': 0.5584},
    {'Model': 'Press FastText CBOW', 'Representation': 'FastText CBOW', 'Architecture': 'CNN', 'Selection': 'CNN selected by dev F1=0.2666', 'Precision': 0.3443, 'Recall': 0.1757, 'F1': 0.2327},
    {'Model': 'Press Word2Vec Skip-gram', 'Representation': 'Word2Vec Skip-gram', 'Architecture': 'CNN', 'Selection': 'CNN selected by dev F1=0.2332', 'Precision': 0.3440, 'Recall': 0.1330, 'F1': 0.1918},
    {'Model': 'Press Word2Vec CBOW', 'Representation': 'Word2Vec CBOW', 'Architecture': 'CNN', 'Selection': 'CNN selected by dev F1=0.1954', 'Precision': 0.2844, 'Recall': 0.1167, 'F1': 0.1655},
    {'Model': 'Medical Word2Vec Skip-gram', 'Representation': 'Word2Vec Skip-gram', 'Architecture': 'CNN', 'Selection': 'CNN selected by dev F1=0.1086', 'Precision': 0.1703, 'Recall': 0.0633, 'F1': 0.0923},
    {'Model': 'Medical FastText CBOW', 'Representation': 'FastText CBOW', 'Architecture': 'CNN', 'Selection': 'CNN selected by dev F1=0.0037', 'Precision': 0.0435, 'Recall': 0.0010, 'F1': 0.0019},
    {'Model': 'Medical Word2Vec CBOW', 'Representation': 'Word2Vec CBOW', 'Architecture': 'CNN', 'Selection': 'CNN selected by dev F1=0.0005', 'Precision': 0.2857, 'Recall': 0.0005, 'F1': 0.0010},
]
existing_df = pd.DataFrame(existing_rows)
final_results = pd.concat([static_test_df, existing_df], ignore_index=True)
final_results = final_results.sort_values('F1', ascending=False).reset_index(drop=True)
pd.set_option('display.float_format', lambda v: f'{v:.4f}')
print('\n=== FINAL 9-MODEL COMPARISON ===')
print(final_results.to_string(index=False))

# save to csv for later use
out_csv = ROOT / 'tp2_ner_outputs' / 'nine_model_comparison.csv'
out_csv.parent.mkdir(exist_ok=True)
final_results.to_csv(out_csv, index=False)
print(f'\nsaved {out_csv}')
