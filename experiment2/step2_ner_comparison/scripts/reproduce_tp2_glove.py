"""
Reproduce the TP2 static NER pipeline with eight embeddings:
6 TP1 models (Word2Vec CBOW/Skip-gram, FastText CBOW per corpus) + 2 GloVe
(Medical/Press, trained by scripts/train_glove.py).

Matches TP2_NER_Ran_Lu.ipynb logic: token-level LSTM/CNN taggers, frozen
embeddings, token CrossEntropyLoss with -100 ignore, seqeval strict IOB2
entity metrics. Training fixed to epochs=20, patience=5 (was 3/1). Dev-based
architecture selection per embedding; test evaluated once.

The BERT row reuses the previous run (dev F1 0.6005 -> epoch 2, test
P 0.5998 / R 0.5223 / F1 0.5584) since the BERT cell is unchanged.
Outputs: tp2_ner_outputs/ (tables CSV/JSON + figures PNG).
"""
import copy
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

ROOT = Path(".")
EMBEDDING_DIR = ROOT / "models_word_vectors"
NER_ROOT = ROOT / "TP_ISD2020/QUAERO_FrenchMed"
OUTPUT_DIR = ROOT / "tp2_ner_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, (torch.get_num_threads() or 4)))
DEVICE = torch.device("cpu")

EMBEDDING_PATHS = {
    "Medical Word2Vec CBOW": EMBEDDING_DIR / "med_word2vec_cbow.model",
    "Medical Word2Vec Skip-gram": EMBEDDING_DIR / "med_word2vec_skipgram.model",
    "Medical FastText CBOW": EMBEDDING_DIR / "med_fasttext_cbow.bin",
    "Medical GloVe": EMBEDDING_DIR / "med_glove.kv",
    "Press Word2Vec CBOW": EMBEDDING_DIR / "press_word2vec_cbow.model",
    "Press Word2Vec Skip-gram": EMBEDDING_DIR / "press_word2vec_skipgram.model",
    "Press FastText CBOW": EMBEDDING_DIR / "press_fasttext_cbow.bin",
    "Press GloVe": EMBEDDING_DIR / "press_glove.kv",
}


def load_conll(path):
    sentences, current = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
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
    for source in ["EMEA", "MEDLINE"]:
        sentences.extend(load_conll(NER_ROOT / source / f"{source}{split}_layer1_ID.conll"))
    return sentences


def load_keyed_vectors(path):
    path = Path(path)
    if path.suffix == ".model":
        return Word2Vec.load(str(path)).wv
    if path.suffix == ".kv":
        return KeyedVectors.load(str(path))
    return load_facebook_vectors(str(path))


splits = {split: load_split(split) for split in ["train", "dev", "test"]}
train_sentences = [[token for token, _ in sentence] for sentence in splits["train"]]
label_names = sorted({tag for sentence in splits["train"] for _, tag in sentence},
                     key=lambda tag: (tag != "O", tag))
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}
print(f"train/dev/test sentences: {len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])} | labels {len(label_names)}")

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
word2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for token in sorted({token for sentence in train_sentences for token in sentence}):
    word2id[token] = len(word2id)
pad_id, unk_id = word2id[PAD_TOKEN], word2id[UNK_TOKEN]


def build_embedding_matrix(keyed_vectors):
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.05, size=(len(word2id), keyed_vectors.vector_size)).astype("float32")
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
    max_len = max(len(t) for t, _ in batch)
    token_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    label_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, (token_ids, tag_ids) in enumerate(batch):
        token_batch[row, :len(token_ids)] = token_ids
        label_batch[row, :len(tag_ids)] = tag_ids
    return token_batch, label_batch


train_loader = DataLoader(StaticNERDataset(splits["train"]), batch_size=32, shuffle=True, collate_fn=collate_static)
dev_loader = DataLoader(StaticNERDataset(splits["dev"]), batch_size=32, shuffle=False, collate_fn=collate_static)
test_loader = DataLoader(StaticNERDataset(splits["test"]), batch_size=32, shuffle=False, collate_fn=collate_static)


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
        "Precision": precision_score(true_sequences, pred_sequences, mode="strict", scheme=IOB2),
        "Recall": recall_score(true_sequences, pred_sequences, mode="strict", scheme=IOB2),
        "F1": f1_score(true_sequences, pred_sequences, mode="strict", scheme=IOB2),
    }


def decode_static(logits, labels):
    predictions = logits.argmax(dim=-1)
    true_sequences, pred_sequences = [], []
    for row in range(labels.size(0)):
        mask = labels[row] != -100
        true_sequences.append([id2label[int(v)] for v in labels[row][mask]])
        pred_sequences.append([id2label[int(v)] for v in predictions[row][mask]])
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
    metrics["Loss"] = total_loss / len(loader)
    return metrics, true_sequences, pred_sequences


def train_static(model, epochs=20, patience=5, learning_rate=2e-3):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    best_state, best_dev = copy.deepcopy(model.state_dict()), {"F1": -1.0}
    history, stale = [], 0
    model.to(DEVICE)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for token_ids, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids.to(DEVICE))
            loss = cross_entropy(logits.reshape(-1, len(label_names)), labels.to(DEVICE).reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        dev_metrics, _, _ = evaluate_static(model, dev_loader)
        history.append({"Epoch": epoch, "Train loss": train_loss / len(train_loader), **{f"Dev {k}": v for k, v in dev_metrics.items()}})
        if dev_metrics["F1"] > best_dev["F1"]:
            best_dev = dev_metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_dev


def main():
    coverage_rows = []
    static_runs = []
    for embedding_name, embedding_path in EMBEDDING_PATHS.items():
        kv = load_keyed_vectors(embedding_path)
        matrix, covered = build_embedding_matrix(kv)
        coverage_rows.append({
            "Embedding": embedding_name,
            "Dimension": kv.vector_size,
            "Train vocabulary coverage": covered / (len(word2id) - 2),
        })
        for architecture, model_class in [("LSTM", TokenLSTMTagger), ("CNN", TokenCNNTagger)]:
            print(f"[{embedding_name} | {architecture}] start", flush=True)
            model = model_class(matrix, len(label_names))
            model, history, dev_metrics = train_static(model)
            print(f"  dev F1 {dev_metrics['F1']:.4f} (epoch {history['Epoch'].iloc[-1]}, best at {history['Dev F1'].idxmax()+1})", flush=True)
            static_runs.append({
                "Embedding": embedding_name,
                "Source corpus": "Medical" if embedding_name.startswith("Medical") else "Press",
                "Representation": embedding_name.split(" ", 1)[1],
                "Architecture": architecture,
                "Model": model,
                "History": history,
                **{f"Dev {k}": v for k, v in dev_metrics.items()},
            })

    coverage_df = pd.DataFrame(coverage_rows)
    dev_df = pd.DataFrame([{k: v for k, v in r.items() if k.startswith("Dev") or k in ["Embedding", "Source corpus", "Architecture"]} for r in static_runs])

    selected = []
    for embedding_name in EMBEDDING_PATHS:
        candidates = [r for r in static_runs if r["Embedding"] == embedding_name]
        selected.append(max(candidates, key=lambda r: r["Dev F1"]))

    static_test_rows, entity_reports = [], {}
    for run in selected:
        test_metrics, true_seq, pred_seq = evaluate_static(run["Model"], test_loader)
        static_test_rows.append({
            "Model": run["Embedding"],
            "Representation": run["Representation"],
            "Architecture": run["Architecture"],
            "Selection": f"{run['Architecture']} selected by dev F1={run['Dev F1']:.4f}",
            "Precision": test_metrics["Precision"],
            "Recall": test_metrics["Recall"],
            "F1": test_metrics["F1"],
        })
        entity_reports[run["Embedding"]] = (true_seq, pred_seq)
    static_test_df = pd.DataFrame(static_test_rows)

    bert_test_row = {
        "Model": "BERT multilingual cased",
        "Representation": "Contextual Transformer",
        "Architecture": "BERT token classifier",
        "Selection": "Epoch 2 selected by dev F1=0.6005",
        "Precision": 0.5998,
        "Recall": 0.5223,
        "F1": 0.5584,
    }
    final_df = pd.concat([static_test_df, pd.DataFrame([bert_test_row])], ignore_index=True)
    final_df = final_df.sort_values("F1", ascending=False).reset_index(drop=True)

    # persist tables
    coverage_df.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
    dev_df.to_csv(OUTPUT_DIR / "static_dev_runs.csv", index=False)
    static_test_df.to_csv(OUTPUT_DIR / "static_test_selected.csv", index=False)
    final_df.to_csv(OUTPUT_DIR / "final_nine_models.csv", index=False)
    json.dump({c: (r["P"], r["R"], r["F1"]) for c, r in final_df.set_index("Model")[["Precision", "Recall", "F1"]].iterrows()},
              open(OUTPUT_DIR / "final_nine_models.json", "w"), ensure_ascii=False, indent=2)

    # entity-type F1 for static models (BERT column reuses previous run in notebook)
    def entity_f1_report(true_sequences, pred_sequences):
        report = classification_report(true_sequences, pred_sequences, output_dict=True, mode="strict", scheme=IOB2, zero_division=0)
        return {e: v["f1-score"] for e, v in report.items() if e not in {"micro avg", "macro avg", "weighted avg"}}

    entity_scores = {name: entity_f1_report(ts, ps) for name, (ts, ps) in entity_reports.items()}
    pd.DataFrame(entity_scores).T.to_csv(OUTPUT_DIR / "entity_type_f1_static.csv")

    # figures
    dev_pivot = dev_df.pivot(index="Embedding", columns="Architecture", values="Dev F1")
    fig, ax = plt.subplots(figsize=(14, 5))
    dev_pivot.plot(kind="bar", ax=ax, color=["#4878CF", "#D65F5F"])
    ax.set_title("Development entity-level F1 for all 16 static runs")
    ax.set_xlabel("Embedding")
    ax.set_ylabel("Strict entity-level F1")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dev_16_runs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    plot_df = final_df.set_index("Model")[["Precision", "Recall", "F1"]]
    plot_df.plot(kind="bar", ax=ax, color=["#4878CF", "#6ACC65", "#D65F5F"])
    ax.set_title("Final test entity-level metrics (8 static + BERT)")
    ax.set_xlabel("Model and representation")
    ax.set_ylabel("Strict seqeval score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_nine_models.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n=== FINAL NINE-MODEL COMPARISON (test) ===")
    print(final_df[["Model", "Architecture", "Precision", "Recall", "F1"]].to_string(index=False))


if __name__ == "__main__":
    main()
