# Word2Vec to BERT — French Medical Named Entity Recognition

Two practicals on semantic representations: from static word embeddings (Word2Vec, FastText, GloVe) to a contextual Transformer (BERT), evaluated on French medical named entity recognition.

## Highlights

- Experiment 1: train static embeddings (Word2Vec CBOW / Skip-gram, FastText CBOW) on the QUAERO FrenchMed and FrenchPress corpora; GloVe implemented from scratch (numpy + AdaGrad) on a toy corpus, compared against SGNS.
- Experiment 2: token-level NER with LSTM / CNN taggers on 8 static embeddings (6 TP1 models + 2 GloVe) plus a multilingual BERT baseline; strict seqeval entity-level F1.
- Fixed training schedule (20 epochs, patience 5 instead of 3/1): best static test F1 rose from 0.23 to 0.34 (Press FastText); GloVe is the best medical embedding (0.285); BERT leads at 0.558.
- Coursework project; data and model weights are intentionally not committed.

## Layout

```
word2vec-to-bert/
├── experiment1/                  # TP1: static word embeddings
│   ├── Word_Embeddings_Ran_Lu.ipynb   # word2vec + fasttext training & analysis
│   └── glove-lab/                # GloVe from scratch + SGNS comparison (toy corpus)
└── experiment2/                  # TP2: medical NER with 8 embeddings + BERT
    ├── TP2_NER_Ran_Lu.ipynb      # 16 static runs + dev selection + final comparison
    ├── scripts/
    │   ├── train_glove.py            # train GloVe vectors on the QUAERO corpora
    │   ├── reproduce_tp2_glove.py    # full 16-run reproduction
    │   ├── tp2_glove_extension.py    # GloVe-only runner (4 runs) merged into the table
    │   ├── cnn_classification.py     # instructor script (adapted in TP2)
    │   └── transformers_classification.py  # instructor script (adapted in TP2)
    └── results/                  # final tables, figures, coverage
```

## Quick start

```bash
pip install -r experiment2/scripts/requirements.txt
# TP1: static embeddings
jupyter notebook experiment1/Word_Embeddings_Ran_Lu.ipynb
# TP2: NER with 8 embeddings + BERT
jupyter notebook experiment2/TP2_NER_Ran_Lu.ipynb
```

## Notes

- The QUAERO corpus (FrenchMed / FrenchPress .ospl and .conll files) comes from the course page (https://saharghannay.github.io/courses/cours1/) and must sit next to the notebooks in `TP_ISD2020/`.
- Trained embedding weights (`models_word_vectors/`) are not committed: run TP1 to produce them, or train GloVe with `experiment2/scripts/train_glove.py`.
- `experiment2/scripts/cnn_classification.py` and `transformers_classification.py` are the instructor's original scripts (Sahar Ghannay, LISN) that the TP2 notebook adapts.
- Results in `experiment2/results/` are from a fixed-seed CPU run (epochs 20, patience 5); BERT row reuses the previous run (dev F1 0.6005, test F1 0.5584).
