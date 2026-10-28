# Word2Vec to BERT (based on Glove Report)

Course practical on semantic representations: from static word embeddings (Word2Vec, FastText, GloVe) to a contextual Transformer (BERT), evaluated on French medical named entity recognition. 

## Highlights

- Experiment 1 — toy corpus: GloVe and SGNS implemented from scratch in numpy (AdaGrad, GloVe weighting f(x)=(x/x_max)^0.75); analogy accuracy GloVe 82.4% vs SGNS 58.8% after balancing per-word contexts.
- Experiment 2, step 1 — word embedding training: Word2Vec CBOW / Skip-gram, FastText CBOW and GloVe (train_glove.py) trained on the QUAERO FrenchMed & FrenchPress corpora.
- Experiment 2, step 2 — NER comparison: token-level NER with LSTM / CNN taggers on 8 static embeddings × 2 architectures (16 runs) plus a multilingual BERT baseline; strict seqeval entity-level F1.
- Training schedule fixed (20 epochs, patience 5 instead of 3/1): best static test F1 rose from 0.23 to 0.34 (Press FastText); GloVe is the best medical embedding (0.285); BERT leads at 0.558.
- Coursework project; corpus and model weights are not committed.

## Layout

```
word2vec-to-bert/
├── experiment1/                          # toy-corpus experiment
│   ├── 01_toy_corpus.py                  # toy corpus generation
│   ├── 02_glove_scratch.py               # GloVe from scratch
│   ├── 03_sgns_scratch.py                # SGNS from scratch (comparison)
│   ├── corpus.txt                        # toy corpus
│   └── figures/                          # loss, PCA, 3D trajectory, SGNS-PMI
└── experiment2/
    ├── step1_word_embeddings/            # train static embeddings
    │   ├── Word_Embeddings_Ran_Lu.ipynb  # word2vec CBOW/Skip-gram + FastText
    │   └── train_glove.py                # GloVe vectors on the QUAERO corpora
    └── step2_ner_comparison/             # NER with 8 embeddings + BERT
        ├── TP2_NER_Ran_Lu.ipynb          # 16 static runs + dev selection + final comparison
        ├── scripts/
        │   ├── reproduce_tp2_glove.py    # full 16-run reproduction
        │   ├── tp2_glove_extension.py    # GloVe-only runner (4 runs)
        │   ├── cnn_classification.py     # instructor script (adapted)
        │   └── transformers_classification.py  # instructor script (adapted)
        └── results/                      # final tables, figures, coverage
```

## Quick start

```bash
pip install -r experiment2/step2_ner_comparison/scripts/requirements.txt
# Experiment 1 — toy GloVe vs SGNS
python experiment1/01_toy_corpus.py && python experiment1/02_glove_scratch.py && python experiment1/03_sgns_scratch.py
# Experiment 2, step 1 — train embeddings
jupyter notebook experiment2/step1_word_embeddings/Word_Embeddings_Ran_Lu.ipynb
python experiment2/step1_word_embeddings/train_glove.py
# Experiment 2, step 2 — NER comparison
jupyter notebook experiment2/step2_ner_comparison/TP2_NER_Ran_Lu.ipynb
```

## Notes

- The QUAERO corpus (FrenchMed / FrenchPress .ospl and .conll files) comes from the course page and must sit next to the notebooks in `TP_ISD2020/`.
- Trained embedding weights (`models_word_vectors/`) are not committed: run step 1 to produce them.
- `cnn_classification.py` and `transformers_classification.py` are the instructor's original scripts (Sahar Ghannay, LISN) that the step-2 notebook adapts.
- Results in `results/` are from a fixed-seed CPU run (epochs 20, patience 5); the BERT row reuses a previous run (dev F1 0.6005, test F1 0.5584).
