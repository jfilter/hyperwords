# hyperwords: Hyperparameter-Enabled Word Representations

hyperwords is a collection of scripts and programs for creating word representations, designed to facilitate academic
research and prototyping of word representations. It allows you to tune many hyperparameters that are pre-set or
ignored in other word representation packages.

hyperwords is free and open software. If you use hyperwords in scientific publication, we would appreciate citations:  
"Improving Distributional Similarity with Lessons Learned from Word Embeddings"
Omer Levy, Yoav Goldberg, and Ido Dagan. TACL 2015.

## Requirements

Running hyperwords may require a lot of computational resources:

- disk space for independently pre-processing the corpus
- internal memory for loading sparse matrices
- significant running time; hyperwords is neither optimized nor multi-threaded

hyperwords assumes a \*nix shell, and requires Python 2.7 (or later, excluding 3+) with the following packages installed:
numpy, scipy, sparsesvd, docopt.

```bash
conda create --name hyperwords python=2.7
conda activate hyperwords
conda install -y -c conda-forge scipy
conda install -y -c anaconda cython docopt
pip install sparsesvd
```

## Quick-Start

1. Download the latest version from BitBucket, unzip, and make sure all scripts have running permissions (chmod 755 \*.sh).
2. Download a text corpus of your choice.
3. To create word vectors...
   - ...with SVD over PPMI, use: _corpus2svd.sh_
   - ...with SGNS (skip-grams with negative sampling), use: _corpus2sgns.sh_
4. The vectors should be available in textual format under <output_path>/vectors.txt

To explore the list of hyperparameters, use the _-h_ or _--help_ option.

## Pipeline

The following figure shows the hyperwords' pipeline:

**DATA:** raw corpus => corpus => pairs => counts => vocab  
**TRADITIONAL:** counts + vocab => pmi => svd  
**EMBEDDINGS:** pairs + vocab => sgns

**raw corpus => corpus**

- _scripts/clean_corpus.sh_
- Eliminates non-alphanumeric tokens from the original corpus.

**corpus => pairs**

- _corpus2pairs.py_
- Extracts a collection of word-context pairs from the corpus.

**pairs => counts**

- _scripts/pairs2counts.sh_
- Aggregates identical word-context pairs.

**counts => vocab**

- _counts2vocab.py_
- Creates vocabularies with the words' and contexts' unigram distributions.

**counts + vocab => pmi**

- _counts2pmi.py_
- Creates a PMI matrix (_scipy.sparse.csr_matrix_) from the counts.

**pmi => svd**

- _pmi2svd.py_
- Factorizes the PMI matrix using SVD. Saves the result as three dense numpy matrices.

**pairs + vocab => sgns**

- _word2vecf/word2vecf_
- An external program for creating embeddings with SGNS. For more information, see:  
  **"Dependency-Based Word Embeddings". Omer Levy and Yoav Goldberg. ACL 2014.**

An example pipeline is demonstrated in: _example_test.sh_

## Evaluation

hyperwords also allows easy evaluation of word representations on two tasks: word similarity and analogies.

### Word Similarity

- _hyperwords/ws_eval.py_
- Compares how a representation ranks pairs of related words by similarity versus human ranking.
- 5 readily-available datasets

### Analogies

- _hyperwords/analogy_eval.py_
- Solves analogy questions, such as: "man is to woman as king is to...?" (answer: queen).
- 2 readily-available datasets
- Shows results of two analogy recovery methods: 3CosAdd and 3CosMul. For more information, see:  
  **"Linguistic Regularities in Sparse and Explicit Word Representations". Omer Levy and Yoav Goldberg. CoNLL 2014.**

These programs assume that the representation was created by hyperwords, and can be loaded by
_hyperwords.representations.embedding.Embedding_. Dense vectors in textual format (such as the ones produced by word2vec
and GloVe) can be converted to hyperwords' format using _hyperwords/text2numpy.py_.
