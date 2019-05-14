#!/bin/sh
set -e
set -x

mkdir -p example

if ! [ $1 = "evaluate" ]; then

# Download and install word2vecf
if [ ! -f word2vecf ]; then
    scripts/install_word2vecf.sh
fi


CORPUS=news.2010.en.shuffled
# Download corpus. We chose a small corpus for the example, and larger corpora will yield better results.
if test -f "example/$CORPUS_clean"; then
    echo 'file exists, skip download'
else
    wget -nc http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
    gzip -d $CORPUS.gz
		# Clean the corpus from non alpha-numeric symbols
		scripts/clean_corpus.sh $CORPUS > $CORPUS.clean
		mv $CORPUS.clean example/$CORPUS.clean
		rm $CORPUS
fi


# Create two example collections of word-context pairs:

# A) Window size 2 with "clean" subsampling
mkdir -p example/w2_sub
python hyperwords/corpus2pairs.py --win 2 --sub 1e-5 example/${CORPUS}.clean > example/w2_sub/pairs
scripts/pairs2counts.sh example/w2_sub/pairs > example/w2_sub/counts
python hyperwords/counts2vocab.py example/w2_sub/counts

# B) Window size 5 with dynamic contexts and "dirty" subsampling
mkdir -p example/w5_dyn_sub_del
python hyperwords/corpus2pairs.py --win 5 --dyn --sub 1e-5 --del example/${CORPUS}.clean > example/w5_dyn_sub_del/pairs
scripts/pairs2counts.sh example/w5_dyn_sub_del/pairs > example/w5_dyn_sub_del/counts
python hyperwords/counts2vocab.py example/w5_dyn_sub_del/counts

# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py --cds 0.75 example/w2_sub/counts example/w2_sub/pmi
python hyperwords/counts2pmi.py --cds 0.75 example/w5_dyn_sub_del/counts example/w5_dyn_sub_del/pmi


# Create embeddings with SVD
python hyperwords/pmi2svd.py --dim 500 --neg 5 example/w2_sub/pmi example/w2_sub/svd
cp example/w2_sub/pmi.words.vocab example/w2_sub/svd.words.vocab
cp example/w2_sub/pmi.contexts.vocab example/w2_sub/svd.contexts.vocab
python hyperwords/pmi2svd.py --dim 500 --neg 5 example/w5_dyn_sub_del/pmi example/w5_dyn_sub_del/svd
cp example/w5_dyn_sub_del/pmi.words.vocab example/w5_dyn_sub_del/svd.words.vocab
cp example/w5_dyn_sub_del/pmi.contexts.vocab example/w5_dyn_sub_del/svd.contexts.vocab


# Create embeddings with SGNS (A). Commands 2-5 are necessary for loading the vectors with embeddings.py
word2vecf/word2vecf -train example/w2_sub/pairs -pow 0.75 -cvocab example/w2_sub/counts.contexts.vocab -wvocab example/w2_sub/counts.words.vocab -dumpcv example/w2_sub/sgns.contexts -output example/w2_sub/sgns.words -threads 10 -negative 15 -size 500;
python hyperwords/text2numpy.py example/w2_sub/sgns.words
rm example/w2_sub/sgns.words
python hyperwords/text2numpy.py example/w2_sub/sgns.contexts
rm example/w2_sub/sgns.contexts

# Create embeddings with SGNS (B). Commands 2-5 are necessary for loading the vectors with embeddings.py
word2vecf/word2vecf -train example/w5_dyn_sub_del/pairs -pow 0.75 -cvocab example/w5_dyn_sub_del/counts.contexts.vocab -wvocab example/w5_dyn_sub_del/counts.words.vocab -dumpcv example/w5_dyn_sub_del/sgns.contexts -output example/w5_dyn_sub_del/sgns.words -threads 10 -negative 15 -size 500;
python hyperwords/text2numpy.py example/w5_dyn_sub_del/sgns.words
rm example/w5_dyn_sub_del/sgns.words
python hyperwords/text2numpy.py example/w5_dyn_sub_del/sgns.contexts
rm example/w5_dyn_sub_del/sgns.contexts

fi

# Evaluate on Word Similarity
echo
echo "WS353 Results"
echo "-------------"

export PYTHONWARNINGS="ignore"
set +x

python hyperwords/ws_eval.py --neg 5 PPMI example/w2_sub/pmi testsets/ws/ws353.txt
python hyperwords/ws_eval.py --eig 0.5 SVD example/w2_sub/svd testsets/ws/ws353.txt
python hyperwords/ws_eval.py --w+c SGNS example/w2_sub/sgns testsets/ws/ws353.txt

python hyperwords/ws_eval.py --neg 5 PPMI example/w5_dyn_sub_del/pmi testsets/ws/ws353.txt
python hyperwords/ws_eval.py --eig 0.5 SVD example/w5_dyn_sub_del/svd testsets/ws/ws353.txt
python hyperwords/ws_eval.py --w+c SGNS example/w5_dyn_sub_del/sgns testsets/ws/ws353.txt


# Evaluate on Analogies
echo
echo "Google Analogy Results"
echo "----------------------"

python hyperwords/analogy_eval.py PPMI example/w2_sub/pmi testsets/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD example/w2_sub/svd testsets/analogy/google.txt
python hyperwords/analogy_eval.py SGNS example/w2_sub/sgns testsets/analogy/google.txt

python hyperwords/analogy_eval.py PPMI example/w5_dyn_sub_del/pmi testsets/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD example/w5_dyn_sub_del/svd testsets/analogy/google.txt
python hyperwords/analogy_eval.py SGNS example/w5_dyn_sub_del/sgns testsets/analogy/google.txt
