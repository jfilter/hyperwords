#!/bin/sh
set -e
set -x

out_folder=example

# for word2vec
threads=4

# sometimes you don't wanna run with all the data
limit_rows=1000000

# disable word2vec (use use GenSim?)
word2vec=0

mkdir -p $out_folder


# Download and install word2vecf
if [ $word2vec != 0 ]; then
    if [ ! -f word2vecf ]; then
        scripts/install_word2vecf.sh
    fi
fi

if ! [ $1 = "evaluate" ]; then

    corpus=news.2010.en.shuffled
    # Download corpus. We chose a small corpus for the $out_folder, and larger corpora will yield better results.
    if test -f "$out_folder/$corpus_clean"; then
        echo 'file exists, skip download'
    else
        wget -nc http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
        cp news.2010.en.shuffled.gz.1 news.2010.en.shuffled.gz
        gzip -d $corpus.gz

        if [ "$limit_rows" -gt "0" ]; then
            head -n $limit_rows $corpus > $corpus.head
            mv $corpus.head $corpus
        fi

        # Clean the corpus from non alpha-numeric symbols
        scripts/clean_corpus.sh $corpus > $corpus.clean
        mv $corpus.clean $out_folder/$corpus.clean
        # rm $corpus
    fi


    # Create two $out_folder collections of word-context pairs:

    # A) Window size 2 with "clean" subsampling
    mkdir -p $out_folder/w2_sub
    python hyperwords/corpus2pairs.py --win 2 --sub 1e-5 $out_folder/${corpus}.clean > $out_folder/w2_sub/pairs
    scripts/pairs2counts.sh $out_folder/w2_sub/pairs > $out_folder/w2_sub/counts
    python hyperwords/counts2vocab.py $out_folder/w2_sub/counts

    # B) Window size 5 with dynamic contexts and "dirty" subsampling
    mkdir -p $out_folder/w5_dyn_sub_del
    python hyperwords/corpus2pairs.py --win 5 --dyn --sub 1e-5 --del $out_folder/${corpus}.clean > $out_folder/w5_dyn_sub_del/pairs
    scripts/pairs2counts.sh $out_folder/w5_dyn_sub_del/pairs > $out_folder/w5_dyn_sub_del/counts
    python hyperwords/counts2vocab.py $out_folder/w5_dyn_sub_del/counts

    # Calculate PMI matrices for each collection of pairs
    python hyperwords/counts2pmi.py --cds 0.75 $out_folder/w2_sub/counts $out_folder/w2_sub/pmi
    python hyperwords/counts2pmi.py --cds 0.75 $out_folder/w5_dyn_sub_del/counts $out_folder/w5_dyn_sub_del/pmi


    # Create embeddings with SVD
    python hyperwords/pmi2svd.py --dim 500 --neg 5 $out_folder/w2_sub/pmi $out_folder/w2_sub/svd
    cp $out_folder/w2_sub/pmi.words.vocab $out_folder/w2_sub/svd.words.vocab
    cp $out_folder/w2_sub/pmi.contexts.vocab $out_folder/w2_sub/svd.contexts.vocab
    python hyperwords/pmi2svd.py --dim 500 --neg 5 $out_folder/w5_dyn_sub_del/pmi $out_folder/w5_dyn_sub_del/svd
    cp $out_folder/w5_dyn_sub_del/pmi.words.vocab $out_folder/w5_dyn_sub_del/svd.words.vocab
    cp $out_folder/w5_dyn_sub_del/pmi.contexts.vocab $out_folder/w5_dyn_sub_del/svd.contexts.vocab

    if [ $word2vec != 0 ]; then
        # Create embeddings with SGNS (A). Commands 2-5 are necessary for loading the vectors with embeddings.py
        word2vecf/word2vecf -train $out_folder/w2_sub/pairs -pow 0.75 -cvocab $out_folder/w2_sub/counts.contexts.vocab -wvocab $out_folder/w2_sub/counts.words.vocab -dumpcv $out_folder/w2_sub/sgns.contexts -output $out_folder/w2_sub/sgns.words -threads $threads  -negative 15 -size 500;
        python hyperwords/text2numpy.py $out_folder/w2_sub/sgns.words
        rm $out_folder/w2_sub/sgns.words
        python hyperwords/text2numpy.py $out_folder/w2_sub/sgns.contexts
        rm $out_folder/w2_sub/sgns.contexts

        # Create embeddings with SGNS (B). Commands 2-5 are necessary for loading the vectors with embeddings.py
        word2vecf/word2vecf -train $out_folder/w5_dyn_sub_del/pairs -pow 0.75 -cvocab $out_folder/w5_dyn_sub_del/counts.contexts.vocab -wvocab $out_folder/w5_dyn_sub_del/counts.words.vocab -dumpcv $out_folder/w5_dyn_sub_del/sgns.contexts -output $out_folder/w5_dyn_sub_del/sgns.words -threads $threads -negative 15 -size 500;
        python hyperwords/text2numpy.py $out_folder/w5_dyn_sub_del/sgns.words
        rm $out_folder/w5_dyn_sub_del/sgns.words
        python hyperwords/text2numpy.py $out_folder/w5_dyn_sub_del/sgns.contexts
        rm $out_folder/w5_dyn_sub_del/sgns.contexts
    fi
fi

# Evaluate on Word Similarity
echo
echo "WS353 Results"
echo "-------------"

export PYTHONWARNINGS="ignore"
set +x

python hyperwords/ws_eval.py --neg 5 PPMI $out_folder/w2_sub/pmi testsets/en/ws/ws353.txt
python hyperwords/ws_eval.py --eig 0.5 SVD $out_folder/w2_sub/svd testsets/en/ws/ws353.txt

if [ $word2vec != 0 ]; then
    python hyperwords/ws_eval.py --w+c SGNS $out_folder/w2_sub/sgns testsets/en/ws/ws353.txt
fi

python hyperwords/ws_eval.py --neg 5 PPMI $out_folder/w5_dyn_sub_del/pmi testsets/en/ws/ws353.txt
python hyperwords/ws_eval.py --eig 0.5 SVD $out_folder/w5_dyn_sub_del/svd testsets/en/ws/ws353.txt

if [ $word2vec != 0 ]; then
    python hyperwords/ws_eval.py --w+c SGNS $out_folder/w5_dyn_sub_del/sgns testsets/en/ws/ws353.txt
fi

# Evaluate on Analogies
echo
echo "Google Analogy Results"
echo "----------------------"

python hyperwords/analogy_eval.py PPMI $out_folder/w2_sub/pmi testsets/en/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD $out_folder/w2_sub/svd testsets/en/analogy/google.txt

if [ $word2vec != 0 ]; then
    python hyperwords/analogy_eval.py SGNS $out_folder/w2_sub/sgns testsets/en/analogy/google.txt
fi

python hyperwords/analogy_eval.py PPMI $out_folder/w5_dyn_sub_del/pmi testsets/en/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD $out_folder/w5_dyn_sub_del/svd testsets/en/analogy/google.txt

if [ $word2vec != 0 ]; then
    python hyperwords/analogy_eval.py SGNS $out_folder/w5_dyn_sub_del/sgns testsets/en/analogy/google.txt
fi
