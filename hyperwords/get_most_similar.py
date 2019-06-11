from docopt import docopt
from scipy.stats.stats import spearmanr

from representations.representation_factory import create_representation


def main():
    args = docopt("""
    Usage:
        get_most_similar.py [options] <representation> <representation_path> <test_words>

    Options:
        --neg NUM    Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c        Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM    Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
    """)

    words = args['<test_words>'].split('_')
    representation = create_representation(args)
    for w in words:
        print w
        print representation.closest(w)


if __name__ == '__main__':
    main()
