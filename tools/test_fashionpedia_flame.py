"""FLAME few-shot vs zero-shot OWLv2 on Fashionpedia (support from train, eval on val).

Same driver as ``test_dior_flame.py``; see it for ``--mode boxes`` / ``--mode active``.
"""

from test_dior_flame import main


if __name__ == "__main__":
    main(default_dataset="fashionpedia", description=__doc__)
