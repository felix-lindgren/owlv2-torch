"""Train per-class FLAME refiners on Fashionpedia train and evaluate on val.

This is the Fashionpedia entry point for the same few-shot pipeline used by
``test_dior_flame.py``. Run with ``--classes`` during iteration; evaluating all
46 categories trains and evaluates a separate refiner for every category.
"""

from test_dior_flame import main


if __name__ == "__main__":
    main(default_dataset="fashionpedia", description=__doc__)
