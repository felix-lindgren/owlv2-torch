"""Evaluate OWLv2 on the Songdo drone traffic dataset (car/bus/truck/motorcycle).

Splits: test (default), train. 4K frames with ~51 vehicles each, so inference is
tiled by default; the train split is 4,335 images and wants --limit while iterating.
"""

from uav_eval import main


if __name__ == "__main__":
    main(default_dataset="songdo", description=__doc__)
