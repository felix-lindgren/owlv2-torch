"""Evaluate OWLv2 on MAVREC (drone and ground views of the same scenes).

Splits: aerial-val (default), ground-val, aerial-train-subset, ground-train-subset.
The full aerial_train split is annotated but its images are not on disk locally.
"""

from uav_eval import main


if __name__ == "__main__":
    main(default_dataset="mavrec", description=__doc__)
