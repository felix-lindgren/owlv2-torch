"""Evaluate OWLv2 on the WiSARD Mt Erie sample (wilderness search and rescue).

Streams: ir (default, 640x512 thermal) and vis (3840x2160 visual). One class,
"human", but the two streams are separate benchmarks - boxes are drawn per
modality and the same person is ~17 px in IR against ~67 px in VIS.
"""

from uav_eval import main


if __name__ == "__main__":
    main(default_dataset="wisard", description=__doc__)
