"""
kNN_Euclidean
=============

Euclidean k-NN drop-in for the TrIM/CSS experiment, mirroring the layout of
the Mondrian_RF package:

    Mondrian_RF/Mondrian_forest.py   <->   kNN_Euclidean/kNN.py
    Mondrian_RF/utils.py             <->   kNN_Euclidean/utils.py

Swapping the two imports at the top of css_utils from

    from Mondrian_RF.Mondrian_forest import TrIM
    from Mondrian_RF.utils import evaluate_all_lifetimes, two_one_norm

to

    from kNN_Euclidean.kNN import TrIM
    from kNN_Euclidean.utils import evaluate_all_lifetimes, two_one_norm

replaces ONLY the downstream regressor (Mondrian forest -> Euclidean k-NN).
The EGOP (H) estimation is unchanged -- see kNN_Euclidean/kNN.py.
"""