# SensitiveAttributesReconstructionCorrector

This repository contains the implementation of the Reconstruction Corrector component introduced in our article "Exploiting Fairness to Enhance Sensitive Attributes Reconstruction"

File `reconstruction_tools_fairlearncomputation.py` contains the implementation of our contributions.

File `example_reconstruction_correction.py` provides an example use of our Python modules. More precisely, it performs the following tasks:

* It loads the UCI Adult Income dataset
* It trains a fair model using this dataset, and the ExponentiatedGradient fair learning technique from the `FairLearn` library.
* It trains a baseline adversary (corresponding to A' in our paper) using our `MachineLearningAttacker` object.
* It performs the confidence scores exponentiation process using our `scale_probas_manual` function.
* It performs the reconstruction correction using our `ReconstructionCorrector` object.
* It evaluates its quality (and the improvement brought by the correction step) using our `evaluate_reconstruction` function.