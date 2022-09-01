# SensitiveAttributesReconstructionCorrector

This repository contains the implementation of the Reconstruction Corrector component introduced in our article "Exploiting Fairness to Enhance Sensitive Attributes Reconstruction"

File `reconstruction_tools_fairlearncomputation.py` contains the implementation of our contributions.

File `example_reconstruction_correction.py` provides an example use of our Python modules. More precisely, it performs the following tasks:

* It loads the UCI Adult Income dataset
* It trains a fair model using this dataset, and the ExponentiatedGradient fair learning technique from the `FairLearn` library.
* It trains a baseline adversary (corresponding to A' in our paper) using our `MachineLearningAttacker` object.
* It performs the adversary's confidence scores exponentiation process using our `scale_probas_manual` function.
* It performs the reconstruction correction using our `ReconstructionCorrector` object.
* It evaluates its quality (and the improvement brought by the correction step) using our `evaluate_reconstruction` function.

# Requirements

* Our `ReconstructionCorrector` object implements the reconstruction correction method described in our article. It builds and solves a Constraint Programming (CP) model, using the [Docplex](https://pypi.org/project/docplex/) Python modelling tool and the [IBM CP Optimizer](https://www.ibm.com/fr-fr/analytics/cplex-cp-optimizer) solver. Both are required for the `ReconstructionCorrector` to be able to run.
* Our example script uses the ExponentiatedGradient fair learning technique from the `FairLearn` library. To run it, [FairLearn](https://fairlearn.org/) must be installed.
* Our example script uses several tools from the `scikit-learn` library. To run it, [scikit-learn](https://scikit-learn.org/stable/) must be installed.
* Other popular required libraries: `numpy`, `pandas`

# Example output

Hereafter is an example output produced by our example script for the statistical parity metric with unfairness tolerance epsilon = 0.0 (random_state_value=42):

```console
Data is split among: 
Training set: 15375 examples
Attack set: 14923 examples
Test set: 14924 examples
Number of features for inference:  13 

Training target fair model...
Trained!
Train unfairness (statistical_parity) =  1.333702367489753e-06
Train accuracy:  0.8394146341463414
Test accuracy:  0.8308094344679711 

Training baseline adversary...
Machine learning attacker ready. Accuracy on attack set is  0.7639884741673926
Trained!
Computing the probabilities exponentiation factor value...
Exponentiating all probabilities to factor  6 

Performing the reconstruction correction...
Done! 

Corrected reconstruction:  0.8300487804878048  (was  0.7654634146341464  before)
Absolute improvement =  0.06458536585365848
Relative improvement =  0.08437420341575318
```