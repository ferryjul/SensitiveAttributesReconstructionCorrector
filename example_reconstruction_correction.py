import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import time
from reconstruction_tools_fairlearncomputation import * # Our work

# Script parameters
random_state_value = 42
script_verbosity = True
reconstruction_verbosity = 'Quiet'#'Quiet' #'Terse'

# Fairness constraint definition
epsilon = 0.0 # unfairness tolerance
metric = 'statistical_parity' # fairness metric (one of the keys of the dictionary below)
metrics = {'statistical_parity':1, 'predictive_equality':3, 'equal_opportunity':4, 'equalized_odds':5}

fairness_metric = metrics[metric]

# Define the data ratios
attacker_set_ratio = 0.33 
test_set_ratio = 0.33
train_set_ratio = 1.0 - attacker_set_ratio - test_set_ratio

# Load and prepare the UCI Adult Income dataset
dataset_id = 1590
categorical_columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical_columns=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
label_0 = '<=50K'
label_1 = '>50K'
sensitive_attr_column = 'sex'
protected_column = '%s_0.0' %sensitive_attr_column
unprotected_column = '%s_1.0' %sensitive_attr_column

X, y = fetch_openml(data_id=dataset_id, as_frame=True, return_X_y=True)
X.dropna(inplace=True) # Remove potential NAs
y = y[X.index.values]

## Transform categorical features to integer values
enc = OrdinalEncoder()
X_encoded = enc.fit_transform(X) ## encode all features
columnsList = X.columns.to_list()
for aCol in categorical_columns: ## but only use the encoding of categorical ones
    colIndex = columnsList.index(aCol)
    X[aCol] = X_encoded[:,colIndex]
    
X = pd.get_dummies(X, columns=[sensitive_attr_column])
y = y.replace([label_1], value=1)
y = y.replace([label_0], value=0)

# Split the dataset
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=1.0-train_set_ratio, random_state=random_state_value)
X_attack, X_test, y_attack, y_test = train_test_split(X_other, y_other, test_size=(test_set_ratio)/(test_set_ratio+attacker_set_ratio), random_state=random_state_value)

# Remove sensitive attribute (as it will not be used for inference)
X_train_non_sensitive = X_train.drop(protected_column, axis=1)
X_train_non_sensitive = X_train_non_sensitive.drop(unprotected_column, axis=1)

X_attack_non_sensitive = X_attack.drop(protected_column, axis=1)
X_attack_non_sensitive = X_attack_non_sensitive.drop(unprotected_column, axis=1)

X_test_non_sensitive = X_test.drop(protected_column, axis=1)
X_test_non_sensitive = X_test_non_sensitive.drop(unprotected_column, axis=1)

if script_verbosity:
    print("Data is split among: ")
    print("Training set: %d examples" %X_train.shape[0])
    print("Attack set: %d examples" %X_attack.shape[0])
    print("Test set: %d examples" %X_test.shape[0])
    print("Number of features for inference: ", X_train_non_sensitive.shape[1], "\n")
    print("Training target fair model...")

# Train a fair model using FairLearn ExponentiatedGradient
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, TruePositiveRateParity, FalsePositiveRateParity, EqualizedOdds
from fairlearn.metrics import MetricFrame, demographic_parity_difference, false_positive_rate, equalized_odds_difference, true_positive_rate_difference, true_negative_rate_difference
from sklearn.metrics import accuracy_score

## Define base learner
from sklearn.tree import DecisionTreeClassifier
clf_base = DecisionTreeClassifier(random_state=1+random_state_value, max_depth=8)

np.random.seed(random_state_value) # important for fairlearn models reproducibility! 

done = False
while not done:
    done = True
    ## Build the fairness constraint object
    if fairness_metric == 1:
        constraint = DemographicParity(difference_bound=epsilon)
    elif fairness_metric == 3:
        constraint = FalsePositiveRateParity(difference_bound=epsilon)
    elif fairness_metric == 4:
        constraint = TruePositiveRateParity(difference_bound=epsilon)
    elif fairness_metric == 5:
        constraint = EqualizedOdds(difference_bound=epsilon)
    else:
        raise ValueError("Metric %d currently not supported.\n" %fairness_metric)

    ## Build the fair classifier object
    clf = ExponentiatedGradient(clf_base, constraint)
    try:
        ## Perform the fair learning/training
        clf.fit(X_train_non_sensitive, y_train, sensitive_features=X_train[protected_column])
    except ValueError as e: # slightly relax constraint to automatically handle fairlearn numerical instability (known bug - see https://gitter.im/fairlearn/community?at=5ffe1a0099549911fc1a78d6)
        done = False
        print("Seed %d, Metric %d, Epsilon %.4f: ValueError raised during clf.fit(...)" %(rank, fairness_metric, epsilon))
        epsilon += 0.0001
        print("Increasing epsilon to %.4f" %epsilon)

## Get target fair model's predictions
y_train_preds = pd.DataFrame(clf.predict(X_train_non_sensitive))
y_test_preds = pd.DataFrame(clf.predict(X_test_non_sensitive))
y_train_preds_binary = np.asarray(y_train_preds).flatten() # just a different format to work with our code
y_test_preds_binary = np.asarray(y_test_preds).flatten() # just a different format to work with our code

## Compute train unfairness using Fairlearn
if fairness_metric == 1:
    train_unf_fairlearn = demographic_parity_difference(y_train, y_train_preds, method='to_overall', sensitive_features=X_train[protected_column])
elif fairness_metric == 3:
    train_unf_fairlearn = true_negative_rate_difference(y_train, y_train_preds, method='to_overall', sensitive_features=X_train[protected_column])
elif fairness_metric == 4:
    train_unf_fairlearn = true_positive_rate_difference(y_train, y_train_preds, method='to_overall', sensitive_features=X_train[protected_column])
elif fairness_metric == 5:
    train_unf_fairlearn = equalized_odds_difference(y_train, y_train_preds, method='to_overall', sensitive_features=X_train[protected_column])
else:
    raise ValueError("Metric %d currently not supported.\n" %fairness_metric)

if script_verbosity :
    print("Trained!")
    print("Train unfairness (%s) = " %metric, train_unf_fairlearn)
    accTraining = accuracy_score(y_train, y_train_preds)
    accTest = accuracy_score(y_test, y_test_preds) 
    print("Train accuracy: ", accTraining)
    print("Test accuracy: ", accTest, "\n")

## Trick: if fairness constraint is not met on the training set due to the ExponentiatedGradient process, we set up a new constraint that would be met given the trained model
while round(epsilon, 4) < np.fabs(train_unf_fairlearn):
    epsilon += 0.001

# Baseline attacker initial reconstruction (adversary A')
## Prepare attack set
X_attack_complete = pd.concat([X_attack_non_sensitive, y_attack], axis=1)
attack_preds = clf.predict(X_attack_non_sensitive)
X_attack_complete['pred'] = attack_preds
X_train_complete = pd.concat([X_train_non_sensitive, y_train], axis=1)
train_preds = clf.predict(X_train_non_sensitive)
X_train_complete['pred'] = train_preds
X_test_complete = pd.concat([X_test_non_sensitive, y_test], axis=1)
test_preds = clf.predict(X_test_non_sensitive)
X_test_complete['pred'] = test_preds

## Use our MachineLearningAttacker wrapper
if script_verbosity :
    print("Training baseline adversary...")

attacker_machine_learning = MachineLearningAttacker(y_train.size, verbose=script_verbosity)

## Define attack model parameters
from sklearn.ensemble import RandomForestClassifier
ml_model = RandomForestClassifier(random_state=2+random_state_value, class_weight="balanced", max_depth=6,  min_samples_split=10) 

attacker_machine_learning.fit(ml_model, X_attack_complete, X_attack[protected_column]) # train attack model on the attack set

if script_verbosity :
    print("Trained!")
    print("Computing the probabilities exponentiation factor value...")

## Use a separate dataset (in practice, X_test) to compute the best exponentiation factor to scale the attacker probabilities
d_i_attacker_val = attacker_machine_learning.predict(X_test_complete)
d_i_attacker_probas_val = attacker_machine_learning.predict_proba(X_test_complete)

best_reconstruction = -1
expo_factor_list = []
reconstr_perf_list = []
for expo_factor_search in range(0, 100):
    d_i_attacker_probas_val_exp = scale_probas_manual(d_i_attacker_probas_val, expo_factor_search)
    reconstruction_corrector = ReconstructionCorrector(verbosity='Quiet')
    if fairness_metric in [1,3,4]:
        status = reconstruction_corrector.fit(d_i_attacker_val, np.asarray(y_test), y_test_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas_val_exp, t_out=60, mode='constraint', fairness_metric=fairness_metric)
    elif fairness_metric == 5:
        status1 = reconstruction_corrector.fit(d_i_attacker_val, np.asarray(y_test), y_test_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas_val_exp, t_out=60, mode='constraint', fairness_metric=3)
        corrected_preds = reconstruction_corrector.predict()
        reconstruction_corrector = ReconstructionCorrector(verbosity=reconstruction_verbosity)
        status2 = reconstruction_corrector.fit(corrected_preds, np.asarray(y_test), y_test_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas_val_exp, t_out=60, mode='constraint', fairness_metric=4)
    corrected_preds = reconstruction_corrector.predict()
    corrected_perf = evaluate_reconstruction(corrected_preds, X_test[protected_column])
    if corrected_perf > best_reconstruction:
        expo_factor_value = expo_factor_search
        best_reconstruction = corrected_perf
    expo_factor_list.append(expo_factor_search)
    reconstr_perf_list.append(corrected_perf)
if script_verbosity:
    print("Exponentiating all probabilities to factor ", expo_factor_value, "\n")
    print("Performing the reconstruction correction...")

d_i_attacker = attacker_machine_learning.predict(X_train_complete)
d_i_attacker_probas = attacker_machine_learning.predict_proba(X_train_complete)

## Finally do the probabilities exponentiation
d_i_attacker_probas = scale_probas_manual(d_i_attacker_probas, expo_factor_value)

# Perform the reconstruction correction
start = time.time()
reconstruction_corrector = ReconstructionCorrector(verbosity=reconstruction_verbosity)
if fairness_metric == 1:
    status = reconstruction_corrector.fit(d_i_attacker, np.asarray(y_train), y_train_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas, t_out=60, fairness_metric=1)
    cpo_train_unf = status[2]
elif fairness_metric == 2:
    raise ValueError("Predictive Parity currently not supported.\n")
elif fairness_metric == 3:
    status = reconstruction_corrector.fit(d_i_attacker, np.asarray(y_train), y_train_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas, t_out=60, fairness_metric=3)
    cpo_train_unf = status[2]
elif fairness_metric == 4:
    status = reconstruction_corrector.fit(d_i_attacker, np.asarray(y_train), y_train_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas, t_out=60, fairness_metric=4)
    cpo_train_unf = status[2]
elif fairness_metric == 5:
    status1 = reconstruction_corrector.fit(d_i_attacker, np.asarray(y_train), y_train_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas, t_out=60, fairness_metric=3)
    corrected_preds = reconstruction_corrector.predict()
    reconstruction_corrector = ReconstructionCorrector(verbosity=reconstruction_verbosity)
    status2 = reconstruction_corrector.fit(corrected_preds, np.asarray(y_train), y_train_preds_binary, epsilon=epsilon, probas=d_i_attacker_probas, t_out=60, fairness_metric=4)
corrected_preds = reconstruction_corrector.predict()
end = time.time()

# Evaluate the reconstructions (original Ŝ and corrected S*)
corrected_perf = evaluate_reconstruction(corrected_preds, X_train[protected_column])
old_perf = evaluate_reconstruction(d_i_attacker, X_train[protected_column])

if script_verbosity:
    print("Done! \n")

print("Corrected reconstruction: ", corrected_perf, " (was ", old_perf, " before)")
print("Absolute improvement = ", (corrected_perf-old_perf))
print("Relative improvement = ", (corrected_perf-old_perf)/old_perf)