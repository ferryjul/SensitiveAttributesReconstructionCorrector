import numpy as np
import pandas as pd

# Wrapper for our baseline adversaries
class MachineLearningAttacker:
    def __init__(self, shape, verbose=True):
        self.shape = shape
        self.verbose = verbose

    def fit(self, model, X_non_sensitive, X_sensitive, fit_args = {}):
        self.model = model
        self.model.fit(X_non_sensitive, X_sensitive, **fit_args)        
        perf = self.model.score(X_non_sensitive, X_sensitive)
        if self.verbose:
            print("Machine learning attacker ready. Accuracy on attack set is ", perf)
        return perf

    def get_model(self):
        return self.model

    # Uses the fitted ML model
    def predict(self, X_non_sensitive):
        return self.model.predict(X_non_sensitive)

    # Get probabilities for all predictions
    def predict_proba(self, X_non_sensitive):
        try:
            all_probas = self.model.predict_proba(X_non_sensitive)
        except AttributeError:
            all_probas = self.model._best_learner.predict_proba(X_non_sensitive)
        for a_pred in all_probas:
            if max(a_pred) <= 10e-5:
                print(a_pred)
        return [max(a_pred) for a_pred in all_probas]

    # Returns the percent of good reconstruction
    # Half of the values should be correctly identified
    def evaluate_reconstruction(self, X_non_sensitive, ground_truth):
        preds = self.predict(X_non_sensitive)
        qual = np.sum(preds == ground_truth) / self.shape
        return qual

# Reconstruction corrector for fairness
class ReconstructionCorrector:
    def __init__(self, verbosity):
        self.fitted = False
        self.verbosity = verbosity
    
    def fit(self, d_i_array, y_i_array, y_i_preds_array, epsilon, probas=None, t_out=60, mode='constraint', fairness_metric=1, proportions_estimations=None, proportions_tolerence=None, constraint_expr='to_overall'):
        if y_i_array.shape == (y_i_preds_array.shape[0],1): # quick fix
            y_i_array = y_i_array.flatten()
        if not fairness_metric in [1, 3, 4]:
            raise ValueError("fairness_metric must be an integer in {1, 3, 4}, got: ", fairness_metric)
        def concerned_by_metric(metric, true_y):
            if metric == 1:
                return True
            elif metric == 3:
                return (true_y == 0)
            elif metric == 4:
                return (true_y == 1)

        if probas is None:
            # Uniform probability
            probas = np.full(d_i_array.shape, 1)

        # sorted arrays with cumulated probabilities for the 4 types of examples
        d_0_y_pred_0 = []
        d_0_y_pred_1 = []
        d_1_y_pred_0 = []
        d_1_y_pred_1 = []

        # indices of the examples (corresponding to the 4 previous arrays)
        d_0_y_pred_0_indexes = []
        d_0_y_pred_1_indexes = []
        d_1_y_pred_0_indexes = []
        d_1_y_pred_1_indexes = []

        # fill in the 4 arrays (after this loop, they are not yet sorted and the probas are not cumulated)
        for i in range(d_i_array.size):
            if y_i_preds_array[i] == 0 and d_i_array[i] == 0 and concerned_by_metric(fairness_metric, y_i_array[i]):
                d_0_y_pred_0.append(probas[i])
                d_0_y_pred_0_indexes.append(i)
            elif y_i_preds_array[i] == 1 and d_i_array[i] == 1 and concerned_by_metric(fairness_metric, y_i_array[i]):
                d_1_y_pred_1.append(probas[i])
                d_1_y_pred_1_indexes.append(i)
            elif y_i_preds_array[i] == 1 and d_i_array[i] == 0 and concerned_by_metric(fairness_metric, y_i_array[i]):
                d_0_y_pred_1.append(probas[i])
                d_0_y_pred_1_indexes.append(i)
            elif y_i_preds_array[i] == 0 and d_i_array[i] == 1 and concerned_by_metric(fairness_metric, y_i_array[i]):
                d_1_y_pred_0.append(probas[i])
                d_1_y_pred_0_indexes.append(i)
            
        def process_indices_probas(probas, indices):
            arr1inds = np.argsort(probas)
            indices = [indices[i] for i in arr1inds]
            probas.append(0.0) # add a first virtual example with probability 0 (corresponding to no change at all for this type of examples)
            probas = np.sort(probas) # sort the probabilities
            probas = np.cumsum(probas) # cumulate them
            # upscale and round to avoid useless overprecision and float computations
            '''
            n_decimals = 1
            n_digits_to_keep = 1
            probas = probas * (10**n_digits_to_keep)
            probas = np.round(probas, decimals=n_decimals)
            '''
            return probas, indices

        d_0_y_pred_0, d_0_y_pred_0_indexes = process_indices_probas(d_0_y_pred_0, d_0_y_pred_0_indexes)
        d_1_y_pred_0, d_1_y_pred_0_indexes = process_indices_probas(d_1_y_pred_0, d_1_y_pred_0_indexes)
        d_0_y_pred_1, d_0_y_pred_1_indexes = process_indices_probas(d_0_y_pred_1, d_0_y_pred_1_indexes)
        d_1_y_pred_1, d_1_y_pred_1_indexes = process_indices_probas(d_1_y_pred_1, d_1_y_pred_1_indexes)

        # get initial group's cardinalities (to compute updated ones based on the made changes)
        n_0_plus = len(d_0_y_pred_1)-1
        n_0_minus = len(d_0_y_pred_0)-1
        n_1_plus = len(d_1_y_pred_1)-1
        n_1_minus = len(d_1_y_pred_0)-1

        tot_examples = n_0_plus + n_0_minus + n_1_plus + n_1_minus
        if self.verbosity != 'Quiet':
            print("Metric %d concerns %d/%d examples (%.2f)." %(fairness_metric, tot_examples, y_i_array.shape[0],  tot_examples/y_i_array.shape[0]))
        
        if fairness_metric == 1:
            assert(tot_examples == y_i_preds_array.size)
        
        # Import CPO solver library
        from docplex.cp.model import CpoModel

        # Build model
        model = CpoModel()

        # Variables
        s_0_1_plus = model.integer_var(0, n_0_plus, 's_0_1_plus')
        s_0_1_minus = model.integer_var(0, n_0_minus, 's_0_1_minus')
        s_1_0_plus = model.integer_var(0, n_1_plus, 's_1_0_plus')
        s_1_0_minus = model.integer_var(0, n_1_minus, 's_1_0_minus')

        s_nb_0_plus = model.integer_var(0, n_0_plus + n_1_plus, 's_nb_0_plus')
        model.add(s_nb_0_plus == n_0_plus - s_0_1_plus + s_1_0_plus)
        s_nb_1_plus = model.integer_var(0, n_0_plus + n_1_plus, 's_nb_1_plus')
        model.add(s_nb_1_plus == n_1_plus - s_1_0_plus + s_0_1_plus)
        s_nb_0_minus = model.integer_var(0, n_0_minus + n_1_minus, 's_nb_0_minus')
        model.add(s_nb_0_minus == n_0_minus - s_0_1_minus + s_1_0_minus)
        s_nb_1_minus = model.integer_var(0, n_0_minus + n_1_minus, 's_nb_1_minus')
        model.add(s_nb_1_minus == n_1_minus - s_1_0_minus + s_0_1_minus)

        #s_nb_0 = model.integer_var(1, y_i_preds_array.size - 1, 's_nb_0')
        s_nb_0 = model.integer_var(1, tot_examples - 1, 's_nb_0')
        model.add(s_nb_0 == s_nb_0_minus + s_nb_0_plus)
        #s_nb_1 = model.integer_var(1, y_i_preds_array.size - 1, 's_nb_1')
        s_nb_1 = model.integer_var(1, tot_examples - 1, 's_nb_1')
        model.add(s_nb_1 == s_nb_1_minus + s_nb_1_plus)

          # Proportions constraint (optional)
        if not(proportions_estimations is None) and not(proportions_tolerence is None) and False:
            #print("proportions_estimations=", proportions_estimations, "+-", 100*proportions_tolerence, "percent.")
            proportion_ub = proportions_estimations*(1.0+proportions_tolerence)
            proportion_lb = proportions_estimations*(1.0-proportions_tolerence)
            #print("reconstruction proportion must be in [%.3f,%.3f]" %(proportion_lb, proportion_ub))
            model.add(s_nb_0 <= proportion_ub * s_nb_1)
            model.add( proportion_lb * s_nb_1 <= s_nb_0)
        
        
        # Fairness constraint
        if (fairness_metric == 1 or fairness_metric == 3 or fairness_metric == 4):
            fairness_constraint = True
            # Statistical Parity
            if constraint_expr=='to_overall':
                if fairness_metric == 1:
                    global_prop = np.sum(y_i_preds_array)/y_i_preds_array.size
                elif fairness_metric == 3:
                    negative_slice = np.where(y_i_array == 0)
                    global_prop = np.sum(y_i_preds_array[negative_slice])/y_i_preds_array[negative_slice].size
                elif fairness_metric == 4:
                    positive_slice = np.where(y_i_array == 1)
                    global_prop = np.sum(y_i_preds_array[positive_slice])/y_i_preds_array[positive_slice].size

                prop_ub = global_prop + epsilon
                prop_lb = global_prop - epsilon

                model.add(s_nb_1_plus <= prop_ub * s_nb_1)
                model.add(prop_lb * s_nb_1 <= s_nb_1_plus)

                model.add(s_nb_0_plus <= prop_ub * s_nb_0)
                model.add(prop_lb * s_nb_0 <= s_nb_0_plus)
            elif constraint_expr=='between_groups':
                other_term_ub = int(np.ceil(tot_examples/2) * np.ceil(tot_examples/2))
                other_term = model.integer_var(0, other_term_ub, 'other_term')
                model.add(other_term == s_nb_0 * s_nb_1)

                # Difference
                #prods_vars_ub = (n_0_plus + n_1_plus) * (y_i_preds_array.size - 1) 
                prods_vars_ub = (n_0_plus + n_1_plus) * (tot_examples - 1) 
                prod1 = model.integer_var(0, prods_vars_ub, 'prod1')
                model.add(prod1 == s_nb_0_plus * s_nb_1)

                prod2 = model.integer_var(0, prods_vars_ub, 'prod2')
                model.add(prod2 == s_nb_1_plus * s_nb_0)

                diff_term = model.integer_var(-prods_vars_ub, prods_vars_ub, 'diff_term')
                model.add(diff_term == prod1 - prod2)
            if self.verbosity != 'Quiet':
                print("constraint set for both groups: positive prediction rate in [", global_prop, "-", epsilon, "," , global_prop, "+", epsilon, ']')
        else:
            fairness_constraint = False

        # Objective
        model.minimize(model.sum([model.element(d_1_y_pred_0, s_1_0_minus), model.element(d_0_y_pred_0, s_0_1_minus), model.element(d_1_y_pred_1, s_1_0_plus), model.element(d_0_y_pred_1, s_0_1_plus)]))
        
        if self.verbosity != 'Quiet':
            print("Model Created!")

        # Solve model
        msol = model.solve(TimeLimit=t_out, Workers=1, LogVerbosity=self.verbosity, RelativeOptimalityTolerance=0.0, OptimalityTolerance=0)#, RelativeOptimalityTolerance=0.0, OptimalityTolerance=0)

        if self.verbosity != 'Quiet':
            print("Model solving ended!")

        import docplex.cp.solution
        if msol.get_solve_status() ==  docplex.cp.solution.SOLVE_STATUS_OPTIMAL or  msol.get_solve_status() == docplex.cp.solution.SOLVE_STATUS_FEASIBLE:
            # check fairness constraint
            if (fairness_metric == 1 or fairness_metric == 3 or fairness_metric == 4):
                if fairness_constraint:
                    ratio_0 = msol.get_var_solution('s_nb_0_plus').get_value()/msol.get_var_solution('s_nb_0').get_value()
                    ratio_1 = msol.get_var_solution('s_nb_1_plus').get_value()/msol.get_var_solution('s_nb_1').get_value()
                    #print("Diff 1 is ", ratio_0 - global_prop)
                    #print("Diff 2 is ", ratio_1 - global_prop)
                    if np.fabs(ratio_0 - global_prop) > np.fabs(ratio_1 - global_prop):
                        train_unf_cpo = ratio_0 - global_prop
                    else:
                        train_unf_cpo = ratio_1 - global_prop
                    #if ((ratio_0 - global_prop) + (ratio_1 - global_prop)) < 10e-8: # same value with opposite sign
                    #    train_unf_cpo = -np.fabs(ratio_0 - global_prop)
                else:
                    train_unf_cpo = -1000

            self.fitted = True
            if self.verbosity != 'Quiet':
                print("n_0_plus=", n_0_plus)
                print("n_1_plus=", n_1_plus)
                print("n_0_minus=", n_0_minus)
                print("n_1_minus=", n_1_minus)
                var_list = ['s_0_1_plus', 's_1_0_plus', 's_0_1_minus', 's_1_0_minus', 's_nb_0_plus', 's_nb_1_plus', 's_nb_0_minus', 's_nb_1_minus', 's_nb_0', 's_nb_1']#, 'other_term']#, 'prod1', 'prod2', 'diff_term'] #'other_term_scaled',
                for var in var_list:
                    print(var, " = ", msol.get_var_solution(var).get_value())
            objective_val = msol.get_objective_values()[0]
            try:
                objective_val = objective_val[0]
            except:
                objective_val = objective_val
            self.d_i_hat_list = np.copy(d_i_array)
            if msol.get_var_solution('s_0_1_plus').get_value() > 0:
                for i in range(msol.get_var_solution('s_0_1_plus').get_value()):
                    index_to_flip = d_0_y_pred_1_indexes[i]
                    self.d_i_hat_list[index_to_flip] = 1
            if msol.get_var_solution('s_0_1_minus').get_value() > 0:
                for i in range(msol.get_var_solution('s_0_1_minus').get_value()):
                    index_to_flip = d_0_y_pred_0_indexes[i]
                    self.d_i_hat_list[index_to_flip] = 1
            if msol.get_var_solution('s_1_0_plus').get_value() > 0:
                for i in range(msol.get_var_solution('s_1_0_plus').get_value()):
                    index_to_flip = d_1_y_pred_1_indexes[i]
                    self.d_i_hat_list[index_to_flip] = 0
            if msol.get_var_solution('s_1_0_minus').get_value() > 0:
                for i in range(msol.get_var_solution('s_1_0_minus').get_value()):
                    index_to_flip = d_1_y_pred_0_indexes[i]
                    self.d_i_hat_list[index_to_flip] = 0
            if msol.get_solve_status() ==  docplex.cp.solution.SOLVE_STATUS_OPTIMAL:
                return "OPTIMAL", objective_val, train_unf_cpo
            else:
                return "FEASIBLE", objective_val, train_unf_cpo
        elif msol.get_solve_status() == docplex.cp.solution.SOLVE_STATUS_INFEASIBLE:
            self.d_i_hat_list = np.copy(d_i_array)
            self.fitted = True
            return_res = "INFEASIBLE", -1, -1
        else:
            return_res = "ERROR", msol.get_solve_status()
        
        return return_res

    def predict(self):
        if self.fitted:
            return np.asarray(self.d_i_hat_list)
        else:
            print("This ReconstructionCorrector is not fitted!")

# Attack success evaluation
def evaluate_reconstruction(X_predicted, ground_truth):
    assert(ground_truth.shape == X_predicted.shape)
    qual = np.sum(X_predicted == ground_truth) / ground_truth.size
    return qual

# To perform the normalization/exponentiation (of the adversary's confidence scores) process
def scale_probas_manual(d_i_attacker_probas, expo_factor, verbose=False):
    if verbose:
        print("Old min = ", np.min(d_i_attacker_probas), " Old max = ", np.max(d_i_attacker_probas)) 

    # Normalize
    min_proba = np.min(d_i_attacker_probas)
    max_proba = np.max(d_i_attacker_probas)
    bias_param_init=10e-10
    d_i_attacker_probas = [1.0+((proba - min_proba) / (max_proba - min_proba)) for proba in d_i_attacker_probas]

    if verbose:
        print("New min = ", np.min(d_i_attacker_probas), " new max = ", np.max(d_i_attacker_probas))    
    
    # Exponentiate    
    d_i_attacker_probas = np.asarray([p**expo_factor for p in d_i_attacker_probas])

    if verbose:
        print("Rate of unique probas: ", np.unique(np.asarray(d_i_attacker_probas)).size/d_i_attacker_probas.size)
        print("New min = ", np.min(d_i_attacker_probas), " new max = ", np.max(d_i_attacker_probas))    
    return d_i_attacker_probas
