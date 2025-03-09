import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


def validate_position(pos, x_min, x_max, integer_indices=None):
    pos = np.clip(pos, x_min, x_max)
    if integer_indices is not None:
        pos[integer_indices] = np.round(pos[integer_indices]).astype(int)
    return pos

def firework_explosion(X, fitness, ymax, H, epsilon, mu2, x_min, x_max):
    N = len(X)
    explosion_mask = np.random.rand(N) < mu2
    Si = (H * (ymax - fitness[explosion_mask] + epsilon) / np.sum(ymax - fitness + epsilon)).astype(int)
    new_individuals = []
    for index, s in zip(np.where(explosion_mask)[0], Si):
        for _ in range(s):
            Xnew = X[index] + np.random.normal(0, 1, X.shape[1])
            Xnew = validate_position(Xnew, x_min, x_max, integer_indices=[0, 1])
            new_individuals.append(Xnew)
    return new_individuals

def gwo_coevo_optimize(objective_function, NP, dim, NG, amax, x_max, x_min, coevolution_interval=10, firework_interval=10, mu2=0.2, H=10, epsilon=0.01):
    X = np.random.uniform(low=x_min, high=x_max, size=(NP, dim))
    fitness = np.array([objective_function(individual) for individual in X])
    best_index = np.argmin(fitness)
    X_best = X[best_index]
    fitness_best = fitness[best_index]
    ymax = np.max(fitness)

    indices_sorted = np.argsort(fitness)
    Xalpha, Xbeta, Xdelta = X[indices_sorted[0]], X[indices_sorted[1]], X[indices_sorted[2]]

    for t in range(1, NG + 1):
        a = amax * (1 - t / NG)
        if t % coevolution_interval == 0:
            indices_sorted = np.argsort(fitness)
            subgroup_size = NP // 3
            PE, PA, PB = X[indices_sorted[:subgroup_size]], X[indices_sorted[subgroup_size:2*subgroup_size]], X[indices_sorted[2*subgroup_size:]]
            XE, XA, XB = PE[0], PA[0], PB[0]

            for group, leader in [(PE, XE), (PA, XA), (PB, XB)]:
                for i in range(len(group)):
                    r = np.random.rand()
                    group[i] = 0.5 * ((1 + r) * group[i] + (1 - r) * leader)
                    group[i] = validate_position(group[i], x_min, x_max)

        if t % firework_interval == 0:
            new_individuals = firework_explosion(X, fitness, ymax, H, epsilon, mu2, x_min, x_max)
            for ind in new_individuals:
                X = np.append(X, [ind], axis=0)

        for i in range(NP):
            r1, r2 = np.random.rand(2)
            A = 2 * a * r1 - a
            C = 2 * r2
            D_alpha, D_beta, D_delta = np.abs(C * Xalpha - X[i]), np.abs(C * Xbeta - X[i]), np.abs(C * Xdelta - X[i])
            X1, X2, X3 = Xalpha - A * D_alpha, Xbeta - A * D_beta, Xdelta - A * D_delta
            X[i] = (X1 + X2 + X3) / 3
            X[i] = validate_position(X[i], x_min, x_max)

        fitness = np.array([objective_function(individual) for individual in X])
        current_best_index = np.argmin(fitness)
        current_best_fitness = fitness[current_best_index]
        if current_best_fitness < fitness_best:
            X_best = X[current_best_index].copy()
            fitness_best = current_best_fitness

    return X_best, fitness_best

def GWO(function, NP, dim, NG, amax, x_max, x_min):
    X0 = np.random.uniform(low=x_min, high=x_max, size=(NP, dim))
    value0 = [function(x) for x in X0]
    index_sort = np.argsort(value0)
    X0 = X0[index_sort]
    value0 = np.array(value0)[index_sort]
    X_best = [X0[0]]
    value_best = [value0[0]]
    Xalpha, Xbeta, Xdelta = X0[0], X0[1], X0[2]

    for i in range(NG):
        a = amax * (1 - i / NG)
        for j in range(NP):
            A1, A2, A3 = 2 * a * np.random.rand() - a, 2 * a * np.random.rand() - a, 2 * a * np.random.rand() - a
            C1, C2, C3 = 2 * np.random.rand(), 2 * np.random.rand(), 2 * np.random.rand()
            Dalpha, Dbeta, Ddelta = np.abs(C1 * Xalpha - X0[j]), np.abs(C2 * Xbeta - X0[j]), np.abs(C3 * Xdelta - X0[j])
            X1, X2, X3 = Xalpha - A1 * Dalpha, Xbeta - A2 * Dbeta, Xdelta - A3 * Ddelta
            X0[j] = (X1 + X2 + X3) / 3
            X0[j][0] = np.clip(X0[j][0], 1, x_max[0])
            X0[j][1] = np.clip(X0[j][1], 1, x_max[1])
            value0[j] = function(X0[j])

        min_index = np.argmin(value0)
        if value0[min_index] < value_best[-1]:
            value_best.append(value0[min_index])
            X_best.append(X0[min_index])
        else:
            value_best.append(value_best[-1])
            X_best.append(X_best[-1])

        index_sort = np.argsort(value0)
        X0 = X0[index_sort]
        Xalpha, Xbeta, Xdelta = X0[0], X0[1], X0[2]

    return X_best[-1], value_best[-1]

# Load data
data = load_iris()
X, y = data.data, data.target

# Define objective function
def rf_objective(params):
    n_estimators, max_depth = int(params[0]), int(params[1])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    return -cross_val_score(rf, X, y, cv=3, n_jobs=-1).mean()

# Initialize parameters
param_bounds = np.array([[10, 300], [1, 50]])
n_runs = 10

# W&B initialization
wandb.init(project='rf-gwo-optimization', entity='bioevai')

# Default Random Forest
default_scores = []
for _ in range(n_runs):
    default_rf = RandomForestClassifier()
    default_rf.fit(X, y)
    default_scores.append(cross_val_score(default_rf, X, y, cv=5).mean())
default_score = np.mean(default_scores)
wandb.log({"Default Random Forest Score": default_score})

# Original GWO optimized Random Forest
original_gwo_scores = []
for _ in range(n_runs):
    best_params_original, best_score_original = GWO(rf_objective, 20, 2, 30, 2, param_bounds[:, 1], param_bounds[:, 0])
    optimized_rf_original = RandomForestClassifier(n_estimators=int(best_params_original[0]), max_depth=int(best_params_original[1]))
    optimized_rf_original.fit(X, y)
    original_gwo_scores.append(cross_val_score(optimized_rf_original, X, y, cv=5).mean())
optimized_score_original = np.mean(original_gwo_scores)
wandb.log({"Original GWO Optimized Score": optimized_score_original})

# Modified GWO optimized Random Forest
modified_gwo_scores = []
for _ in range(n_runs):
    best_params_modified, best_score_modified = gwo_coevo_optimize(rf_objective, 20, 2, 30, 2, param_bounds[:, 1], param_bounds[:, 0])
    optimized_rf_modified = RandomForestClassifier(n_estimators=int(best_params_modified[0]), max_depth=int(best_params_modified[1]))
    optimized_rf_modified.fit(X, y)
    modified_gwo_scores.append(cross_val_score(optimized_rf_modified, X, y, cv=5).mean())
optimized_score_modified = np.mean(modified_gwo_scores)
wandb.log({"Modified GWO Optimized Score": optimized_score_modified})

# Output comparison results
print("Default Random Forest Score:", default_score)
print("Original GWO Optimized Score:", optimized_score_original)
print("Modified GWO Optimized Score:", optimized_score_modified)

# Finish the W&B run
wandb.finish()
