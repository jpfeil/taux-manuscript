import optuna
import numpy as np
import logging
import copy
import os
import timeout_decorator

from sklearn.metrics import RocCurveDisplay

from sklearn import linear_model
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

logger = logging.getLogger()

optuna.logging.set_verbosity(optuna.logging.ERROR)

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Set random seed so this part is reproducible
# https://www.random.org/ 2023-08-09
np.random.seed(3866)

class StopWhenPerfectScoreReached:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value == 1.0:
            print("Stopping study because perfect score is reached")
            study.stop()
            
def get_elasticnet_model(X_train: np.ndarray, y_train: np.ndarray, n_trials=100):

    @timeout_decorator.timeout(120, timeout_exception=optuna.TrialPruned)
    def objective(trial):
        
        features = trial.suggest_categorical("features", [10, 25, 50, 100, 200, 500])
    
        splits = trial.suggest_categorical("splits", [2, 3, 4, 5])
        
        penalty = trial.suggest_categorical("penalty", ["elasticnet"])
        
        solver = trial.suggest_categorical("solver", ["saga"])

        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        
        logreg_c = trial.suggest_float("logreg_c", 1e-4, 1e2, log=True)
        
        weight = trial.suggest_categorical("weight", [None, "balanced"])

        max_iter = trial.suggest_int("max_iter", 1000, 10000, log=True)

        clf = linear_model.LogisticRegression(
                        C=logreg_c,
                        penalty=penalty,
                        solver=solver,
                        random_state=3866, 
                        max_iter=max_iter,
                        l1_ratio=l1_ratio,
                        class_weight=weight,
                        n_jobs=10)

        try:
            # Assumes matrix is sorted by feature importance
            _X_train = X_train[:, :features]
            np.take(_X_train, np.random.rand(_X_train.shape[1]).argsort(), axis=1, out=_X_train)
            
            scores = cross_val_score(clf, 
                                    _X_train, 
                                    y_train, 
                                    cv=RepeatedStratifiedKFold(n_splits=splits, n_repeats=5, random_state=3866),
                                    n_jobs=splits * 5,
                                    scoring="f1")
            
            return np.median(scores).item()
        
        except Warning:
            pass
            
        except Exception as e:
            raise optuna.TrialPruned(e)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, catch=(), callbacks=[StopWhenPerfectScoreReached()])

    clf = linear_model.LogisticRegression(
        C=study.best_params["logreg_c"],
        penalty=study.best_params["penalty"],
        solver=study.best_params["solver"],
        random_state=3866, 
        max_iter=study.best_params["max_iter"], 
        l1_ratio=study.best_params["l1_ratio"],
        class_weight=study.best_params["weight"],
        n_jobs=10)
    
    features = study.best_params["features"]
    
    _X_train = X_train[:, :features]
    final_features = np.random.rand(_X_train.shape[1]).argsort()
    np.take(_X_train, final_features, axis=1, out=_X_train)

    clf.fit(_X_train, y_train)

    return clf, final_features, study.best_value, study.best_params


def get_svm_rbf_model(X_train, y_train, n_trials=500):
    
    @timeout_decorator.timeout(120, timeout_exception=optuna.TrialPruned)
    def objective(trial) -> float:
        
        features = trial.suggest_categorical("features", [10, 25, 50, 100, 200, 500])
    
        kernel = trial.suggest_categorical("kernel",
                                           ["rbf"])
        
        weight = trial.suggest_categorical("weight",
                                           [None, "balanced"])
        
        gamma = trial.suggest_float("gamma", 1e-4, 1e4, log=True)
        
        logreg_c = trial.suggest_float("logreg_c", 1e-4, 1e4, log=True)
        
        splits = trial.suggest_categorical("splits", [2, 3, 4, 5])
        
        max_iter = trial.suggest_int("max_iter", 1000, 10000, log=True)

        clf = SVC(kernel=kernel, 
                  class_weight=weight, 
                  C=logreg_c,
                  gamma=gamma,
                  random_state=3866, 
                  max_iter=max_iter,
                  probability=True)

        try:
            # Assumes matrix is sorted by feature importance
            _X_train = X_train[:, :features]
            np.take(_X_train, np.random.rand(_X_train.shape[1]).argsort(), axis=1, out=_X_train)
            
            scores = cross_val_score(clf, 
                                     _X_train, 
                                     y_train, 
                                     cv=RepeatedStratifiedKFold(n_splits=splits, n_repeats=5, random_state=3866),
                                     n_jobs=splits * 5,
                                     scoring="f1")
            
            return np.median(scores).item()
        
        except Warning:
            pass
            
        except Exception as e:
            logger.exception(e)
        
        return np.nan

    study = optuna.create_study(direction="maximize", 
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))

    study.optimize(objective, n_trials=n_trials, catch=(), callbacks=[StopWhenPerfectScoreReached()])

    clf = SVC(kernel=study.best_params["kernel"], 
              class_weight=study.best_params["weight"], 
              C=study.best_params["logreg_c"], 
              gamma=study.best_params["gamma"], 
              random_state=3866,
              max_iter=study.best_params["max_iter"], 
              probability=True)
    
    features = study.best_params["features"]
    _X_train = X_train[:, :features]
    final_features = np.random.rand(_X_train.shape[1]).argsort()
    np.take(_X_train, final_features, axis=1, out=_X_train)

    clf.fit(_X_train, y_train)

    return clf, final_features, study.best_value, study.best_params
    
def get_svm_linear_model(X_train, y_train, n_trials=500):

    @timeout_decorator.timeout(120, timeout_exception=optuna.TrialPruned) 
    def objective(trial) -> float:
        
        features = trial.suggest_categorical("features", [10, 25, 50, 100, 200, 500])
    
        kernel = trial.suggest_categorical("kernel",
                                           ["linear"])
        
        weight = trial.suggest_categorical("weight",
                                           [None, "balanced"])
        
        gamma = trial.suggest_float("gamma", 1e-4, 1e4, log=True)
        
        logreg_c = trial.suggest_float("logreg_c", 1e-4, 1e4, log=True)
        
        splits = trial.suggest_categorical("splits", [2, 3, 4, 5])
        
        max_iter = trial.suggest_int("max_iter", 1000, 10000, log=True)

        clf = SVC(kernel=kernel, 
                  class_weight=weight, 
                  C=logreg_c,
                  gamma=gamma,
                  random_state=3866, 
                  max_iter=max_iter,
                  probability=True)

        try:
            _X_train = X_train[:, :features]
            np.take(_X_train, np.random.rand(_X_train.shape[1]).argsort(), axis=1, out=_X_train)
            
            scores = cross_val_score(clf, 
                                     _X_train, 
                                     y_train, 
                                     cv=RepeatedStratifiedKFold(n_splits=splits, n_repeats=5, random_state=3866),
                                     n_jobs=splits * 5,
                                     scoring="f1")
            
            return np.median(scores).item()
        
        except Warning:
            pass
            
        except Exception as e:
            logger.exception(e)
        
        return np.nan

    study = optuna.create_study(direction="maximize", 
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))

    study.optimize(objective, n_trials=n_trials, catch=(), callbacks=[StopWhenPerfectScoreReached()])

    clf = SVC(kernel=study.best_params["kernel"], 
              class_weight=study.best_params["weight"], 
              C=study.best_params["logreg_c"], 
              gamma=study.best_params["gamma"], 
              random_state=3866, 
              max_iter=study.best_params["max_iter"], 
              probability=True)
    
    features = study.best_params["features"]
    _X_train = X_train[:, :features]
    final_features = np.random.rand(_X_train.shape[1]).argsort()
    np.take(_X_train, final_features, axis=1, out=_X_train)

    clf.fit(_X_train, y_train)

    return clf, final_features, study.best_value, study.best_params
