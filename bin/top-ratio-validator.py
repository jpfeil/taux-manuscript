import pandas as pd
import optuna
import argparse
import numpy as np
import logging
import os

import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay

from sklearn.metrics import roc_auc_score

from pathlib import Path

import sys

sys.path.append(Path(os.getcwd()).parents[2].as_posix())

from lib.validation import get_elasticnet_model, get_svm_linear_model, get_svm_rbf_model

logger = logging.getLogger()

optuna.logging.set_verbosity(optuna.logging.ERROR)

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Set random seed so this part is reproducible
# https://www.random.org/ 2023-08-09
np.random.seed(3866)

def run_top_ratio_analysis(input_dir, output_dir, date, effect, nratios, frac, corr):
    #
    # Train
    #    
    train_minmax_path = os.path.join(input_dir, f"synthetic-train-minmax-eff-{effect}-ndiff-{nratios}-frac-{frac}-corr-{corr}-{date}.tsv")
    
    assert os.path.exists(train_minmax_path)
    
    train_minmax = pd.read_csv(train_minmax_path,
                     sep='\t', 
                     index_col=0)

    train_minmax = train_minmax.sample(frac=1.0, axis=1)
    
    ratio_path = os.path.join(input_dir, f"synthetic-train-minmax-ratios-eff-{effect}-ndiff-{nratios}-frac-{frac}-corr-{corr}-splits-1-{date}.tsv")
    
    assert os.path.exists(ratio_path)
    
    trainR = pd.read_csv(ratio_path,
                         sep='\t', 
                         header=None,
                         comment="#")
    
    trainR.columns = ["numerator", "denominator", "score"]
    
    trainR["ratio"] = trainR["numerator"] + ":::" + trainR["denominator"]
    
    top_ratios = trainR.sort_values("score", ascending=False).head(500)
    
    trainX = pd.DataFrame(index=top_ratios["ratio"], columns=train_minmax.columns, dtype=np.float64)
    
    for i, row in top_ratios.iterrows():
        trainX.loc[row.ratio] = train_minmax.loc[row["numerator"], :] / train_minmax.loc[row["denominator"], :]
        
    trainY = np.array([0 if x.startswith("nonresponders") else 1 for x in trainX.columns])
    
    #
    # Test
    #
    test_minmax_path = os.path.join(input_dir, f"synthetic-test-minmax-eff-{effect}-ndiff-{nratios}-frac-{frac}-corr-{corr}-{date}.tsv")
    
    assert os.path.exists(test_minmax_path)
    
    test_minmax = pd.read_csv(test_minmax_path,
                        sep='\t', 
                        index_col=0)

    test_minmax = test_minmax.sample(frac=1.0, axis=1)
    
    testX = pd.DataFrame(index=top_ratios["ratio"], columns=test_minmax.columns, dtype=np.float64)
    
    for i, row in top_ratios.iterrows():
        testX.loc[row.ratio] = test_minmax.loc[row["numerator"], :] / test_minmax.loc[row["denominator"], :]
        
    testY = np.array([0 if x.startswith("nonresponders") else 1 for x in testX.columns])
    
    # Elastic Net
    elastic_net_model, elastic_net_model_features, elastic_net_model_best_value, elastic_net_model_params = get_elasticnet_model(trainX.values.T, trainY, n_trials=100)
    
    # RBF SVM Model
    rbf_svm_model, rbf_svm_model_features, rbf_svm_model_best_value, rbf_svm_params = get_svm_rbf_model(trainX.values.T, trainY, n_trials=100)
    
    # Linear SVM Model 
    linear_svm_model, linear_svm_model_features, linear_svm_model_best_value, linear_svm_params = get_svm_linear_model(trainX.values.T, trainY, n_trials=100)
    
    #
    # ROC
    #
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(np.linspace(0, 1.0, 10), np.linspace(0.0, 1.0, 10), linestyle='--', color="gray")
    axes[1].plot(np.linspace(0, 1.0, 10), np.linspace(0.0, 1.0, 10), linestyle='--', color="gray")

    stats = []
    
    for name, model, _features in [("ElasticNet", elastic_net_model, elastic_net_model_features), 
                                ("Linear SVM", linear_svm_model, linear_svm_model_features), 
                                ("RBF SVM", rbf_svm_model, rbf_svm_model_features)]:
        
        X_train = trainX.values.T
        _X_train = X_train[:, _features]

        RocCurveDisplay.from_estimator(model, 
                                       _X_train, 
                                       trainY, 
                                       name=name,
                                       ax=axes[0])
        
        train_auc = roc_auc_score(trainY, model.predict_proba(_X_train)[:, 1])
        stats.append((name, "train", effect, nratios, frac, corr, train_auc, train_minmax_path))
        
        X_test = testX.values.T
        _X_test = X_test[:, _features]

        RocCurveDisplay.from_estimator(model, 
                                       _X_test, 
                                       testY, 
                                       name=name,
                                       ax=axes[1])
                                  
        test_auc = roc_auc_score(testY, model.predict_proba(_X_test)[:, 1])
        stats.append((name, "test", effect, nratios, frac, corr, test_auc, test_minmax_path))

        
    axes[0].set_title(f"ROC Curve for Synthetic Training Data\nES: {effect} DEGs: {nratios} Response%: {frac} Corr Thresh: {corr}")
    axes[1].set_title(f"ROC Curve for Synthetic Testing Data\nES: {effect} DEGs: {nratios} Response%: {frac} Corr Thresh: {corr}")

    plt.tight_layout()

    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, f"synthetic-top-ratios-prediction-roc-curve-plots-eff-{effect}-ndiff-{nratios}-frac-{frac}-corr-{corr}-{date}.svg"), 
                format='svg', 
                bbox_inches='tight')
    
    return pd.DataFrame(stats, 
                        columns=["model", "dataset", "effect", "ndeg", "subtype_frac", "corr_thresh", "AUC", "path"])
    
    
def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--effect", type=float, required=True)
    parser.add_argument("--nratios", type=int, required=True)
    parser.add_argument("--corr", type=float,required=True)
    parser.add_argument("--frac", type=float, required=True)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"top-ratio-synthetic-validation-eff-{args.effect}-ndiff-{args.nratios}-frac-{args.frac}-corr-{args.corr}-{args.date}.tsv")
    if not os.path.exists(output_path):
        print(f"Effect Size: {args.effect} N Ratios: {args.nratios} Response Fraction: {args.frac} Correlation thresh: {args.corr}")
        stats = run_top_ratio_analysis(args.input_dir, args.output_dir, args.date, args.effect, args.nratios, args.frac, args.corr)
        stats.to_csv(output_path, sep='\t')
    
    
if __name__ == "__main__":
    main()