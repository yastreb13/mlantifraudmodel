import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve)

from features import add_features
from thresholds import porog_by_recall, porog_by_fpr


DATA_PATH = "creditcard.csv"
TEST_SIZE=0.2
RANDOM_STATE=13

TARGET_RECALL=0.80
MAX_FPR=None

OUT_BEST="best_model.joblib"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def time_based_split(df: pd.DataFrame, test_size: float):
    """
    Начальные даты на обучение, поздние на тест
    """
    df=df.sort_values("Time").reset_index(drop=True)
    split_idx=int(len(df)*(1-test_size))

    train_df=df.iloc[:split_idx]
    test_df=df.iloc[split_idx:]

    X_train=train_df.drop(columns=["Class"])
    y_train=train_df["Class"].astype(int)
    X_test=test_df.drop(columns=["Class"])
    y_test=test_df["Class"].astype(int)

    return X_train, X_test, y_train, y_test


def evaluate(y_true, proba, threshold, name):
    """
    Функция для оценки бинарной классификационной модели
    """

    pred=(proba>=threshold)

    print("\n")
    print("MODEL:", name)
    print("Threshold:", round(threshold, 6))
    print("ROC-AUC:", roc_auc_score(y_true, proba))
    print("PR-AUC :", average_precision_score(y_true, proba))
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))
    print(classification_report(y_true, pred, digits=4))

def plot_recall_vs_fpr(y_true, proba):
    """
    Recall  против FPR
    """
    
    thresholds=np.linspace(0, 1, 1001)
    recalls, fprs = [], []

    for t in thresholds:
        pred=(proba>=t)
        tn, fp, fn, tp =confusion_matrix(y_true, pred).ravel()
        fpr=fp/(fp+tn+1e-12)
        rec=tp/(tp+fn+1e-12)
        fprs.append(fpr)
        recalls.append(rec)

    plt.figure()
    plt.plot(fprs, recalls)
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall")
    plt.title("Recall vs FPR (Test)")
    plt.grid(True)
    plt.show()


def plot_score_hist(y_true, proba):
    """
    Гистограмма распределения вероятностей
    """
    plt.figure()
    plt.hist(proba[y_true==0], bins=50, alpha=0.7, label="Legit")
    plt.hist(proba[y_true==1], bins=50, alpha=0.7, label="Fraud")
    plt.xlabel("Предсказанная fraud proba")
    plt.ylabel("Count")
    plt.title("Score distributions (Test)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pr_curve_with_point(y_true, proba, thr):
    """
    Построение PRC с отмеченной точкой выбранного порога
    """
    precision, recall, thr_pr= precision_recall_curve(y_true, proba)

    if len(thr_pr)>0:
        idx=int(np.argmin(np.abs(thr_pr-thr)))
        p_at=precision[idx]
        r_at=recall[idx]
    else:
        p_at, r_at =precision[-1], recall[-1]

    plt.figure()
    plt.plot(recall, precision)
    plt.scatter([r_at], [p_at])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve (Test)")
    plt.grid(True)
    plt.show()

def calc_loss(y_true, proba, thr, cost_fp, cost_fn):
    """
    Посчитаем потери банка на fraud
    """
    pred=(proba>=thr)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    loss = cost_fp * fp + cost_fn * fn
    return loss, fp, fn, tp, tn


def main():
    df=load_data(DATA_PATH)

    print("Data shape:", df.shape)

    X_train, X_test, y_train, y_test = time_based_split(df, TEST_SIZE)

    feat=FunctionTransformer(add_features, validate=False)

    pipe=Pipeline(steps=[
        ("feat", feat),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    tscv=TimeSeriesSplit(n_splits=5)

    param_grid = [
        {
            "clf": [LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=-1
            )],
            "clf__C": [0.1, 1.0, 5.0],
        },
        {
            "clf": [RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )],
            "clf__n_estimators": [200, 400],
        },
        # {
        #     "clf": [RandomForestClassifier(
        #         random_state=RANDOM_STATE,
        #         n_jobs=-1,
        #         class_weight="balanced_subsample"
        #     )],
        #     "clf__max_depth": [None, 10, 20],
        # }
        {
            "clf": [CatBoostClassifier(
                random_seed=RANDOM_STATE,
                verbose=False,
                loss_function="Logloss"
            )],
            "clf__depth": [4, 6, 8],
            "clf__learning_rate": [0.03, 0.1],
            "clf__iterations": [300, 600],
        }
    ]

    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="average_precision",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    gs.fit(X_train, y_train)

    best_model=gs.best_estimator_
    print("Лучшая CV PR-AUC:", gs.best_score_)

    proba_test=best_model.predict_proba(X_test)[:,1]

    thr=(
        porog_by_recall(y_test, proba_test, TARGET_RECALL)
        if MAX_FPR is None
        else porog_by_fpr(y_test, proba_test, MAX_FPR)
    )

    evaluate(y_test, proba_test, thr, "Модель, найденная GridSearch")
    plot_recall_vs_fpr(y_test, proba_test)
    plot_score_hist(y_test, proba_test)
    plot_pr_curve_with_point(y_test, proba_test, thr)

    joblib.dump({"model": best_model, "threshold": thr}, OUT_BEST)
    print("Сохранено в:", OUT_BEST)

    cost_fp = 0.09
    cost_fn = 9.25

    pred=(proba_test>=thr)
    tn, fp, fn, tp =confusion_matrix(y_test, pred).ravel()
    loss=cost_fp*fp+cost_fn*fn

    print("\nПотери: ", loss)
    print("FP:", fp, "FN:", fn, "TP:", tp, "TN:", tn)

if __name__ == "__main__":
    main()

