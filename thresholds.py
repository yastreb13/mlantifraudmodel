import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

def porog_by_recall(y_true, proba, target_recall: float) -> float:
    """
    Максимизируем порог, при котором recall не меньше target_recall
    """
    precision, recall, thresholds =precision_recall_curve(y_true, proba)
    idx=np.where(recall[:-1]>=target_recall)[0]
    return float(thresholds[idx[-1]]) if len(idx) else 0.5

def porog_by_fpr(y_true, proba, max_fpr: float) -> float:
    """
    Максимизируем recall, при подборе порога, при условии, что FPR не больше max_fpr
    """
    thresholds=np.linspace(0, 1, 2000+1)
    best_thr=0.5
    best_recall=-1.0

    for t in thresholds:
        pred=(proba>=t).astype(int)
        tn, fp, fn, tp =confusion_matrix(y_true, pred).ravel()
        fpr=fp/(fp+tn+1e-12)
        recall=tp/(tp+fn+1e-12)

        if fpr<=max_fpr and recall>best_recall:
            best_recall=recall
            best_thr=t

    return float(best_thr)
