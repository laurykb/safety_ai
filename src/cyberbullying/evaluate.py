from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report
import pandas as pd

from cyberbullying.config import REPORTS_DIR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
import numpy as np
from sklearn.metrics import f1_score


def write_model_report (
    y_pred,
    y_test,
    file_name,
) :
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(REPORTS_DIR / f"{file_name}.csv", index=True)

def test_model (
    model,
    X_test,
    y_test,
    file_name,  
    write=False,
) :
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    if write :
        print(f"Writing model report to {REPORTS_DIR / f'{file_name}.csv'}")
        write_model_report(
            y_pred=y_pred,
            y_test=y_test,
            file_name=file_name,
        )
    return y_pred

def roc_curve(y_test, y_prob) :
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate"
)
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def pr_curve(y_test, y_prob) :
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print("Average Precision (AP):", ap)
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.show()

def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Cyberbullying", "Cyberbullying"])
    disp.plot(cmap='Blues')

def balance_accuracy(y_test, y_pred):
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy:", bal_acc)

def mcc(y_test, y_pred):
    mcc = matthews_corrcoef(y_test, y_pred)
    print("MCC:", mcc)

def find_threshold(y_test, y_prob):

    best_thresh, best_f1 = 0, 0
    for t in np.linspace(0, 1, 101):
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"Best F1={best_f1:.3f} at threshold={best_thresh:.2f}")
