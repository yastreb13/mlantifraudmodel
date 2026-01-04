import numpy as np
import pandas as pd

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    X: DataFrame без колонки Class
    Добвляем:
      hour из time
      log_amount из amount
      убираем amount
    """
    X=X.copy()
    X["log_amount"]=np.log1p(X["Amount"])
    X["hour"]=((X["Time"]/3600)%24).astype(int)
    X=X.drop(columns=["Amount"])
    return X
