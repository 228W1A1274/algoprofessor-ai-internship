import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

def train_ensemble():
    df = pd.read_csv('titanic_cleaned.csv')
    X = df.drop(columns=['survived'])
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. XGBoost
    print("Training XGBoost...")
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, pred_xgb)
    print(f"XGBoost Accuracy: {acc_xgb*100:.2f}%")

    # 2. LightGBM
    print("Training LightGBM...")
    model_lgb = lgb.LGBMClassifier()
    model_lgb.fit(X_train, y_train)
    pred_lgb = model_lgb.predict(X_test)
    acc_lgb = accuracy_score(y_test, pred_lgb)
    print(f"LightGBM Accuracy: {acc_lgb*100:.2f}%")

    return acc_xgb, acc_lgb

if __name__ == "__main__":
    train_ensemble()
