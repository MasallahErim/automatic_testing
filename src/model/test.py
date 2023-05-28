# import pickle
import pandas as pd
import joblib
model = joblib.load(open("src/model/models/model.pkl", "rb"))
X_test  = pd.read_csv("src/model/testdata/X_test.csv")
del X_test["Unnamed: 0"]
y_hat = model.predict(X_test)


