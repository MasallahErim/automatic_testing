import pickle5
import pandas as pd

model = pickle5.load(open("models/model.pkl", "rb"))

X_test  = pd.read_csv("./testdata/X_test.csv")

del X_test["Unnamed: 0"]

# Test on the model
y_hat = model.predict(X_test)