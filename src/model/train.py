from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import pickle
import joblib
import pandas as pd

df = pd.read_csv("./processed_data.csv")

X = df.drop(columns=["median_house_value","Unnamed: 0"])
y = df["median_house_value"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Train a model
reg = LinearRegression().fit(X, y.ravel())
# Print out training r2
print(reg.score(X,y.ravel() ))

# Write the model to a file


filename = './models/model.pkl'

joblib.dump(reg, filename)



# Write the model to a file

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
X_test.to_csv("testdata/X_test.csv")
y_test.to_csv("testdata/y_test.csv")


