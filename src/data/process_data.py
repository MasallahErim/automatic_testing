import pandas as pd 
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("src/data/a.csv")


df.describe().T
df.info()
df["ocean_proximity"].value_counts()


def deletenullvalue(df):
    for key, value in df.isnull().sum().to_dict().items():
        if (value*100/len(df) < 5.0) & value >0:
            df.dropna(subset=[key],inplace=True)
    else:
        if (df[key].dtype !="object") & value !=0:
            df[key].fillna(df[key].mean(), inplace=True)
    return df 
def outlierdetect(df):
    for col in df.columns:
        if df[col].dtype !="object":
            std = df[col].std()
            mean = df[col].mean()
            upper = mean + 3 * std
            lower = mean - 3 * std
            index_of_autlier = df[(df[col] > upper) | (df[col] < lower)].index
            df.drop(index_of_autlier, axis=0, inplace=True)
    return df
def getdummies_will(df, colname):
    df = pd.get_dummies(df, columns=[colname], dtype=int)
    return df
def scale(df):
    scale = StandardScaler()
    dfs = pd.DataFrame(scale.fit_transform(df), columns=df.columns)
    return dfs     


df = deletenullvalue(df)       
df = outlierdetect(df)
df = getdummies_will(df, "ocean_proximity")
dfs = scale(df)


dfs.to_csv("src/data/processed_data/processed_data.csv")


