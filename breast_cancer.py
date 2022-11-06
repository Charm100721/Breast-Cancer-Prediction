import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("breast_cancer.csv")
data.drop(columns="Unnamed: 32", inplace=True)
data.drop(columns="id", inplace=True)
data["Result"] = data["diagnosis"].apply(lambda x: 1 if x=="M" else 0)
data.drop(columns="diagnosis", inplace=True)


X = data[["radius_mean", "perimeter_mean", "concave points_mean", "radius_worst", "perimeter_worst", "area_worst", "concave points_worst"]]
Y = data.iloc[0:570, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)


st.title("Breast Cancer Prediction")

def features():
    radius_mean = st.slider("RADIUS MEAN", 0.0,30.0, 17.99)
    perimeter_mean = st.slider("PERIMETER MEAN", 40.0,190.0,122.80)
    concave_points_mean = st.slider("CONCAVE POINTS MEAN", 0.0,1.0,0.147)
    radius_worst = st.slider("RADIUS WORST", 7.0,37.0,25.38)
    perimeter_worst = st.slider("PERIMETER WORST", 50.0,250.0, 184.6)
    area_worst = st.slider("AREA WORST", 185.0,4254.0,2019.0)
    concave_points_worst = st.slider("CONCAVE POINTS WORST", 0.0,1.0,0.2654)
    New_dataset = {
                        "RADIUS MEAN" : radius_mean,
                        "PERIMETER MEAN" : perimeter_mean,
                        "CONCAVE POINTS MEAN" : concave_points_mean,
                        "RADIUS WORST" : radius_worst,
                        "PERIMETER WORST" : perimeter_worst,
                        "AREA WORST" : area_worst,
                        "CONCAVE POINTS WORST" : concave_points_worst
        }
    features = pd.DataFrame(New_dataset, index=[0])
    return features

df = features()
st.write(df)


def predict():
    features = np.array(["RADIUS MEAN", "PERIMETER MEAN", "CONCAVE POINTS MEAN", "RADIUS WORST", "PERIMETER WORST", "AREA WORST", "CONCAVE POINTS WORST"])
    prediction = model.predict(df)
    if prediction[0] == 0:
        st.success("The breast cancer is BENIGN")
    else:
        st.error("The breast cancer is MALIGNANT")  

predict_button = st.button("Predict")

if predict_button == True:
    predict()

 
