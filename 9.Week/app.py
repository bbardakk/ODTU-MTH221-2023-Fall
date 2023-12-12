import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

st.title('Prediction of IRIS Flower')


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# User input Slider
st.sidebar.header('Adjust Features')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    return pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)

input_df = user_input_features()

# Train Model
X = df[iris.feature_names]
y = df['target']
model = LogisticRegression()
model.fit(X, y)

# Inference
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Show user inputs and the outputs
st.subheader('Inputs')
st.write(input_df)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
st.write('Probability')
st.write(prediction_proba)
