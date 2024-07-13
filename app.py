import numpy as np
import pandas as pd
import base64
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


wine_df = pd.read_csv('winequality-red.csv')

X = wine_df.drop('quality', axis=1)
y = wine_df['quality'].apply(lambda yval: 1 if yval >= 7 else 0)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

X_test_prediction = model.predict(X_test)
accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Model accuracy: {accuracy:.2f}")

def set_background():
    background_css = """
    <style>
    .stApp {
        background: linear-gradient(to right, #722f37 , #54121a , #4f0303);
    }
    .header {
        display: flex;
        align-items: center;
        border-bottom: 2px solid #3a0101 ;
    }
    .header h1{
        margin-top : 10px;
        font-family: cursive;
    }
    .header img {
        height: 80px;
    }
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

def display_header():
    left_image_path = "./Pics/2.png"
    right_image_path = "./Pics/3.png"
    left_image_base64 = base64.b64encode(open(left_image_path, "rb").read()).decode()
    right_image_base64 = base64.b64encode(open(right_image_path, "rb").read()).decode()
    
    st.markdown(
        f"""
        <div class="header">
            <img src="data:image/png;base64,{left_image_base64}" class="img-fluid">
            <h1>Wine Quality Prediction</h1>
            <img src="data:image/png;base64,{right_image_base64}" class="img-fluid">
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    set_background()
    display_header()

    st.write("""
    ## Predict the Quality of Wine
    """)
    
    features = {}
    features['fixed acidity'] = st.number_input('Fixed Acidity', 4.0, 15.0, 7.4)
    features['volatile acidity'] = st.number_input('Volatile Acidity', 0.1, 1.5, 0.7)
    features['citric acid'] = st.number_input('Citric Acid', 0.0, 1.0, 0.0)
    features['residual sugar'] = st.number_input('Residual Sugar', 1.0, 15.0, 2.5)
    features['chlorides'] = st.number_input('Chlorides', 0.01, 0.1, 0.07)
    features['free sulfur dioxide'] = st.number_input('Free Sulfur Dioxide', 1, 50, 30)
    features['total sulfur dioxide'] = st.number_input('Total Sulfur Dioxide', 5, 200, 100)
    features['density'] = st.number_input('Density', 0.990, 1.005, 0.996)
    features['pH'] = st.number_input('pH', 2.5, 4.0, 3.3)
    features['sulphates'] = st.number_input('Sulphates', 0.3, 2.0, 0.7)
    features['alcohol'] = st.number_input('Alcohol', 8.0, 15.0, 10.0)


    if st.button('Predict'):
        feature_values = np.array([features[feature] for feature in X.columns])
        prediction = model.predict(feature_values.reshape(1, -1))
        if prediction[0] == 1:
            st.write(""" ## Good Quality Wine 🍷🥂 """)
        else:
            st.write(""" ## Bad Quality Wine ❌ """)

if __name__ == '__main__':
    main()