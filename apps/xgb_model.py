import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV


def app():
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 100)
    pd.options.display.float_format = '{:,.2f}'.format
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('XGB model')
    df = pd.read_csv('apps/orly_processed2.csv')
    X = df.iloc[:, 2:]
    y = df.iloc[:, 0]

    model_operation = st.checkbox("Click for choosing the model mode you wish to apply")
    modes = ["out-of-the-box", "tuned params"]

    if model_operation:
        mode = st.selectbox("The available modes are:", modes)
        if mode == modes[0]:
            score = cross_val_score(XGBClassifier(use_label_encoder=False), X, y, cv=10, scoring='accuracy').mean()
            st.write("The average score of XGB " + modes[0] + " is: ", score)
        elif mode == modes[1]:
            clf = XGBClassifier(colsample_bytree=0.3, gamma=4, learning_rate=0.02, max_depth=10, n_estimators=400, subsample=0.85)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf.fit(X_train, y_train)
            test_score = clf.score(X_test,y_test)
            train_score = clf.score(X_train, y_train)
            st.write("The test score is: ", test_score)
            st.write("The train score is: ", train_score)
