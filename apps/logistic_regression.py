import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



def app():
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 100)
    pd.options.display.float_format = '{:,.2f}'.format
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('Logistic Regression model')
    df = pd.read_csv('apps/orly_processed2.csv')
    X = df.iloc[:, 2:]
    y = df.iloc[:, 0]

    choose_max_iter = st.checkbox("Choose max iterations for the regression:")
    iteration_options = list(np.arange(100,1100,100))
    if choose_max_iter:
        iterations = st.selectbox("Number of iteration options:", iteration_options)
        choose_folds = st.checkbox("Choose number of folds for cross-validation:")
        folds_options = list(np.arange(2,11)) 
        if choose_folds:
            folds = st.selectbox("Number of cv folds:", folds_options)

            mean_accuracy = cross_val_score(LogisticRegression(max_iter=iterations), X, y, cv=folds, scoring='accuracy').mean()
            st.write("The mean accuracy is:", mean_accuracy)
    
