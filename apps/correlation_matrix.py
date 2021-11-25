import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso


#@st.cache(allow_output_mutation=True)
def app():
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 100)
    pd.options.display.float_format = '{:,.2f}'.format
    st.set_option('deprecation.showPyplotGlobalUse', False)

    df = pd.read_csv('apps/orly_processed2.csv')

    X = df.iloc[:, 2:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    pipeline = Pipeline([('scaler',StandardScaler()), ('model',Lasso())])
    search = GridSearchCV(pipeline, {'model__alpha':np.arange(0.1,10,0.1)}, cv=5, scoring="neg_mean_squared_error", verbose=3)

    lasso = []
    features = list(X.columns)

    # for i in range(10):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #     search.fit(X_train, y_train)
    #     coefficients = search.best_estimator_.named_steps['model'].coef_
    #     importance = np.abs(coefficients)
    #     lasso += list(np.array(features)[importance > 0])
    # lasso = list(set(lasso))

    lasso = ['WP1', 'make_eye_contact2_12_18', 'WP2', 'HP4', 'WP9_H', 'WP18_H', 'HP1', 'point_upon_request2_12_18', 
             'afraid_parents_till_18months', 'HP2', 'Mileston_failure_6w_18m', 'walk_with_help2_12_18']

    df_reduced = df[['Autism'] + lasso]
    corr = df_reduced.corr()
    st.dataframe(corr.style.background_gradient(cmap='coolwarm').set_precision(3))

    return X, y


    # available_modes = [1,2,3,4,5]
    # new_file = st.selectbox("Chose new session", available_modes)
    # st.write(df.head(new_file))


