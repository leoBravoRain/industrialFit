import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# from sklearn import tree
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# svr = SVR(kernel='linear')
# from sklearn.neighbors import KNeighborsClassifier

# ##Metrics
# from sklearn.metrics import mean_squared_error
# from math import sqrt

# # scatter matrix
# from pandas.plotting import scatter_matrix

# ##Functions and variables to use
# superv = {'Linear Regression': LinearRegression(),
#         # 'Suppor Vector Regression':svr,
#         # 'Decision Tree Regressor': tree.DecisionTreeRegressor(),
#         # 'Random Forest Regressor': RandomForestRegressor()
#         }

# unsuperv = {}

# sup_model_to_models = {'Supervised':superv,
#                     # 'Unsupervised':unsuperv
#                     }

##Center text
def main():

    st.markdown("""
            # Industrial Fit 
            """)

    st.markdown("""
            Ingresa los datos y obtén una predicción del tiempo de falla de un equipo
            """)

    ##Side bar
    st.sidebar.markdown("""
            # Sigue estos pasos
            """)

    # sup_model = st.sidebar.radio(
    #     '1. Select your Machine Learning category:',
    #     (
    #         'Supervised',
    #         # 'Unsupervised'
    #     )
    # )

    # if sup_model is not None:
    #     models_d = sup_model_to_models[sup_model]

    uploaded_file = st.sidebar.file_uploader('Sube el archivo: ', type='csv', encoding='auto', key=None)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)
        data.dropna(inplace=True)

        # get time
        # if st.sidebar.checkbox("Obtener tiempo de falla"):
        if st.sidebar.button("Obtener tiempo de falla"):

            # Do all computing to get the prediction time
            
            # PASTE CODE HERE

            # get prediction time
            predictionTime = 15

            # dislplay prediction
            st.markdown("La predicción del tiempo de falla es: {} [seg]".format(predictionTime))


if __name__ == "__main__":
    main()