import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

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

    user_input = st.sidebar.number_input("Tiempo a analizar: ", 0.0)

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)
        data.dropna(inplace=True)

        # get time
        # if st.sidebar.checkbox("Obtener tiempo de falla"):
        if st.sidebar.button("Obtener tiempo de falla"):

                # Do all computing to get the prediction time
                
                # PASTE CODE HERE

                y = data.iloc[:, 0]
                print(y.shape)

                # Specify the sample size
                size = data.shape[0]
                x = scipy.arange(size)
                print(x.shape)

                # get prediction time
                predictionTime = 15


                # Step 3: Obtain the distribution parameters

                # Define the distribution with the best fit
                dist = getattr(scipy.stats, 'gamma')

                # Fit the distribution to the data
                param = dist.fit(y)

                # Distribution parameters (e.g. 'a', 'b' for beta distribution)
                args = param[:-2]

                # Location parameter
                loc = param[-2]

                # Scale parameter
                scale = param[-1]

                # display results
                st.markdown("Probability of failure before time 300: " + str(round(scipy.stats.gamma.cdf(user_input, *args, loc=loc, scale=scale)*100,2)) + "%")

                st.markdown("Reliability Estimation at time 300: " + str(round(scipy.stats.gamma.sf(user_input, *args, loc=loc, scale=scale)*100,2)) + "%")

                st.markdown("Probability density function at time 300: " + str(round(scipy.stats.gamma.pdf(user_input, *args, loc=loc, scale=scale)*100,2)))

                # print("Probability of failure before time 300:", round(scipy.stats.gamma.cdf(300, *args, loc=loc, scale=scale)*100,2),"%")

                # print("Reliability Estimation at time 300:", round(scipy.stats.gamma.sf(300, *args, loc=loc, scale=scale)*100,2),"%")

                # print("Probability density function at time 300", round(scipy.stats.gamma.pdf(300, *args, loc=loc, scale=scale)*100,2))

if __name__ == "__main__":
    main()