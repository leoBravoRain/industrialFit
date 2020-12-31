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

#     user_input = st.sidebar.number_input("Tiempo a analizar: ", 0.0)
    user_input = 400

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)
        data.dropna(inplace=True)

        # get time
        # if st.sidebar.checkbox("Obtener tiempo de falla"):
        if st.sidebar.button("Analizar tiempos de falla"):

                # get data
                y = data.iloc[:, 0]
                # Specify the sample size
                size = data.shape[0]
                x = scipy.arange(size)


                # Obtain the distribution parameters
                distName = ['alpha', 'beta', 'expon', 'gamma', 'norm', 'rayleigh']
                distName  = distName[2]

                # Define the distribution with the best fit
                dist = getattr(scipy.stats, distName)

                # Fit the distribution to the data
                param = dist.fit(y)

                # Distribution parameters (e.g. 'a', 'b' for beta distribution)
                args = param[:-2]

                # Location parameter
                loc = param[-2]

                # Scale parameter
                scale = param[-1]

                # get results

                # 3) Mean failure time
                # This is the mean of the distirubtion
                meanFailureTime = dist.mean(loc = loc, scale = scale)
                # display results
                st.markdown("""
                        ### Tiempos de falla
                """)
                st.markdown("Tiempo medio de falla: " + str(round(meanFailureTime, 2)))

                # get data to plot
                a = np.arange(0, 2000)

                # 1) getting confiability of times
                # probabily that the machine is operating without failure until that time
                reliability = scipy.stats.expon.sf(a, *args, loc=loc, scale=scale)

                # 2) failure rate 
                # how many failures will be at an specific time
                failureRate = scipy.stats.expon.pdf(a, *args, loc=loc, scale=scale) / scipy.stats.expon.sf(a, *args, loc=loc, scale=scale)


                fig, ax = plt.subplots(2, 1, tight_layout = True)

                # reliability
                ax[0].plot(a, reliability, label = "reliability")
                ax[0].set_title("Confiabilidad")
                ax[0].set_xlabel("Tiempo")
                ax[0].set_ylabel("Confiabilidad %")

                # failure rate
                ax[1].plot(a, failureRate, label = "failure rate")
                ax[1].set_title("Tasa de falla")
                ax[1].set_xlabel("Tiempo")
                ax[1].set_ylabel("Tasa de falla")

                st.pyplot(fig)


if __name__ == "__main__":
    main()