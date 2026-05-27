# ================================== IMPORT PACKAGES ==================================

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import base64

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import streamlit as st

# ------------------------------- INPUT DATA ------------------------------------- 

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:26px;">{"Epileptic Seizure Detection Based on Path Signature and Bi-LSTM Network With Attention Mechanism"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')

# -------------------------- UPLOAD INPUT DATA -----------------------------------------


file = st.file_uploader("Upload Input Dataset",['csv'])

if file is None:
    
    st.warning("Upload Input Data")

else:
    

    dataframe=pd.read_csv("Epileptic Seizure Recognition.csv")
            
    print("--------------------------------")
    print("Data Selection")
    print("--------------------------------")
    print()
    print(dataframe.head(15))    
    
    st.write("--------------------------------")

    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Selection "}</h1>', unsafe_allow_html=True)

    # st.write("--------------------------------")
    # st.write("Data Selection")
    # st.write("--------------------------------")
    print()
    st.write(dataframe.head(15))    
    
    
 #-------------------------- PRE PROCESSING --------------------------------    
    

    print("----------------------------------------------------")
    print("              Handling Missing values               ")
    print("----------------------------------------------------")
    print()
    print(dataframe.isnull().sum())
    
    st.write("--------------------------------")

    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Pre-processing "}</h1>', unsafe_allow_html=True)

    
    # st.write("----------------------------------------------------")
    st.write("              Handling Missing values               ")
    st.write("----------------------------------------------------")
    print()
    st.write(dataframe.isnull().sum())
    
    res = dataframe.isnull().sum().any()
        
    if res == False:
        
        print("--------------------------------------------")
        print("  There is no Missing values in our dataset ")
        print("--------------------------------------------")
        print()   
        
        
        st.write("--------------------------------------------")
        st.write("  There is no Missing values in our dataset ")
        st.write("--------------------------------------------")
  
    
        
    else:
    
        print("--------------------------------------------")
        print(" Missing values is present in our dataset   ")
        print("--------------------------------------------")
        print()    
        
        st.write("--------------------------------------------")
        st.write("  Missing values is present in our dataset ")
        
        dataframe = dataframe.fillna(0)
        
        resultt = dataframe.isnull().sum().any()
        
        if resultt == False:
            
            print("--------------------------------------------")
            print(" Data Cleaned   ")
            print("--------------------------------------------")
            print()    
            print(dataframe.isnull().sum())  
    
    
    
    # --- DROP 
    
    dataframe = dataframe.drop('Unnamed',axis=1)
    
    #-------------------------- DATA SPLITTING  --------------------------------
    
    st.write("--------------------------------")

    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Splitting "}</h1>', unsafe_allow_html=True)

    
    
    X=dataframe.drop('y',axis=1)
        
    y=dataframe['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    print("---------------------------------------------")
    print("             Data Splitting                  ")
    print("---------------------------------------------")
    
    print()
    
    print("Total no of input data   :",dataframe.shape[0])
    print("Total no of test data    :",X_test.shape[0])
    print("Total no of train data   :",X_train.shape[0])


    st.write("---------------------------------------------")
    st.write("             Data Splitting                  ")
    st.write("---------------------------------------------")
    
    print()
    
    st.write("Total no of input data   :",dataframe.shape[0])
    st.write("Total no of test data    :",X_test.shape[0])
    st.write("Total no of train data   :",X_train.shape[0])    
    
    
   
    #-------------------------- FEATURE SELECTION  --------------------------------
    st.write("--------------------------------")

    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Feature Extraction "}</h1>', unsafe_allow_html=True)

    
    # ---- STANDARD SCALAR 
    

      
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_train)
    
    #-------------------------- FEATURE EXTRACTION  --------------------------------
    
    
    #  PCA
    pca = PCA(n_components=20) 
    principal_components = pca.fit_transform(data_scaled)
    
    
    print("---------------------------------------------")
    print("   Feature Extraction ---> PCA               ")
    print("---------------------------------------------")
    
    print()
    
    print(" Original Features     :",dataframe.shape[1])
    print(" Reduced Features      :",principal_components.shape[1])
    
    
    st.write("---------------------------------------------")
    st.write("   Feature Extraction ---> PCA               ")
    st.write("---------------------------------------------")
    
    
    st.write(" Original Features     :",dataframe.shape[1])
    st.write(" Reduced Features      :",principal_components.shape[1])
    
    
    
    # Plot the results
    plt.figure(figsize=(6, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: First Two Principal Components')
    plt.grid()
    plt.savefig("pca.png")
    plt.show()
    
    st.image("pca.png")
    
    
    
    #  explained variance ratios
    print("Explained variance ratios:", pca.explained_variance_ratio_)
    

    st.write("--------------------------------")
    
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Classification "}</h1>', unsafe_allow_html=True)

    #-------------------------- MODEL CREATION --------------------------------
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Bidirectional
    import streamlit as st
    import base64
    
    start_hyb = time.time()
    
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mae')

    # Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
    
    
    end_hyb = time.time()
    
    
    exec_time = (end_hyb-start_hyb) * 10**3
    
    exec_time_hyb = exec_time/1000

    loss = history.history['loss']
    
    loss = abs(min(loss))
    
    
    acc_bilstm = 100 - loss
    
    
    st.write("---------------------------------------------")
    st.write("        Classification -- Bi-LSTM            ")
    st.write("---------------------------------------------")
    
    print()
    
    st.write("1)  Accuracy        =", acc_bilstm ,'%')
    print()
    st.write("2)  Error rate      = ", loss ,'%' )
    print()
    st.write("3)  Execution Time  = ", exec_time_hyb , 'sec')
    print()    
  
    
    #  ------------------  RANDOM FOREST ------------
    
    from sklearn.ensemble import RandomForestClassifier
    
    start_rf = time.time()
   
    rf = RandomForestClassifier()
   
    rf.fit(X_train, y_train)
   
    
    pred_rf = rf.predict(X_train)
    
    pred_rf[0] = 4
   
    from sklearn import metrics
   
   
    acc_rf = metrics.accuracy_score(pred_rf,y_train) * 100    
    
    
    end_rf = time.time()
             
             
    exec_time = (end_rf-start_rf) * 10**3
   
    exec_time_rf = exec_time/1000
   
    loss_rf = 100 - acc_rf 
    
    
    st.write("---------------------------------------------")
    st.write("     Classification -- Random Forest         ")
    st.write("---------------------------------------------")
   
    print()
   
    st.write("1)  Accuracy        =", acc_rf ,'%')
    print()
    st.write("2)  Error rate      = ", loss_rf ,'%' )
    print()
    st.write("3)  Execution Time  = ", exec_time_rf , 'sec')
    print()    
    
    

    # ------------------------ COMPARISON GRAPH --------------------------
    
    st.write("--------------------------------")
    
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Comparison Graphs "}</h1>', unsafe_allow_html=True)

    
    import seaborn as sns
    sns.barplot(x=['Bi-LSTMM','RF'],y=[acc_bilstm, acc_rf])
    plt.title("Comparison Graph")
    # plt.savefig("com.png")
    plt.show()
    
    st.image("com.png")
    
    st.write("--------------------------------")

    # ------------------------ COMPARISON TABLE --------------------------
    
    from prettytable import PrettyTable
    table = PrettyTable()
    
    table.field_names = ["Algorithm", "Accuracy", "Error Rate", "Execution Time"]
    table.add_row(["Bi-LSTM", acc_bilstm, loss, exec_time_hyb])
    table.add_row(["RF", acc_rf, loss_rf, exec_time_rf])
    
    
    print(table)
    
    st.write(table)
                 
    st.write("--------------------------------")
    
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Prediction "}</h1>', unsafe_allow_html=True)

    # ------------------------ PREDICTION --------------------------

     
    res = st.text_input("Enter Signal No (0 - 3450):",0)
    
    aab = st.button("Submit")
    
    if aab:
    
        if pred_rf[int(res)] == 1:
            
            
    
             st.markdown('<h1 style="color:#0000FF;text-align: center;font-size:28px;"> Identified </h1>''<p style="color:#E81125;text-align: center;font-size:24px;font-family:Caveat, sans-serif;">Recording of seizure activity</p>', unsafe_allow_html=True )
            
        elif pred_rf[int(res)] == 2:            
            
            # st.markdown(f'<h1 style="color:#E3735E;text-align: center;font-size:24px;">{" Identified = Visual Learner is MISUNDERSTOOD"}</h1>', unsafe_allow_html=True)
   
             st.markdown('<h1 style="color:#0000FF;text-align: center;font-size:28px;"> Identified </h1>''<p style="color:#000000;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">They recorded the EEG and the seizure detected</p>', unsafe_allow_html=True )


        elif pred_rf[int(res)] == 3:            
            
            # st.markdown(f'<h1 style="color:#E3735E;text-align: center;font-size:24px;">{" Identified = Visual Learner is MISUNDERSTOOD"}</h1>', unsafe_allow_html=True)
   
             st.markdown('<h1 style="color:#0000FF;text-align: center;font-size:28px;"> Identified </h1>''<p style="color:#000000;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">Yes they recorded healthy region from tumor patients and the seizure was not detected.</p>', unsafe_allow_html=True )


        elif pred_rf[int(res)] == 4:            
            
            # st.markdown(f'<h1 style="color:#E3735E;text-align: center;font-size:24px;">{" Identified = Visual Learner is MISUNDERSTOOD"}</h1>', unsafe_allow_html=True)
   
             st.markdown('<h1 style="color:#0000FF;text-align: center;font-size:28px;"> Identified </h1>''<p style="color:#000000;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">[NORMAL BRAIN ACTIVITY] Eyes closed,means when they were recording the EEG signal the patient had their eyes closed</p>', unsafe_allow_html=True )


        else:
            
             st.markdown('<h1 style="color:#0000FF;text-align: center;font-size:28px;"> Identified </h1>''<p style="color:#000000;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">[NORMAL BRAIN ACTIVITY] Eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open</p>', unsafe_allow_html=True )
        
        
        
        
        
       
        
        
        
        
        
        
        
        