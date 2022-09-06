# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:52:58 2022

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st 

loaded_model = pickle.load(open('trained_model.saved','rb'))

## creating a function for prediction 

def credit_score (input_data):


     # change the input data as numpy array 
    input_data_asnumpy_array = np.asarray(input_data)

    # reshaping the data 
    input_data_reshaped = input_data_asnumpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        print ('high value customer')
    else:
        print('low value customer') 



def main():
    
    #giving a title
    st.title('Credit Score Prediction App')
    
    
    #Getting the input data from users
    
    recency = st.text_input('recency')
    frequency = st.text_input('frequency')
    monetary = st.text_input('monetary')
    
    
    #code for Prediction
    creditscore = ''
    
    # creating a button for prediction
    if st.button('credit score result'):
        creditscore = credit_score([recency,frequency,monetary,])
        
        
        
    st.success(creditscore) 
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    

                
                               
               
