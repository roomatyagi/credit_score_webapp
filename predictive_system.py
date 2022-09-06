# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:50:23 2022

@author: DELL
"""
import numpy as np
import pickle
# loading the saved model 
loaded_model = pickle.load(open(r'E:\Project_72\deployment_model\trained_model.saved','rb'))

input_data = (0,1,2)

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
                           
                    