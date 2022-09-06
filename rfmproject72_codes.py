

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns 
import datetime as dt

data = pd.read_excel(r"E:\Project_72\CreditAnalysis_data.xlsx")

# Drop unwanted column 
data = data.drop(['Unnamed: 0'], axis=1)
# checking duplicate records 
duplicate = data.duplicated()
sum(duplicate) 
#cheking null value in dataset 
data.isna().sum()
data = data.dropna()#removed null value records
data.isna().sum()

import seaborn as sns
#Visualizing the Missing values
sns.heatmap(data.isnull(),cbar=False,cmap='viridis')
data.isna().sum()
data.columns
# Exploratory data analysis #
data.info()#overview of the data
desc = data.describe()
#Automated EDA sweetviz library is used
# pip install sweetviz

import sweetviz 
my_report = sweetviz.analyze[data]


### RFM analysis and get rfm_score ###
#calculating frequency
frequency = data.groupby('master_order_id')['created'].count()
frequency = frequency.reset_index()
frequency.head()

#calculating recency
data['created'] = pd.to_datetime(data['created'])
data['diff'] = max(data['created'])- data['created']
recency = data.groupby('master_order_id')['diff'].min()
#recency['diff'] = recency['diff'].dt.days
recency = recency.reset_index()
rec = recency.head(50)
recency['diff'] = recency['diff'].dt.days
recency

#calculating monetary 
monetary = data.groupby('master_order_id')['bill_amount'].sum()
monetary = monetary.reset_index()
monetary.tail()
 

#merging frequency , recency , monetary in a one dataframe
rfm_table = pd.merge(recency, frequency, on='master_order_id',how = 'inner')
rfm_table = pd.merge(rfm_table,monetary, on = 'master_order_id',how = 'inner')
rfm_table.columns = ['master_order_id','recency','frequency','monetary']
rfm_table

#findout rfmscore by using quantiles

quantiles = rfm_table.quantile(q = [0.25,0.50,0.75])
quantiles = quantiles.to_dict()
def r_score (x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4
    

def fm_score (x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1
    
rfm_table ['r_quartile'] = rfm_table['recency'].apply(r_score, args =('recency',quantiles))  
rfm_table ['f_quartile'] = rfm_table['frequency'].apply(fm_score, args=('frequency',quantiles))   
rfm_table ['m_quartile'] = rfm_table['monetary'].apply(fm_score, args=('monetary',quantiles)) 

rfm_table['rfm_segment'] = rfm_table.r_quartile.map(str)+rfm_table.f_quartile.map(str)+rfm_table.m_quartile.map(str)
rfm_table['rfm_score'] = rfm_table[['r_quartile','f_quartile','m_quartile']].sum(axis=1)
rfm_table['rfm_segment'] 

                  

#Now make data ready for apply kmeans clustering

# checking skewness 
sns.distplot(rfm_table['recency'])
sns.distplot(rfm_table['frequency'])
sns.distplot(rfm_table['monetary'])

#Applying log transformation to remove skewness in data 
#Removing Skewness
km_rfm_table = rfm_table[['recency','frequency','monetary']]
km_rfm_table = np.log(km_rfm_table+1)
sns.distplot(km_rfm_table['frequency'])
sns.distplot(km_rfm_table['monetary'])
sns.distplot(km_rfm_table['recency'])

### Now apply Standardization 
from sklearn.preprocessing import StandardScaler
# Initialise the Scaler
scaler = StandardScaler()
# To scale data
km_rfm_table = scaler.fit_transform(km_rfm_table)
# Convert the array back to a dataframe
km_rfm_table = pd.DataFrame(km_rfm_table)
km_rfm_table = km_rfm_table.rename(columns = {0:'recency',1:'frequency',2:'monetary'})
d = km_rfm_table.describe()



#fitting k-Means clustering to rfm _norm 
from sklearn.cluster import	KMeans
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(km_rfm_table)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(km_rfm_table)
model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
# creating a  new column and assigning it to new column 
rfm_table['cluster'] = mb
rfm_table.head()



# Building a model by using decision tree classifier

data_class = pd.concat([km_rfm_table, rfm_table['cluster']], axis =1)

data_class['cluster'].unique()
data_class['cluster'].value_counts()
colnames = list(data_class.columns)
predictors = colnames[:3]
target = colnames[3]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data_class, test_size = 0.3)


from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy


######### Deployment codes ########

## making a predictive model ###3
input_data = (0,1,2)

# change the input data as numpy array 
input_data_asnumpy_array = np.asarray(input_data)

# reshaping the data 
input_data_reshaped = input_data_asnumpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
    print ('High value customer')
else:
    print('low value customer')


# Saving the trained model

import pickle    
filename = 'trained_model.saved'
pickle.dump(model,open(filename, 'wb'))

# loading the saved model 
loaded_model = pickle.load(open('trained_model.saved','rb'))

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
                           
                    

   
    







        
        
        
        
        
        
        
        
        
        
        
        
        
        
























