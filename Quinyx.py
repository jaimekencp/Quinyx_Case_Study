import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Parse vessel data into a pandas dataframe
vessel_df = pd.read_excel("VesselData.xlsx", sheet_name="VesselData")

# Drop arrival time columns
vessel_df.drop(['eta','ata','atd','earliesteta','latesteta'], axis=1, inplace=True)

# Get one hot encoding of stevedorenames
one_hot = pd.get_dummies(vessel_df['stevedorenames'])
# Drop stevedorenames as it is now encoded
vessel_df.drop('stevedorenames',axis = 1, inplace=True)
# Join the encoded df
vessel_df = vessel_df.join(one_hot)

# make traveltype binary
vessel_df['traveltype'].replace({'ARRIVAL' : 1,'SHIFT' : 0},inplace=True)

# check isremarkable column only has one unique value, if so drop it
vessel_df['isremarkable'].nunique()
# only 1 unique values, therefore we can drop this column as to avoid overfitting in our model
vessel_df.drop('isremarkable',axis=1, inplace=True)

# Remove hasnohamis column as it's al NA
vessel_df.drop('hasnohamis', axis=1, inplace=True)

# Remove rows with Nan values
vessel_df.dropna(axis=0, inplace=True) 

# Split the data such that: 60% - train set, 20% - validation set, 20% - test set
train, validate, test = np.split(vessel_df.sample(frac=1, random_state=42), [int(.6*len(vessel_df)), int(.8*len(vessel_df))])

# Seperate what we want to predict from our dataframe
to_predict = ['discharge1', 'load1', 'discharge2', 'load2', 'discharge3', 'load3', 'discharge4', 'load4']
y = train[to_predict]
X = train.drop(to_predict, axis=1)


true_test_values = test[to_predict]
test.drop(to_predict,axis=1, inplace= True)

# define model
model = DecisionTreeRegressor()
# fit model to training set
model.fit(X, y)
# make predictions on test set
predictions = model.predict(test)
# convert ndarray to pandas df
predictions_df = pd.DataFrame(predictions, columns = ['discharge1',  'load1',  'discharge2',  'load2',  'discharge3',  'load3',  'discharge4',  'load4'])

# Here I'm using MSE as my metric to evaluate my predictions between each variable
discharge1_MSE = mean_squared_error(predictions_df ['discharge1'], true_test_values['discharge1']) # 111578924.44884287
load1_MSE = mean_squared_error(predictions_df ['load1'], true_test_values['load1']) # 74595.59607186358
discharge2_MSE = mean_squared_error(predictions_df ['discharge2'], true_test_values['discharge2']) # 83816012.9732034
load2_MSE = mean_squared_error(predictions_df ['load2'], true_test_values['load2']) # 46283.759439707675
discharge3_MSE = mean_squared_error(predictions_df ['discharge3'], true_test_values['discharge3']) # 167117976.8909866
load3_MSE = mean_squared_error(predictions_df ['load3'], true_test_values['load3']) # 57472.55846528624
discharge4_MSE = mean_squared_error(predictions_df ['discharge4'], true_test_values['discharge4']) #108981733.6582018
load4_MSE = mean_squared_error(predictions_df ['load4'], true_test_values['load4']) #141412818.61909595