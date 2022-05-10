# Business Scenario
# This mini-project builds off of the customer types example that you have
# already worked on. Previously, you constructed a K-Nearest neighbors (KNN) model to
# predict the customers category. Your model predicts the following:

# 1 for "loyal", 2 for "impulse", 3 for "discount", 4 for "need-based", and 5 for "wandering".

# Your features in your KNN model were the following:
# 1. 'householdincome'
# 2. 'householdsize'
# 3. 'educationlevel'
# 4. 'gender' where 0 is for Male and 1 is for Female.

# You have a .pkl file containing your KNN model (from Task 8 in that activity).
# Your are tasked with using that model to predict the customer groups for all
# of the data contained in the customers.txt file.

# Your job is to make an Excel file containing the predictions and
# the predicted probabilities. A sample of how your data should look in Excel is contained in the
# output.png file.

# Task 1
# Review the code from the customer types KNN example. Understand the final model
# and how it was created.

# Task 2
# Import the .pkl file containing the trained model.
# There are two .pkl files in the .zip folder. The first file was created with an
# older version of sklearn and the "UPDATED" file was created with a newer version
# of sklearn. The model is identical, so you only have to load one of them.
# If you get a Traceback error similar to the following:
# ModuleNotFoundError: No module named 'sklearn.neighbors._dist_metrics'
# Then, try the other file. One of them *should* work.

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#importing pklfile using joblib
pklfile = 'knn-model_customer_types-UPDATED.pkl'
loaded_model = joblib.load(open(pklfile, 'rb'))

# Task 3
# Read in the data in the customers.txt file
df = pd.read_csv('customers.txt', delimiter='|', index_col=False, dtype=str)
print(df.isnull().sum())

#changing values from text to numeric using map function and reassigning file
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df.to_csv('customers_cleaned.txt', index=False)

#reading in "cleaned" file
#df = pd.read_csv('customers_cleaned.txt', delimiter=',', dtype=float)

#scaling our data because revenue is huge compared to other values
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
print(scaled_df)

#reassigning binary values to gender because scaled gender doesn't make sense
scaled_df[:, [3]] = df.iloc[:,3].values.reshape([500,1])
print(scaled_df)

# Task 4
# Make the predictions and output the results to excel.
my_categories = {"loyal": 1, "impulse": 2, "discount": 3, "need-based": 4, "wandering": 5}
my_features = scaled_df
np.set_printoptions(suppress=True)
print(my_features)
prediction = loaded_model.predict(my_features)
print(prediction)
probabilities = loaded_model.predict_proba(my_features)
print(probabilities)

#converting numeric values back to text and creating a new dataframe -- preparing for excel output
my_categories = {1: "loyal", 2: "impulse", 3: "discount", 4: "need-based", 5: "wandering"}
gender = {0: "Male", 1: "Female"}
df['prediction'] = prediction.tolist()
df['prediction'] = df['prediction'].map(my_categories)
df['gender'] = df['gender'].map(gender)
df1 = pd.DataFrame(probabilities, columns=['loyal_prob', 'impulse_prob', 'discount_prob','need-based_prob', 'wandering_prob'], dtype=float)
df1.head()

#joining the new dataframe created in previous step to old dataframe
df = df.join(df1)
print(df)

#outputting to excel
df.to_excel('excel_output.xlsx', index=False)