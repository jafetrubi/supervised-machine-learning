import pymongo as pym
import pandas as pd
from bson.objectid import ObjectId
import statistics as stats
import numpy as np
import researchpy as rp
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')

# Connect to the MongoDB server
try:
  client = pym.MongoClient('172.28.8.65', 27017)
  client.server_info()
except:
  print('Something went wrong connecting to Mongo')

db = client['project']
customers = db.customers

# Check to see if there are duplicate records
val = list(customers.aggregate([
   {"$group": {"_id": "$net_user",
               "uniqueIds": {"$addToSet": "$net_user"},
               "uniques": {"$sum": 1}
               }
   },
   {"$match": {"uniques": {"$gt": 1}}}]))

# Load data from Mongo into Pandas
index = []
my_data = []

# Separating code into lists for quicker loading
first_columns = ['age', 'male']
my_friends = ['friend_cnt', 'avg_friend_age', 'avg_friend_male', 'friend_country_cnt', 'subscriber_friend_cnt']
second_columns = ['songsListened', 'lovedTracks','posts','playlists', 'shouts']
my_delta1 = ['delta1_friend_cnt','delta1_avg_friend_age', 'delta1_avg_friend_male', 'delta1_friend_country_cnt',
          'delta1_subscriber_friend_cnt', 'delta1_songsListened', 'delta1_lovedTracks', 'delta1_posts', 'delta1_playlists',
             'delta1_shouts', 'delta1_good_country']
third_columns = ['adopter','tenure','good_country']
my_delta2 = ['delta2_friend_cnt','delta2_avg_friend_age', 'delta2_avg_friend_male', 'delta2_friend_country_cnt',
             'delta2_subscriber_friend_cnt', 'delta2_songsListened', 'delta2_lovedTracks', 'delta2_posts', 'delta2_playlists',
             'delta2_shouts', 'delta2_good_country']
my_columns = first_columns + my_friends + second_columns + my_delta1 + third_columns + my_delta2

for c in customers.find({}):
   my_list = []
   index.append(c.get('net_user'))
   gender = c.get('male')
   age = c.get('age')
   friends = c.get('friends')
   delta1 = c.get('delta1')
   delta2 = c.get('delta2')
   if age is not None:
       my_list.append(age)
   else:
       my_list.append(np.nan)
   if gender is not None:
       my_list.append(gender)
   else:
       my_list.append(np.nan)
   for x in my_friends:
       if friends.get(x) == 'NULL' or friends.get(x) is None:
           my_list.append(np.nan)
       else:
           my_list.append(friends.get(x))
   for a in second_columns:
       my_list.append(c.get(a))
   for b in my_delta1:
       if delta1.get(b) == 'NULL' or delta1.get(b) is None:
           my_list.append(np.nan)
       else:
           my_list.append(delta1.get(b))
   for d in third_columns:
       my_list.append(c.get(d))
   for e in my_delta2:
       if delta2.get(e) == 'NULL' or delta2.get(e) is None:
           my_list.append(np.nan)
       else:
           my_list.append(delta2.get(e))
   my_data.append(my_list)

my_df = pd.DataFrame(my_data, index=index, columns=my_columns)

# Assigning zero or average values to replace null values
zero_columns = ['subscriber_friend_cnt','playlists','delta1_friend_cnt','delta1_avg_friend_age',
                'delta1_avg_friend_male', 'delta1_friend_country_cnt', 'delta1_subscriber_friend_cnt',
                'delta1_songsListened',	'delta1_lovedTracks','delta1_posts','delta1_playlists',
                'delta1_shouts', 'delta1_good_country', 'adopter', 'tenure', 'good_country',
                'delta2_friend_cnt', 'delta2_avg_friend_age','delta2_avg_friend_male',
                'delta2_friend_country_cnt','delta2_subscriber_friend_cnt','delta2_songsListened',
                'delta2_lovedTracks','delta2_posts','delta2_playlists','delta2_shouts','delta2_good_country']
my_df[['age','avg_friend_age']] = my_df[['age','avg_friend_age']].fillna(value=24)
my_df['avg_friend_male'] = my_df['avg_friend_male'].fillna(value=0.6)
my_df[zero_columns] = my_df[zero_columns].fillna(0)
my_df['shouts'] = my_df['shouts'].fillna(21)
my_df['male'] = my_df['male'].fillna(np.random.randint(2))
my_df['friend_cnt'] = my_df['friend_cnt'].fillna(12)
my_df['friend_country_cnt'] = my_df['friend_country_cnt'].fillna(3)

#my_df.to_csv('output.csv')

np.set_printoptions(suppress=True)
eligible_features = my_delta1
my_feature_list = random.sample(eligible_features, 6)
#can use all 11 for other models except for KNN

# Creating our Numpy arrays from feature and targets
my_features = my_df[my_feature_list].values
my_targets = my_df['adopter'].values

# Splitting data into training and testing
from sklearn.model_selection import train_test_split
f_train, f_test, t_train, t_test = train_test_split(my_features, my_targets, test_size=0.30, random_state=60)

# Creating SMOTE oversampled data
from imblearn.over_sampling import SMOTE
over_sampler = SMOTE(random_state=23)
smote_feature, smote_target = over_sampler.fit_resample(f_train, t_train)
# print(np.bincount(smote_target))
# print(np.bincount(t_train.flatten()))

# Creating NearMiss undersampled data
from imblearn.under_sampling import NearMiss
under_sampler = NearMiss()
nm_feature, nm_target = under_sampler.fit_resample(f_train, t_train)
# print(np.bincount(nm_target))
# print(np.bincount(t_train.flatten()))

# To graphically display the confusion matrix, execute the following function. Then, call that function!
def confusion(test, predict, title, labels, size=2):
    """Plot the confusion matrix to make it easier to interpret.
       This function produces a colored heatmap that displays the relationship
        between predicted and actual types from a machine learning method."""

    # Make a 2D histogram from the test and result arrays.
    # pts is essentially the output of the scikit-learn confusion_matrix method.
    pts, xe, ye = np.histogram2d(test, predict, bins=size)

    # For simplicity we create a new DataFrame for the confusion matrix.
    pd_pts = pd.DataFrame(pts.astype(int), index=labels, columns=labels)

    # Display heatmap and add decorations.
    hm = sns.heatmap(pd_pts, annot=True, fmt="d")
    hm.axes.set_title(title, fontsize=20)
    hm.axes.set_xlabel('Actual (True) Target', fontsize=18)
    hm.axes.set_ylabel('Predicted Label', fontsize=18)

# Train first KNN model
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
num_neighbors = 100
#square root of training split size
knn = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean', weights='uniform')
knn.fit(f_train, t_train)
score_test = 100 * knn.score(f_test, t_test)
score_train = 100 * knn.score(f_train, t_train)
print(f'\nKNN ({num_neighbors} neighbors) prediction accuracy with test data = {score_test:.1f}%\n'
      f'KNN ({num_neighbors} neighbors) prediction accuracy with training data = {score_train:.1f}%')
predicted_labels = knn.predict(f_test)
print(f'Base KNN model confusion matrix: \n{confusion_matrix(t_test, predicted_labels)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), predicted_labels, 'Base KNN Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'KNN_confusion.png'))
plt.close('all')

# Oversampled KNN model
oversampled_knn = knn.fit(smote_feature,smote_target)
smote_predictions = oversampled_knn.predict(f_test)
os_score_test = 100 * knn.score(f_test, t_test)
os_score_train = 100 * knn.score(f_train, t_train)
print(f'\nOversampled KNN ({num_neighbors} neighbors) prediction accuracy with test data = {os_score_test:.1f}%\n'
      f'Oversampled KNN ({num_neighbors} neighbors) prediction accuracy with training data = {os_score_train:.1f}%')
print(f'Oversampled KNN model confusion matrix: \n{confusion_matrix(t_test, smote_predictions)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), smote_predictions, 'Oversampled KNN Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'osKNN_confusion.png'))
plt.close('all')

# Undersampled KNN model
us_model = knn.fit(nm_feature, nm_target)
us_score_test = 100 * us_model.score(f_test, t_test)
us_score_train = 100 * us_model.score(f_train, t_train)
us_predictions = us_model.predict(f_test)
print(f'\nUndersampled KNN ({num_neighbors} neighbors) prediction accuracy with test data = {us_score_test:.1f}%\n'
      f'Undersampled KNN ({num_neighbors} neighbors) prediction accuracy with training data = {us_score_train:.1f}%')
print(f'Undersampled KNN model confusion matrix: \n{confusion_matrix(t_test, us_predictions)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), us_predictions, 'Undersampled KNN Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'usKNN_confusion.png'))
plt.close('all')

# Base Logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# The smaller the C hyper-parameter should result in a smoother curve (i.e., less chance for over-fitting).
# Let's eventually both of the following
# C=1E6
# Or C=1
estimator = LogisticRegression(C=1000000, random_state=60, solver='liblinear')
# Fit a new model to the training model and predict on the test data.
lr_model = estimator.fit(f_train, t_train)
predicted = lr_model.predict(f_test)
lr_score = 100.0 * accuracy_score(t_test, predicted)
print(f'\nBase Logistic Regression score using the logistic regression algorithm = {lr_score:4.1f}%')
print(f'Base Logistic Regression confusion matrix: \n{confusion_matrix(t_test.reshape(t_test.shape[0]), predicted)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), predicted, 'Base Logistic Regression Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'LR_confusion.png'))
plt.close('all')

# Oversampled Logistic Regression model
smote_model = estimator.fit(smote_feature, smote_target)
predicted = smote_model.predict(f_test)
os_lr_score = 100.0 * accuracy_score(t_test, predicted)
print(f'\nOversampled Logistic Regression score using the logistic regression algorithm = {os_lr_score:4.1f}%')
print(f'Oversampled Logistic Regression confusion matrix: \n{confusion_matrix(t_test.reshape(t_test.shape[0]), predicted)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), predicted, 'Oversampled Logistic Regression Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'osLR_confusion.png'))
plt.close('all')

# Undersampled Logistic Regression model
us_model = estimator.fit(nm_feature, nm_target)
predicted = us_model.predict(f_test)
us_lr_score = 100.0 * accuracy_score(t_test, predicted)
print(f'\nUndersampled Logistic Regression score using the logistic regression algorithm = {us_lr_score:4.1f}%')
print(f'Undersampled Logistic Regression confusion matrix: \n{confusion_matrix(t_test.reshape(t_test.shape[0]), predicted)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), predicted, 'Undersampled Logistic Regression Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'usLR_confusion.png'))
plt.close('all')

# Base Decision Tree model
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(max_depth=2, max_features=3, random_state=70)
dt_model = dt_classifier.fit(f_train, t_train)
predicted_labels = dt_classifier.predict(f_test)
score_test = 100 * dt_model.score(f_test, t_test)
score_train = 100.0 * dt_model.score(f_train, t_train)
print(f'\nBase Decision Tree algorithm test score: {score_test:.1f}%')
print(f'\nBase Decision Tree algorithm training score: {score_train:.1f}%')
print(f'Base Decision Tree confusion matrix: \n{confusion_matrix(t_test, predicted_labels)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), predicted_labels, 'Base Decision Tree Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'DT_confusion.png'))
plt.close('all')

# Oversampled Decision Tree model
os_model = dt_classifier.fit(smote_feature, smote_target)
predicted_labels = os_model.predict(f_test)
score_test = 100 * os_model.score(f_test, t_test)
score_train = 100 * os_model.score(f_train, t_train)
print(f'\nOversampled Decision Tree algorithm test score: {score_test:.1f}%')
print(f'\nOversampled Decision Tree algorithm training score: {score_train:.1f}%')
print(f'Oversampled Decision Tree confusion matrix: \n{confusion_matrix(t_test, predicted_labels)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), predicted_labels, 'Oversampled Decision Tree Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'osDT_confusion.png'))
plt.close('all')

# Undersampled Decision Tree model
us_model = dt_classifier.fit(nm_feature, nm_target)
predicted_labels = us_model.predict(f_test)
score_test = 100 * us_model.score(f_test, t_test)
score_train = 100 * us_model.score(f_train, t_train)
print(f'\nUndersampled Decision Tree algorithm test score: {score_test:.1f}%')
print(f'\nUndersampled Decision Tree algorithm training score: {score_train:.1f}%')
print(f'Undersampled Decision Tree confusion matrix: \n{confusion_matrix(t_test, predicted_labels)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), predicted_labels, 'Undersampled Decision Tree Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'usDT_confusion.png'))
plt.close('all')

for name, val in zip(my_delta1, us_model.feature_importances_):
    print(f'Undersampled Decision Tree feature importance: {name} importance = {100.0*val:5.2f}%')

# Base Random Forest
from sklearn.ensemble import RandomForestClassifier
my_classifier = RandomForestClassifier(bootstrap=True, n_estimators=300, oob_score=True)
rf_model = my_classifier.fit(f_train, t_train)
model_predictions = rf_model.predict(f_test)
score_test = 100 * rf_model.score(f_test, t_test)
score_train = 100 * rf_model.score(f_train, t_train)
print(f'\nRandom forest prediction accuracy with testing data = {score_test:.1f}%.')
print(f'Random forest prediction accuracy with training data = {score_train:.1f}%.')
print(f'Random forest confusion matrix: \n{confusion_matrix(t_test, model_predictions)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), model_predictions, 'Base Random Forest Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'RF_confusion.png'))
plt.close('all')

# Oversampled Random Forest
os_model = my_classifier.fit(smote_feature, smote_target)
os_predictions = os_model.predict(f_test)
score_test = 100 * os_model.score(f_test, t_test)
score_train = 100 * os_model.score(f_train, t_train)
print(f'\nOversampled Random Forest prediction accuracy with testing data = {score_test:.1f}%.')
print(f'Oversampled Random Forest prediction accuracy with training data = {score_train:.1f}%.')
print(f'Oversampled Random Forest confusion matrix: \n{confusion_matrix(t_test, os_predictions)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), os_predictions, 'Oversampled Random Forest Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'osRF_confusion.png'))
plt.close('all')

# Undersampled Random Forest
us_model = my_classifier.fit(nm_feature, nm_target)
us_predictions = us_model.predict(f_test)
score_test = 100 * us_model.score(f_test, t_test)
score_train = 100 * us_model.score(f_train, t_train)
print(f'\nUndersampled Random Forest prediction accuracy with testing data = {score_test:.1f}%.')
print(f'Undersampled Random Forest prediction accuracy with training data = {score_train:.1f}%.')
print(f'Undersampled Random Forest confusion matrix: \n{confusion_matrix(t_test, us_predictions)}')

# Call visualization function
plt.close('all')
target_names = ['Not an Adopter', 'Adopter']
confusion(t_test.flatten(), us_predictions, 'Undersampled Random Forest Model', target_names)
plt.savefig(os.path.join(os.getcwd(), 'usRF_confusion.png'))
plt.close('all')





