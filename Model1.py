
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

data =pd.read_csv('CarPrice_training.csv')
data_corr = data.iloc[:,:]
#first preprocessing # Remove missing value
data.dropna(axis=0,how='any',inplace=True)
data['cylindernumber']=data['cylindernumber'].apply(convert_no_num)
data['doornumber'] = data['doornumber'].apply(convert_no_num)
X=data.iloc[:,0:24]
Y=data['price']
cols=('CarName','aspiration','carbody','drivewheel','enginelocation','enginetype','fuelsystem')
X=Feature_Encoder(X,cols);
X['fueltype']= OneHotEncoder().fit_transform(X).toarray()
Y=np.expand_dims(Y, axis=1)
X=featureScaling(np.array(X),0,1);

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)
#calculate Correlation between features
corr = data_corr.corr()
top_feature = corr.index[abs(corr['price']>0.5)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data_corr[top_feature].corr()
sns.heatmap(top_corr, annot=True)
#plt.show()
poly_features = PolynomialFeatures(degree=4)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_train_predicted = poly_model.predict(X_train_poly)
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
true_price_value=np.asarray(y_test)[0]
predicted_price_value=prediction[0]
print('True price value: ' + str(true_price_value))
print('Predicted price value: ' + str(predicted_price_value))





