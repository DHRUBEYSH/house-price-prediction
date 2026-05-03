import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error


# df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\House Price Prediction\\Data\\Raw Data\\archive (1).zip")
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\House Price Prediction\\Data\\cleaned Data\\Housing.csv")
location_encoder = LabelEncoder()
City_encoder = LabelEncoder()
df['Location'] = location_encoder.fit_transform(df['Location'])
df['City'] = City_encoder.fit_transform(df['City'])

x = df.drop(columns=['Price','BED','WashingMachine','Microwave'], axis=1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=42
)


xgb_model = XGBRegressor(n_estimators=700,learning_rate=0.1,max_depth=5,subsample=0.8,min_child_weight=1,gamma=0.1,colsample_bytree=0.7)

xgb_model.fit(X_train,y_train)

y_pred = xgb_model.predict(X_test)

print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
kf = KFold(n_splits=10, shuffle=True, random_state=42)
score = cross_val_score(xgb_model,x,y,cv=kf,scoring='r2')

print(score)
print(score.mean())
print(score.std())


import joblib
import os
os.makedirs("Models",exist_ok=True)
joblib.dump(xgb_model,"Models/model.pkl")
joblib.dump(location_encoder,"Models/location_encoder.pkl")
joblib.dump(City_encoder,"Models/City_encoder.pkl")
joblib.dump(x.columns.to_list(),"Models/features.pkl")

print("Model and encoders saved successfully!")