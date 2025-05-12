import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import raw_data,processed_data
import os


#Loading Data

df=pd.read_csv(raw_data)
print(df.info())


def data_prep(df):
  cat_cols = ['Campaign_Type','Duration', 'Target_Audience', 'Channel_Used', 'Location', 'Language', 'Customer_Segment']
  df_prep = df.drop(['Company','Date'],axis=1)
  df_prep=df_prep.set_index('Campaign_ID')
  df_prep['Acquisition_Cost'] = df_prep['Acquisition_Cost'].str.replace(r'[^\d.]','',regex=True).astype(float)
  df_prep = pd.get_dummies(df_prep, columns=cat_cols, drop_first=False)
  return df_prep

def target_category_quartile(df_prep,target='Conversion_Rate'):
  q1=df_prep[target].quantile(0.25)
  q3=df_prep[target].quantile(0.5)
  df_prep['Conversion']=(df_prep['Conversion_Rate'].apply(lambda x:'Low' if x<q1 else('Medium' if q1<x<q3 else 'High')))
  df_prep.drop('Conversion_Rate',axis=1,inplace=True)
  return df_prep


def data_split(df_encoded,target='Conversion',test_size=0.4,random_state=42):
  df_train, df_test = train_test_split(df_encoded, test_size=test_size, shuffle=True, random_state=random_state)
  df_train.to_json(os.path.join(processed_data,'processed_train.json'))
  df_test.to_json(os.path.join(processed_data,'processed_test.json'))



df_prep=data_prep(df)
df_encoded=target_category_quartile(df_prep)
data_split(df_encoded)
