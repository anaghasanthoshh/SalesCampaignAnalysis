import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import raw_data,processed_data
from data_preprocessing import data_prep

df=pd.read_csv(raw_data)

#Correlation
df_encoded=data_prep(df)
df_encoded_numeric = df_encoded.select_dtypes(include=['float64', 'int64'])
corr_matrix = df_encoded_numeric.corr()
target_correlation = corr_matrix['Conversion_Rate'].sort_values(ascending=False)
sns.heatmap(corr_matrix,annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
print(target_correlation)


#Box plot of conversion rate
plt.figure(figsize=(8,6))
plt.boxplot(df_encoded['Conversion_Rate'])
plt.show()





# #Scatter plots
# numeric_cols=[col for col in df_encoded_numeric.columns ]
# for col_x in numeric_cols:
#     for col_y in numeric_cols:
#
#         plt.scatter(df[col_x],df[col_y])
#         plt.title(f'{col_x} vs {col_y}')
#         plt.show()



