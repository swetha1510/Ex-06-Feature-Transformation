## Ex-06-Feature-Transformation
## Aim:
1.To read and perform feature transformation for the given dataset.

## Explanation:
Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column (feature) and transform the values, which are useful for our further analysis. It is a technique by which we can boost our model performance.

## Algorithm:
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Transformation techniques to all the features of the data set

## STEP 4
Save the data to the file

## Program:

Name : SWETHA P
Register numnber : 212222100053

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()


## OUTPUT:
## Dataset:
![dsex61](https://user-images.githubusercontent.com/120623583/234917720-462e847b-2ea5-43ef-bce7-e338fa5dbd7d.png)

## Head:
![dsex62](https://user-images.githubusercontent.com/120623583/234917796-0802f3fc-d244-4256-9c32-1b612cebf703.png)

## null data:
![dsex63](https://user-images.githubusercontent.com/120623583/234917829-1cf696d4-f94f-44e3-b2d4-2e0f40ce4ced.png)

## Information:
![dsex64](https://user-images.githubusercontent.com/120623583/234917853-b5378822-86fa-4d59-a931-51a993591c89.png)

##Description:
![dsex65](https://user-images.githubusercontent.com/120623583/234917866-0a7a99e0-3aac-43c1-b48c-7195214d058a.png)

## Highly Positive Skew:
![dsex66](https://user-images.githubusercontent.com/120623583/234917957-f94c164f-ff2a-431c-9a93-721441c7926f.png)

## Highly Negative Skew:
![dsex67](https://user-images.githubusercontent.com/120623583/234917984-ab38a36d-724b-4853-b8ee-6c07b2e664e2.png)

## Moderate Positive Skew:
![dsex68](https://user-images.githubusercontent.com/120623583/234918009-4def52ee-beec-446f-b6b3-f30ab19ddea6.png)

## Moderate Negative Skew:
![dsex69](https://user-images.githubusercontent.com/120623583/234918031-21529aa2-5226-444c-9b11-4de4e6e1ee86.png)

## Log of Highly Positive Skew:
![dsex610](https://user-images.githubusercontent.com/120623583/234918064-6c1e84c8-1427-497a-90c3-00d852c73ef4.png)

## Log of Moderate Positive Skew:
![dsex611](https://user-images.githubusercontent.com/120623583/234918270-32c5749f-fd27-44a2-a76b-8a05b1fc4c41.png)

## Reciprocal of Highly Positive Skew:
![dsex612](https://user-images.githubusercontent.com/120623583/234918289-615a08ec-dd77-415c-b9b4-3022b48526a1.png)

## Square root tranformation:
![dsex613](https://user-images.githubusercontent.com/120623583/234918311-136647a6-e41b-4d8e-9d58-d935e5dd62cc.png)

## Power transformation of Moderate Positive Skew:
![dsex614](https://user-images.githubusercontent.com/120623583/234918351-062dada3-6f4a-437c-a37c-a94ba0b5630e.png)

## Power transformation of Moderate Negative Skew:
![dsex615](https://user-images.githubusercontent.com/120623583/234918389-f8f3fb09-4ebd-4817-9265-0207eb00b0d1.png)

## Quantile transformation:
![dsex616](https://user-images.githubusercontent.com/120623583/234920891-98126916-c467-40fa-a6e7-43df2a0e4fcd.png)

## Result:
Thus, Feature transformation is performed and executed successfully for the given dataset.
    
    
    
    
    
    
    
    
