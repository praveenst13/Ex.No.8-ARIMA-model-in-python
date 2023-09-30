# Ex.No.8-ARIMA-model-in-python
## AIM:
 To Implementation of ARIMA Model Using Python.
## Procedure:
1.Import necessary libraries

2.Read the CSV file,Display the shape and the first 20 rows of the dataset

3.Set the figure size for plots

4.Import the SARIMAXfrom statsmodels.tsa.statespace.sarimax 

5.Calculate root mean squared error

6.Calculate mean squared error

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
df=pd.read_csv("/content/TEMP.csv",index_col=0,parse_dates=True)
data1=df
df.pop("JAN")
df.pop("FEB")
df.pop("MAR")
df.pop("APR")
df.pop("MAY")
df.pop("JUN")
df.pop("JUL")
df.pop("AUG")
df.pop("SEP")
df.pop("OCT")
df.pop("NOV")
df.pop("DEC")
df.pop("JAN-FEB")
df.pop("MAR-MAY")
df.pop("OCT-DEC")
df.pop("JUN-SEP")
df.shape
df.head()
x=df.values
x
df.plot(figsize=(10,5))
from statsmodels.tsa.stattools import adfuller

dftest= adfuller(df['ANNUAL'],autolag='AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ",dftest[1])
print("3. Number Of Lags : ",dftest[2])
print("4.Num of observation used FOr ADF Regression  and Critical value Calculation :",dftest[3])
for key,val in dftest[4].items():
     print("\t",key, ":",val)
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(df,trace=True,suppress_warnings=True)
stepwise_fit.summary()
train=x[:len(df)-12]
test=x[len(df)-12:]
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train,order = (0, 1, 1),seasonal_order =(2, 1, 1, 12))

result = model.fit()
result.summary()
start=len(train)
end=len(train)+len(test) -1
pred=result.predict(start,end,type='levels')
pred
plt.plot(pred)
plt.plot(test)
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
rmse(test, pred)

# Calculate mean squared error
mean_squared_error(test, pred)
pred
```
## Output:
### df.shape
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/bf155ea1-7aae-400a-9757-945e2c9931a0)
### df.head()
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/1598183f-06dd-45f6-b9b5-3a23661f5177)
### x values
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/104bdf19-5c3f-4cd3-b410-a6fba7cebb0e)
### df.plot()
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/7503b243-ae53-4904-b09d-1645b802dd23)
### key,val
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/ce598a58-d652-424c-ae84-98c496f63d1b)
### stepwise_fit.summary()
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/3c4828d0-dd03-41c6-b1d8-89a6fd6fef46)
### result.summary()
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/4a0f8d78-a0dd-4d9f-aa2d-f99ce0204a27)
### pred
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/1f7859a4-71bc-4b4a-b655-4450b43550e5)
### plt.plot(pred),plt.plot(test)
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/73bd08e3-7514-4f78-825e-92aa9c21564f)

### Calculate mean squared error
![image](https://github.com/praveenst13/Ex.No.8-ARIMA-model-in-python/assets/118787793/447476f0-19f7-4ebf-bfcc-c9df9081fe1b)


## Result:
Thus we have successfully implemented the ARIMA Model using above mentioned program.
