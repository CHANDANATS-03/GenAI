
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import datetime
#apple stack, we can change to "TSLA',"GOOG","MSFT"
thicker="AAPL"
end_date=datetime.date.today()-datetime.timedelta(days=1)
start_date=end_date-datetime.timedelta(days=365*10)

df=yf.download(thicker,start=start_date,end=end_date)
print(df.head())
print(df.tail())

#use multiple features

features=["Open","High","Low","Close","Volume"]
data=df[features].values #convert to numpy array
print(data[:5])
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data)
scaled_data[:5]

#step 4:create dataset function
def create_dataset(data,time_step=60):
  x,y=[],[]
  for i in range(len(data)-time_step-1):
    x.append(data[i:(i+time_step)])#last 'time_step' days of all features
    y.append(data[i+time_step,[0,3]])   #predict[Open,close]
  return np.array(x),np.array(y)
time_step=60
x,y=create_dataset(scaled_data,time_step)
#Reshaping into 3D:[sample,time_step,features]
x=x.reshape(x.shape[0],x.shape[1],x.shape[2])

#step 5: Build GRU Model
model=Sequential()
model.add(GRU(units=64,return_sequences=True,input_shape=(time_step,x.shape[2])))
model.add(GRU(units=64))
model.add(Dense(units=2))   #Output both open and close
model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error')

#step 7: Predict Todays Open and close
last_60_days=scaled_data[-time_step:]  #last 60 rows
#print("Last 60 days data:",last_60_days)
last_60_days=last_60_days.reshape(1,time_step,x.shape[2])  #reshape to 3D and include all features and batch sizee
print("Reshaped last last 60 days data:",last_60_days)
predicted_prices=model.predict(last_60_days)
print("Scaled predicted prices (Open,Close):",predicted_prices)
#Rebuild full 5-features array for inverse transform
# Format:[Open,high.low,close,volume]
dummy=np.zeros((1,5))
dummy[0,0]=predicted_prices[0,0] #Open
dummy[0,3]=predicted_prices[0,1]  #Close
print("Dummy for inverse transform:",dummy)
#inverse transform using_same scaler
predicted_prices_real=scaler.inverse_transform(dummy)[:,[0,3]]  #get only open &close
print(f"Predicted  Opening Price for Today:${predicted_prices_real[0][0]:.2f}")
print(f"Predicted Closing Price for Today:${predicted_prices_real[0][1]:.2f}")