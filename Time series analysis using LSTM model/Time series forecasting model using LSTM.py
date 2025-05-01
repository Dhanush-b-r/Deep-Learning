import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

air_passengers=pd.read_csv(r'E:\Digital_University\Sem_2\Deep_Learning\AirPassengers.csv')
air_passengers.dtypes
air_passengers.describe()

air_passengers.columns=['Month','Passengers']

air_passengers.isnull().sum()

air_passengers['Month']=pd.to_datetime(air_passengers['Month'])
air_passengers.set_index('Month', inplace=True)



# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(air_passengers.index, air_passengers['Passengers'], label='Passengers')
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.title("Time Series Graph")
plt.legend()
plt.show()

#scaling the data
air_passengers = air_passengers['Passengers'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(air_passengers)
scaled_data.shape

x=[]
y=[]
def create_seq(df,b):
    for i in range(len(df)-b):
        x.append(df[i:i+b])
        y.append(df[b+i])

create_seq(scaled_data,12)

x,y=np.array(x),np.array(y)
x.shape

#model building
model = Sequential([
    LSTM(100, activation='tanh', input_shape=(x.shape[1], x.shape[2])),
    Dense(1)
]) 

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=8, validation_split=0.2)

y_pred = model.predict(x)
y_pred_inv = scaler.inverse_transform(y_pred)
y_true_inv = scaler.inverse_transform(y)

plt.plot(y_true_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title('LSTM Forecast - Passengers')
plt.show()