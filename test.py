from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


model = Sequential()

from keras.layers import Dense, Activation

x = np.linspace(0,6.0,100)
y = np.sin(x)

model.add(Dense(units=20, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(units=20, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error',
              optimizer='sgd')

model.fit(x,y,epochs=5000,batch_size=10)

yp = model.predict(x,batch_size=10)

plt.plot(x,y,label='truth')
plt.plot(x,yp,label='pred')
plt.legend()
plt.show()




