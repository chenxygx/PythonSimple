from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(20, 64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(64, 64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(64, 1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_tran, y_train, nb_epoch=20, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)
