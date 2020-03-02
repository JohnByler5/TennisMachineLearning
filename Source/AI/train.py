from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from time import time


epochs = 10
batch_size = 64
name = f'tennis_model_{epochs}_{batch_size}_{time()}'
dropout = 0.2
nodes = 256
activation = 'relu'
lr = 1e-3                                                                           
decay = 1e-6
optimizer = Adam(lr=lr, decay=decay)
loss = 'binary_crossentropy'
metrics = ['accuracy']


def get_data():
	t_x = []
	t_y = []
	v_x = []
	v_y = []
	return t_x, t_y, v_x, v_y


def create_model():
	tf_model = Sequential()
	tf_model.add(Dense(nodes, activation=activation))
	tf_model.add(Dropout(dropout))
	tf_model.add(Dense(nodes, activation=activation))
	tf_model.add(Dropout(dropout))
	tf_model.add(Dense(nodes, activation=activation))
	tf_model.add(Dropout(dropout))
	tf_model.add(Dense(2, activation='softmax'))
	return tf_model


train_x, train_y, validation_x, validation_y = get_data()
model = create_model()
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
tensorboard = TensorBoard(log_dir=f'logs/{name}')
filepath = 'Tennis_Final-{epoch:02d)-{val_acc:.3f}'
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])
