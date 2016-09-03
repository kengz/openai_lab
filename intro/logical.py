import tensorflow as tf
import tflearn

# Logical NOT operator
X = [[0.], [1.]]
Y = [[1.], [0.]]

# try to change up X again
with tf.Graph().as_default():
  # this shape cuz the next layer must take 2D+ tensor
  g = tflearn.input_data(shape=[None, 1])
  g = tflearn.fully_connected(g, 128, activation='linear')
  g = tflearn.fully_connected(g, 128, activation='linear')
  g = tflearn.fully_connected(g, 1, activation='sigmoid')
  g = tflearn.regression(g, optimizer='sgd', learning_rate=2., loss='mean_square')

  # Train
  m = tflearn.DNN(g)
  m.fit(X, Y, n_epoch=1000, snapshot_epoch=False)

  print('Testing NOT operator')
  print('Prediction', m.predict(X))
  print('Target Result', Y)

# also try to draw tensorboard
# safe load models