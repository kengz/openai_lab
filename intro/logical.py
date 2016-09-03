import tensorflow as tf
import tflearn


# train a model, given a graph g, data X, Y
def train_model(g, X, Y, model_name='sample_model.tflearn'):
    m = tflearn.DNN(g)
    m.fit(X, Y, n_epoch=1000, snapshot_epoch=False)
    m.save(model_name)
    # m.load(model_name)

    print('Testing NOT operator')
    print('Prediction', m.predict(X))
    print('Target Result', Y)


# Logical NOT operator
def run_NOT():
    X = [[0.], [1.]]
    Y = [[1.], [0.]]

    # this shape cuz the next layer must take 2D+ tensor
    g = tflearn.input_data(shape=[None, 1])
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(
        g, optimizer='sgd', learning_rate=2., loss='mean_square')

    train_model(g, X, Y)


# Logical OR operator
def run_OR():
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y = [[0.], [1.], [1.], [1.]]

    g = tflearn.input_data(shape=[None, 2])
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(
        g, optimizer='sgd', learning_rate=2., loss='mean_square')

    train_model(g, X, Y)


with tf.Graph().as_default():
    # visualization: tensorboard --logdir='/tmp/tflearn_logs'
    # run_NOT()
    run_OR()
