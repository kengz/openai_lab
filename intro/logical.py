import os
import tensorflow as tf
import tflearn


# train a model, given a graph g, data X, Y
def train_model(g, X, Y, model_name='models/sample_model.tfl'):
    if not os.path.isdir('models'):
        os.mkdir('models')
    m = tflearn.DNN(g)
    m.fit(X, Y, n_epoch=1000, snapshot_epoch=False)
    m.save(model_name)
    # m.load(model_name)

    print('Testing operator')
    print('Prediction', m.predict(X))
    print('Target Result', Y)
    return m


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


# Logical AND operator
def run_AND():
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y = [[0.], [0.], [0.], [1.]]

    g = tflearn.input_data(shape=[None, 2])
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(
        g, optimizer='sgd', learning_rate=2., loss='mean_square')

    train_model(g, X, Y)


# Logical XOR operator
def run_XOR():
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y = [[0.], [1.], [1.], [0.]]

    g = tflearn.input_data(shape=[None, 2])
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(
        g, optimizer='sgd', learning_rate=2., loss='mean_square')

    train_model(g, X, Y)


# using graph combo with multiple optimizers
# a bit of a cheat, but ok
# Logical XOR operator
def run_combo_XOR():
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y_nand = [[1.], [1.], [1.], [0.]]
    Y_or = [[0.], [1.], [1.], [1.]]

    g = tflearn.input_data(shape=[None, 2])

    # Nand graph
    g_nand = tflearn.fully_connected(g, 32, activation='linear')
    g_nand = tflearn.fully_connected(g_nand, 32, activation='linear')
    g_nand = tflearn.fully_connected(g_nand, 1, activation='sigmoid')
    g_nand = tflearn.regression(
        g_nand, optimizer='sgd', learning_rate=2., loss='binary_crossentropy')

    # Nand graph
    g_or = tflearn.fully_connected(g, 32, activation='linear')
    g_or = tflearn.fully_connected(g_or, 32, activation='linear')
    g_or = tflearn.fully_connected(g_or, 1, activation='sigmoid')
    g_or = tflearn.regression(
        g_or, optimizer='sgd', learning_rate=2., loss='binary_crossentropy')

    g_xor = tflearn.merge([g_nand, g_or], mode='elemwise_mul')

    m = train_model(g_xor, X, [Y_nand, Y_or])
    # sess = tf.Session()  # separate from DNN session
    sess = m.session  # separate from DNN session
    print(
        sess.run(tflearn.merge([Y_nand, Y_or], mode='elemwise_mul')))


with tf.Graph().as_default():
    # visualization: tensorboard --logdir='/tmp/tflearn_logs'
    # run_NOT()
    # run_OR()
    # run_AND()
    # run_XOR()
    run_combo_XOR()
