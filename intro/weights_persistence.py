import os
import tflearn
import tensorflow as tf
import tflearn.datasets.mnist as mnist


def run_mnist():
    X, Y, testX, testY = mnist.load_data(one_hot=True)
    g = tflearn.input_data(shape=[None, 784], name='input')
    g = tflearn.fully_connected(g, 128, name='dense1')
    g = tflearn.fully_connected(g, 256, name='dense2')
    g = tflearn.fully_connected(g, 10, activation='softmax', name='softmax')
    g = tflearn.regression(
        g, optimizer='adam',
        learning_rate=0.001,
        loss='categorical_crossentropy')

    if not os.path.isdir('models'):
        os.mkdir('models')
    m = tflearn.DNN(g, checkpoint_path='models/model.tfl.ckpt')
    m.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          # Snapshot (save & evaluate) model every epoch.
          snapshot_epoch=True,
          # Snapshot (save & evalaute) model every 500 steps.
          snapshot_step=500,
          run_id='model_and_weights')
    m.save('models/mnist.tfl')

    # load from file or ckpt and continue training
    m.load('models/mnist.tfl')
    # m.load('models/mnist.tfl.ckpt-500')
    m.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          # Snapshot (save & evaluate) model every epoch.
          snapshot_epoch=True,
          # Snapshot (save & evalaute) model every 500 steps.
          snapshot_step=500,
          run_id='model_and_weights')

    # retrieve layer by name, print weights
    dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
    print('Dense1 layer weights:')
    print(m.get_weights(dense1_vars[0]))
    # or using generic tflearn function
    print('Dense1 layer biases:')
    with m.session.as_default():
        print(tflearn.variables.get_value(dense1_vars[1]))

    # or can even retrieve using attr `W` or `b`!
    print('Dense2 layer weights:')
    dense2 = tflearn.get_layer_by_name('dense2')
    print(dense2)
    print(m.get_weights(dense2.W))
    print('Dense2 layer biases:')
    with m.session.as_default():
        print(tflearn.variables.get_value(dense2.b))


run_mnist()
