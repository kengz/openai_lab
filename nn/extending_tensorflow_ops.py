import tensorflow as tf
import tflearn

import tflearn.datasets.mnist as mnist
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)


def run():
    # model variables
    X = tf.placeholder('float', [None, 784])
    Y = tf.placeholder('float', [None, 10])

    W1 = tf.Variable(tf.random_normal([784, 256]))
    W2 = tf.Variable(tf.random_normal([256, 256]))
    W3 = tf.Variable(tf.random_normal([256, 10]))
    b1 = tf.Variable(tf.random_normal([256]))
    b2 = tf.Variable(tf.random_normal([256]))
    b3 = tf.Variable(tf.random_normal([10]))

    def dnn(x):
        # using tflearn PReLU activation ops
        x = tflearn.prelu(tf.add(tf.matmul(x, W1), b1))
        tflearn.summaries.monitor_activation(x)  # Monitor activation
        x = tflearn.prelu(tf.add(tf.matmul(x, W2), b2))
        tflearn.summaries.monitor_activation(x)  # Monitor activation
        x = tf.nn.softmax(tf.add(tf.matmul(x, W3), b3))
        return x

    net = dnn(X)

    # use objective ops from TFLearn
    loss = tflearn.categorical_crossentropy(net, Y)
    # use metric ops from TFLearn
    acc = tflearn.metrics.accuracy_op(net, Y)
    # use SGF Optimizer class from TFLearn
    optimizer = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=200)
    # Because of lr decay, it is required to first build the Optimizer with
    # the step tensor that will monitor training step.
    # (Note: When using TFLearn estimators wrapper, build is self managed,
    # so only using above `Optimizer` class as `DNN` optimizer arg is enough).
    step = tflearn.variable('step', initializer='zeros', shape=[])
    optimizer.build(step_tensor=step)
    optim_tensor = optimizer.get_tensor()

    # Use TFLearn Trainer
    # def training op for backprop
    trainop = tflearn.TrainOp(loss=loss, optimizer=optim_tensor,
                              metric=acc, batch_size=128,
                              step_tensor=step)

    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=3)
    trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={
                X: testX, Y: testY}, n_epoch=2, show_metric=True)


with tf.Graph().as_default():
    run()
