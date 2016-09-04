# good read: http://cs231n.github.io/convolutional-networks/
import tensorflow as tf
import tflearn

import tflearn.datasets.mnist as mnist
mnist_data = mnist.read_data_sets(one_hot=True)
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)


# convnet rule of thumb:
# input layer size shd be powers of 2
# conv layer shd have small filter:
# K number of filter(kernel),
# F size (height == width),
# S stride
# zero padding P
# F=3, P=1; or F=5, P=2 (no bigger)
#
# pool layer: F=2, S=2, max pool
#
# layer patterns:
# INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
def run():
    X = tf.placeholder(shape=(None, 784), dtype=tf.float32)
    Y = tf.placeholder(shape=(None, 10), dtype=tf.float32)

    net = tf.reshape(X, [-1, 28, 28, 1])  # batch, height, width, chnl

    # 32 filters, each of size 3(x3)
    net = tflearn.conv_2d(net, 32, 3, activation='relu')
    # pool kernel size 2, stride size default kernel soze
    net = tflearn.max_pool_2d(net, 2)
    # for "encourage some kind of inhibition and boost the neurons with
    # relatively larger activations"
    net = tflearn.local_response_normalization(net)
    # The dropout method is introduced to prevent overfitting. At each training stage, individual nodes are either "dropped out" of the net with probability {\displaystyle 1-p} 1-p or kept with probability {\displaystyle p} p, so that a reduced network is left
    # keep_prob=0.8
    net = tflearn.dropout(net, 0.8)

    # 64 filters
    net = tflearn.conv_2d(net, 64, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.dropout(net, 0.8)

    # FC
    net = tflearn.fully_connected(net, 128, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 256, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 10, activation='softmax')

    # --------------------------------------
    # really manual tf way
    # # Defining other ops using Tensorflow
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    # optimizer_minop = optimizer.minimize(loss)

    # # start
    # init = tf.initialize_all_variables()

    # with tf.Session() as sess:
    #     sess.run(init)
    #     batch_size = 128
    #     for epoch in range(2):
    #         avg_cost = 0.
    #         total_batch = int(mnist_data.train.num_examples/batch_size)
    #         for i in range(total_batch):
    #             batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
    #             sess.run(optimizer_minop, feed_dict={X: batch_xs, Y: batch_ys})
    #             cost = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys})
    #             avg_cost += cost/total_batch
    #             if i % 20 == 0:
    #                 print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,
    #                       "Loss:", str(cost))

    # --------------------------------------
    # use trainer class
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)),
        tf.float32), name='acc')

    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                              metric=accuracy, batch_size=128)

    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0)
    trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={
                X: testX, Y: testY}, n_epoch=2, show_metric=True)
    trainer.save('models/mnist_cnn.tfl')


with tf.Graph().as_default():
    run()
