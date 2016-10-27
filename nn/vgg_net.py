
# Applying VGG 16-layers convolutional network to Oxford's 17 Category Flower
# Dataset classification task.
import tflearn
import os
import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True)

def run():
    net = tflearn.input_data(shape=[None, 224, 224, 3])

    net = tflearn.conv_2d(net, 64, 3, activation='relu')
    net = tflearn.conv_2d(net, 64, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)

    net = tflearn.conv_2d(net, 128, 3, activation='relu')
    net = tflearn.conv_2d(net, 128, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)

    net = tflearn.conv_2d(net, 256, 3, activation='relu')
    net = tflearn.conv_2d(net, 256, 3, activation='relu')
    net = tflearn.conv_2d(net, 256, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)

    net = tflearn.conv_2d(net, 512, 3, activation='relu')
    net = tflearn.conv_2d(net, 512, 3, activation='relu')
    net = tflearn.conv_2d(net, 512, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)

    net = tflearn.conv_2d(net, 512, 3, activation='relu')
    net = tflearn.conv_2d(net, 512, 3, activation='relu')
    net = tflearn.conv_2d(net, 512, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)

    net = tflearn.fully_connected(net, 4096, activation='relu')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 4096, activation='relu')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 17, activation='softmax')

    net = tflearn.regression(net, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

    m = tflearn.DNN(net, checkpoint_path='models/vgg_net',
                    max_checkpoints=1, tensorboard_verbose=3)
    m.fit(X, Y, n_epoch=500, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_net')
    m.save('models/vgg_net.tfl')


run()
