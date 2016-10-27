import tflearn
from tflearn.data_utils import pad_sequences, to_categorical
from tflearn.datasets import imdb

train, test, _ = imdb.load_data(
    path='data/imdb.pkl', n_words=20000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# binarizer, increase rank
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)


def run():
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=20000, output_dim=128)
    net = tflearn.bidirectional_rnn(
        net, tflearn.BasicLSTMCell(128), tflearn.BasicLSTMCell(128))
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(
        net, optimizer='adam', loss='categorical_crossentropy')

    m = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
    m.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=64)
    m.save('models/bidirectional_rnn.tfl')

run()
