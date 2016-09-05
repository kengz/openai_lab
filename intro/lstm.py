import tflearn
from tflearn.data_utils import pad_sequences, to_categorical
from tflearn.datasets import imdb

train, test, _ = imdb.load_data(
  path='data/imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# binarizer, increase rank
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)


# words -> vocab id transform -> embedding
def run():
  net = tflearn.input_data([None, 100])
    # embed int vector to compact real vector
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    # fucking magic of rnn
    # if dynamic lstm, backprop thru time till the seq ends,
    # but padding is needed to feed input dim; tail not used
    net = tflearn.lstm(net, 128, dropout=0.8, dynamic=True)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
     learning_rate=0.001,
     loss='categorical_crossentropy')

    m = tflearn.DNN(net)
    m.fit(trainX, trainY, validation_set=(testX, testY),
      show_metric=True, batch_size=32)
    m.save('models/lstm.tfl')

    run()
