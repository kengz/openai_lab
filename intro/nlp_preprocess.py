# illustrate NLP preprocessing
# convert str into vector number of fixed length for input
# then can further be embedded into a real vector
import pandas
import tflearn
import tensorflow
from tensorflow.contrib import learn

MAX_TOKEN_LEN = 200
MIN_FREQ = 0  # cum freq to be added to vocab
VOCAB_PATH = 'models/dbpedia.tfl.vocab'

dbpedia = learn.datasets.load_dataset('dbpedia')
# X = ['Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.', ...]
# Y = [int labels]

# limiting size first
str_trainX = pandas.DataFrame(dbpedia.train.data)[1][:5600]
trainY = pandas.Series(dbpedia.train.target)[:5600]
str_testX = pandas.DataFrame(dbpedia.test.data)[1][:700]
testY = pandas.Series(dbpedia.test.target)[:700]


# --------------------------------------
# TFLearn vocab processor
encoder = tflearn.data_utils.VocabularyProcessor(
    MAX_TOKEN_LEN, min_frequency=MIN_FREQ)

trainX = encoder.fit_transform(str_trainX)
testX = encoder.fit_transform(str_testX)
encoder.save(VOCAB_PATH)

# encoder.restore(VOCAB_PATH)
# trainX = encoder.transform(str_trainX)
# testX = encoder.transform(str_testX)

org = encoder.reverse(list(trainX)[:1])
print(list(org))


# --------------------------------------
# sklearn
import sklearn
from sklearn import feature_extraction

encoder = feature_extraction.text.TfidfVectorizer()
trainX = encoder.fit_transform(str_trainX).todense()
testX = encoder.fit_transform(str_testX).todense()
# not really that vital to invert X
org = encoder.inverse_transform(testX[:1])
print(list(org))
