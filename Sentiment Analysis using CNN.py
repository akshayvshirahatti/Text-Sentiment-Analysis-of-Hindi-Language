import theano
import numpy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences


seed = 7
numpy.random.seed(seed)


theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'off'


positive_examples=list(open("C:\\Users\\AMAN\\Desktop\\hindi_pos.txt",mode='r',encoding="utf-8"))
positive_examples = [s.strip() for s in positive_examples]
negative_examples=list(open("C:\\Users\\AMAN\\Desktop\\hindi_neg.txt",mode='r',encoding="utf-8"))
negative_examples = [s.strip() for s in negative_examples]
x=positive_examples+negative_examples

xa=np.asarray(x)
print(xa.shape)
positive_labels = [[1] for _ in positive_examples]
negative_labels = [[0] for _ in negative_examples]
y = np.concatenate([positive_labels, negative_labels], 0)
print(y.shape)


from sklearn.model_selection import train_test_split
xa_train,xa_test,y_train,y_test=train_test_split(xa,y,test_size=0.3,random_state=4)
print(xa_train.shape)
print(xa_test.shape)
print(y_train.shape)
print(y_test.shape)

tokenizer = Tokenizer(num_words=None,split=' ',lower=True)
tokenizer.fit_on_texts(xa_train)
integer_sentences_train = tokenizer.texts_to_sequences(xa_train)
data_train = pad_sequences(integer_sentences_train,padding='post',truncating='post',value=0.)
print(data_train[0])
top_words = 5000 #len(tokenizer.word_index)
print(top_words)
max_words = 30

tokenizer.fit_on_texts(xa_test)
integer_sentences_test = tokenizer.texts_to_sequences(xa_test)
data_test = pad_sequences(integer_sentences_test,padding='post',truncating='post',value=0.)
print(data_test[0])

data_train = sequence.pad_sequences(data_train, maxlen=max_words, dtype='float32')
data_test = sequence.pad_sequences(data_test, maxlen=max_words, dtype='float32')


model = Sequential()
model.add(Embedding(input_dim = top_words, output_dim = 20, input_length=max_words))
model.add(Convolution1D(nb_filter=20, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(data_train[:-1],y_train[:-1], batch_size=128,epochs=5,validation_data=(data_test, y_test), verbose=1,sample_weight=None, initial_epoch=0)
yp = model.predict(data_test[:-1], batch_size=32, verbose=1)
ypreds = np.argmax(yp, axis=1)
scores = model.evaluate(data_test[:-1], y_test[:-1], verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

