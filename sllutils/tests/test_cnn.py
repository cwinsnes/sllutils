import sllutils
from keras.datasets import mnist
import numpy as np
import shutil

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    binarizer = sllutils.ml.Binarizer(range(10))
    y_train = binarizer(y_train)
    y_test = binarizer(y_test)

    dnn = sllutils.ml.DNN()
    dnn.set_input_shape((1, 28, 28))
    dnn.conv2D(32, 3, activation='relu')
    dnn.conv2D(64, 3, activation='relu')
    dnn.maxpool2D()
    dnn.dropout(0.25)
    dnn.flatten()
    dnn.dense(128, 'relu')
    dnn.dropout(0.5)
    dnn.dense(10, 'softmax')
    dnn.build('categorical_crossentropy', 'adam')

    # Should yield about 99% accuracy
    dnn.train(np.asarray(x_train), np.asarray(y_train), 128, epochs=12, verbose=True)
    preds = dnn.predict(np.asarray(x_test))
    preds = np.argmax(preds, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(sum(preds == y_test)/len(preds))

    dnn.save('/tmp/testsave.model')
    dnn.load_model('/tmp/testsave.model')

    del(dnn)
    dnn = sllutils.ml.DNN.load('/tmp/testsave.model')
    preds = dnn.predict(np.asarray(x_test))
    preds = np.argmax(preds, axis=1)
    print(sum(preds == y_test)/len(preds))
    shutil.rmtree('/tmp/testsave.model', True)
