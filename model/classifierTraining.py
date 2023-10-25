import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# read datasets
# 0: other gestures
# 1: first gesture
# 2: second gesture
dataset0 = np.genfromtxt(fname='C:\\Users\\Nikos\\PycharmProjects\\audioOutputSwitch\\data\\0.csv', delimiter=',')
dataset1 = np.genfromtxt(fname='C:\\Users\\Nikos\\PycharmProjects\\audioOutputSwitch\\data\\1.csv', delimiter=',')
dataset2 = np.genfromtxt(fname='C:\\Users\\Nikos\\PycharmProjects\\audioOutputSwitch\\data\\2.csv', delimiter=',')

# merge datasets
dataset = np.concatenate([dataset0, dataset1, dataset2], axis=0)

# shuffle
np.random.shuffle(dataset)

xDataset = dataset[:, :-1]
yDataset = dataset[:, -1]

# split to training validation, testing datasets
xTrain, xRem, yTrain, yRem = train_test_split(xDataset, yDataset, train_size=0.70, random_state=0)
xVal, xTest, yVal, yTest = train_test_split(xRem, yRem, train_size=0.5, random_state=0)

# model creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# model.summary()

# create callbacks for saving checkpoints and early stopping based on validation metric
checkPointCallback = tf.keras.callbacks.ModelCheckpoint('C:\\Users\\Nikos\\PycharmProjects\\audioOutputSwitch\\model\\modelCheckpoints\\classifierCheckpoint.hdf5', verbose=1, save_weights_only=False)
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train model
model.fit(
    xTrain,
    yTrain,
    epochs=1000,
    batch_size=128,
    validation_data=(xVal, yVal),
    callbacks=[checkPointCallback, earlyStoppingCallback]
)

# test the model on the testing dataset and print evaluation metrics
model = tf.keras.models.load_model('C:\\Users\\Nikos\\PycharmProjects\\audioOutputSwitch\\model\\modelCheckpoints\\classifierCheckpoint.hdf5')

def print_confusion_matrix(yTrue, yPred, report=True):
    labels = sorted(list(set(yTrue)))
    cmx_data = confusion_matrix(yTrue, yPred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(yTrue)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(yTrue, yPred))


yPred = model.predict(xTest)
yPred = np.argmax(yPred, axis=1)

print_confusion_matrix(yTest, yPred)