import tensorflow as tf

from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

train_images_path = "images/train"
test_images_path = "images/validation"

test_data_prep = ImageDataGenerator(rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    )
train_data_prep = ImageDataGenerator(rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=True,
    )



data = test_data_prep.flow_from_directory(train_images_path,
target_size=(48, 48),
batch_size=32,
class_mode='categorical',
shuffle=True,
seed=42
)
val_data = train_data_prep.flow_from_directory(test_images_path,
target_size=(48, 48),
batch_size=32,
class_mode='categorical',
shuffle=True,
seed=42
)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation="softmax"))

model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


model.fit(data, epochs=10, validation_data=val_data)

model.summary()

test_loss, test_acc = model.evaluate(val_data, verbose=2)
print(test_acc)

model.save("face_recogn.h5")
sess = tf.compat.v1.keras.backend.get_session()
graph_def = sess.graph.as_graph_def()

frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def, ["dense_1"])
tf.io.write_graph(frozen_graph, "face_recogn", "model.pb", as_text=False)