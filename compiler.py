import tensorflow as tf

# Wczytanie modelu z pliku h5
model = tf.keras.models.load_model("face_recogn.h5")

# Zapisanie modelu jako SavedModel
tf.saved_model.save(model, "face_recogn_pb")
