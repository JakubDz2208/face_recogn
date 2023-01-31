import cv2
import keras
import numpy as np

model = keras.models.load_model("face_recogn.h5")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Only consider faces within the specified square
        if x > 100 and y > 100 and x + w < 500 and y + h < 500:
            # Crop the face and resize it
            face_cropped = frame[y:y + h, x:x + w]
            face_cropped = cv2.flip(face_cropped, 1)

            face_resized = cv2.resize(face_cropped, (48, 48))


            # Predict emotions
            class_names = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6,}
            predictions = model.predict(np.expand_dims(face_resized, axis=0))
            prediction = predictions[0]
            label = np.argmax(prediction)
            class_value = list(class_names.keys())[label]

            # Display the results
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(class_value)
            cv2.putText(frame, text, (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    
    # Display the resulting frame
    cv2.imshow("Emotion detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
