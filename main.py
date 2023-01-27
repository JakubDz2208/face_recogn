import cv2
import tensorflow as tf

# model = cv2.dnn.readNetFromCaffe("face_recogn.h5")


net = cv2.dnn.readNetFromTensorflow("face_recogn_pb/saved_model.pb", "face_recogn_pb/variables/variables.index")

cap = cv2.VideoCapture(0)
cap.set(4,480)
cap.set(3,640)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Przetwarzanie obrazu z kamerki
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    output = net.forward()

    # Wyświetlenie wyniku na ekranie
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()