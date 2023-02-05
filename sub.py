import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.h5', compile=False)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to match the input size of the model
    frame = cv2.resize(frame, (224, 224))

    # Convert the frame to a 4D tensor
    frame = np.expand_dims(frame, axis=0)

    # Make a prediction on the frame
    prediction = model.predict(frame)

    # Check if the prediction corresponds to the object of interest
    if prediction[0][1] > 0.5:
        # Do something (e.g. draw a bounding box)
        pass

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Destroy the window
cv2.destroyAllWindows()