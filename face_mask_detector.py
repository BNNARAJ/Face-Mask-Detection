import tensorflow as tf
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',  # Download from OpenCV repo
    'res10_300x300_ssd_iter_140000.caffemodel'
)

def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype('int')
            faces.append((x1, y1, x2-x1, y2-y1))
    return faces

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained mask detection model
cnn = tf.keras.models.load_model('my_model.keras')  

# Initialize variables for smoothing and accuracy tracking
smoothed_faces = []
smoothing_factor = 0.9  # Adjust as needed
accuracy_window = []  # To store recent accuracy values
accuracy_window_size = 10  # Number of frames to consider for accuracy calculation

# Function to predict whether the person is wearing a mask or not
def predict_mask(face_roi):
    # Preprocess the face region
    face_roi = cv2.resize(face_roi, (64, 64))
    face_roi = img_to_array(face_roi)
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = face_roi / 255.0  # Normalize pixel values

    # Predict probabilities for with mask and without mask
    result = cnn.predict(face_roi)

    # Threshold for classification
    threshold = 0.50

    # Determine the prediction based on the threshold
    if result[0][0] > threshold:
        return 'with mask', result[0][0] * 100
    else:
        return 'without mask', result[0][0] * 100 

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    faces = detect_faces_dnn(frame)
    # Process each detected face
    for (x, y, w, h) in faces:
        # Smooth the coordinates
        if len(smoothed_faces) > 0:
            x_smoothed = int(smoothing_factor * smoothed_faces[-1][0] + (1 - smoothing_factor) * x)
            y_smoothed = int(smoothing_factor * smoothed_faces[-1][1] + (1 - smoothing_factor) * y)
            w_smoothed = int(smoothing_factor * smoothed_faces[-1][2] + (1 - smoothing_factor) * w)
            h_smoothed = int(smoothing_factor * smoothed_faces[-1][3] + (1 - smoothing_factor) * h)
        else:
            x_smoothed, y_smoothed, w_smoothed, h_smoothed = x, y, w, h

        # Crop the face region
        #face_roi = frame[y_smoothed:y_smoothed+h_smoothed, x_smoothed:x_smoothed+w_smoothed]
        # Add margin to bounding box
        margin = 32  # pixels to expand; increase if needed

        x1 = max(0, x_smoothed - margin)
        y1 = max(0, y_smoothed - margin)
        x2 = min(frame.shape[1], x_smoothed + w_smoothed + margin)
        y2 = min(frame.shape[0], y_smoothed + h_smoothed + margin)

        face_roi = frame[y1:y2, x1:x2]
                                                                                                                                                                                                                                                                                                                                                                                                            
        # Predict whether there is a mask on the face and get the confidence score
        prediction, confidence = predict_mask(face_roi)

        # Set the color based on the prediction
        color = (0, 255, 0)  # Default to green for "with mask"
        if prediction == 'without mask':
            color = (0, 0, 255)  # Change color to red for "without mask"

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x_smoothed, y_smoothed), (x_smoothed + w_smoothed, y_smoothed + h_smoothed), color, 2)

        # Display the prediction and confidence on the frame
        text = f'{prediction} ({confidence:.2f}%)'
        cv2.putText(frame, text, (x_smoothed, y_smoothed + h_smoothed + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Only update smoothed_faces if at least one face was detected
        if len(faces) > 0:
            smoothed_faces.append((x_smoothed, y_smoothed, w_smoothed, h_smoothed))
            if len(smoothed_faces) > 10:  # Adjust the window size as needed
                smoothed_faces.pop(0)


    # Display the resulting frame
    cv2.imshow('Face Detection and Mask Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
