import numpy as np
import cv2

image_path = 'images/bus.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt.txt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

np.random.seed(543210) # Set seed for colors to be consistent across program runs
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# image = cv2.imread(image_path) # Uncomment this line if you want to use image file instead of webcam
cap = cv2.VideoCapture(0)

while True: # Remove this line if you want to use image file instead of webcam

    _, image = cap.read() # Comment this line if you want to use image file instead of webcam

    height, width = image.shape[0], image.shape[1] # Unindent this line until 'cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_id], 2)' ONCE for using image file instead of webcam

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:

            class_id = int(detections[0, 0, i, 1])

            upper_left_x = int(detections[0, 0, i, 3] * width)
            upper_left_y = int(detections[0, 0, i, 4] * height)
            lower_right_x = int(detections[0, 0, i, 5] * width)
            lower_right_y = int(detections[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_id]}: {confidence:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_id], 3)
            cv2.putText(image, prediction_text, (upper_left_x, upper_left_y-15 if upper_left_y > 30 else upper_left_y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_id], 2)

    # Unindent the two lines below for using image file instead of webcam
    # # Set resizable property for window showing image
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # Show image
    cv2.imshow("Detected Object(s)", image)
    cv2.waitKey(5)

cv2.destroyAllWindows()