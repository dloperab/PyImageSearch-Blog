# import the necessary packages
from pyimagesearch.centroid_tracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

PROTOTXT = "..\\..\extras\\caffe_models\\face_detector\\deploy.prototxt"
MODEL = "..\\..\\extras\\caffe_models\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE = 0.5

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # read the next frame from the video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the frame, pass it through the network,
    # obtain out predictions, and initilize the of bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted probability
        # is greater than a maximum threshold
        if detections[0, 0, i, 2] > CONFIDENCE:
            # compute (x,y)-coords of the bbox for the object, the update the bbox rects list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # draw a bbox surrounding the object so we can visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # update our centroid tracker using the computed set of bbox rects
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    
    # if the key 'q' was pressed, break the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup resources
cv2.destroyAllWindows()
vs.stop()
