import cv2
import zmq
import base64
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib


context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:5555')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

def smile(mouth):
            MAR = dist.euclidean(mouth[0], mouth[6])
            return MAR

shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
    
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while True:
    
    frame1 = footage_socket.recv_string()
    img = base64.b64decode(frame1)
    npimg = np.fromstring(img, dtype=np.uint8)
    frame = cv2.imdecode(npimg, 1)
    
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
        
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth= shape[mStart:mEnd]
        MAR= smile(mouth)
       
       

        if MAR >= 52 and MAR < 53:
            cv2.putText(frame, '60%', (rect.right(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif MAR >= 53 and MAR < 54:
            cv2.putText(frame, '80%', (rect.right(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif MAR >= 55:
            cv2.putText(frame, '100%', (rect.right(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif MAR <49:
            cv2.putText(frame, 'Not-Smiling', (rect.right(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif MAR >50 and MAR < 51:
            cv2.putText(frame, '40%', (rect.right(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

    cv2.imshow("Smile-Detector", frame)

    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break
