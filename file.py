import cv2
import math
import argparse


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 0.5, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (255,248,220), int(round(frameHeight/150)), 2)
    return frameOpencvDnn,faceboxes


parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)','(15-25)', '(19-36)','(38-43)','(38-43)','(60-100)']
genderList=['Male','Female']

faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg, faceboxes = highlightFace(faceNet,frame)
    if not faceboxes:
        print("No face detected")

    for facebox in faceboxes:
        face = frame[max(0, facebox[1]-padding):
                   min(facebox[3]+padding, frame.shape[0]-1), max(0, facebox[0]-padding):
                   min(facebox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (facebox[0], facebox[1]-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,248,220), 1, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
        
        
        
        #COMMAND TO DETECT AGE THROUGH IMAGE : python file.py --image <image name.type>
        #COMMAND TO DETECT LIVE WEBCAM : python file.py
