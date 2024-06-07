import os
import pickle
import cv2
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage
from datetime import datetime
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://attendance-446db-default-rtdb.firebaseio.com/",
    'storageBucket':"attendance-446db.appspot.com"
})
bucket = storage.bucket()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
imgbackground=cv2.imread('resources/background.png')
foldermodepath='resources/Modes'
modepathlist=os.listdir(foldermodepath)
imgmodelist=[]
for mode in modepathlist:
    imgmodelist.append(cv2.imread(os.path.join(foldermodepath,mode)))
file=open('Encodesfile.p','rb')
encodelistwithids=pickle.load(file)
file.close()
enclodelistknown, studentids = encodelistwithids
print(studentids)
counter=0
modetype=0
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read the frame from camera")
        break
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facecurrframes = face_recognition.face_locations(imgs)
    encodecurrframe = face_recognition.face_encodings(imgs, facecurrframes)
    imgbackground[162:162 + 480, 55:55 + 640] = img
    imgbackground[44:44 + 633, 808:808 + 414]=imgmodelist[modetype]
    if facecurrframes:
        for faceencode, faceloc in zip(encodecurrframe, facecurrframes):
            matches = face_recognition.compare_faces(enclodelistknown, faceencode)
            distance = face_recognition.face_distance(enclodelistknown, faceencode)
            matchindx = np.argmin(distance)
            if matches[matchindx]:
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgbackground = cvzone.cornerRect(imgbackground, bbox, rt=0)
                id = studentids[matchindx]
                print(id)
                if counter == 0:
                    counter = 1
            if counter != 0:
                if counter == 1:
                    studentinfo = db.reference(f'Students/{id}').get()
                    modetype = 1
                    blob = bucket.get_blob(f'Images/{id}.png')
                    # array = np.frombuffer(blob.download_as_string(), np.uint8)
                    # imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                    # Update data of attendance
                    datetimeObject = datetime.strptime(studentinfo['last_attendance_time'],
                                                       "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()

                    if secondsElapsed > 30:
                        ref = db.reference(f'Students/{id}')
                        studentinfo['total_attendance'] += 1
                        ref.child('total_attendance').set(studentinfo['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modetype = 3
                        counter = 0
                        imgbackground[44:44 + 633, 808:808 + 414] = imgmodelist[modetype]

                if modetype != 3:

                    if 10 < counter < 20:
                        modetype = 2

                    imgbackground[44:44 + 633, 808:808 + 414] = imgmodelist[modetype]
                    if counter <= 10:
                        cv2.putText(imgbackground, str(studentinfo['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgbackground, str(studentinfo['major']), (1006, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgbackground, str(id), (1006, 493),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                        (w, h), _ = cv2.getTextSize(studentinfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgbackground, str(studentinfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        # imgbackground[175:175 + 216, 909:909 + 216] = imgStudent

                    counter += 1
                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        studentInfo = []
                        imgStudent = []
                        imgbackground[44:44 + 633, 808:808 + 414] = imgmodelist[modeType]
    else:
        modetype = 0
        counter = 0
    cv2.imshow("webcam", imgbackground)
    cv2.waitKey(1)
