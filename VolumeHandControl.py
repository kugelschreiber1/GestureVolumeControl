# landmark 4 = thumb tip
# landmark 8 = index finger tip
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

######################################
wCam, hCam = 640, 480
#####################################


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detectionConf=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
#  volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]  # -65
maxVol = volRange[1]  # 0
vol = 0
volBar = 400
volPercentage = 0
area = 0
colorVolume = (255, 0, 0)
while True:
    success, img = cap.read()

    # Find hand to use
    img = detector.findHands(img)
    lmList, boundingbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        # Filter based on size(will use the bounding box)
        area = (boundingbox[2] - boundingbox[0]) * (boundingbox[3] - boundingbox[1]) // 100
        # print(area)
        if 250 < area < 1000:
            # print('yes')
            # Find the distance between the index finger and the thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # print(length)

            # Covert distance to actual volume
            volBar = np.interp(length, [50, 300], [400, 150])
            volPercentage = np.interp(length, [50, 300], [0, 100])

            # Reduce resolution to make the volume change smoother
            smoothness = 10
            volPercentage = smoothness * round(volPercentage / smoothness)

            # Check which of the fingers are up in the visible range
            fingers = detector.fingersUp()
            # print(fingers)

            # if pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPercentage / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVolume = (0, 255, 0)
                time.sleep(0.15)
            else:
                colorVolume = (255, 0, 0)

            # Hand range 50 - 300
            # Windows volume range = -65 - 0

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPercentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    currentVolume = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Volume Set: {int(currentVolume)}', (350, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, colorVolume, 3)

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
