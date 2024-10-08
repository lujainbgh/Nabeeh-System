import cv2

def find_webcam():
    index = 0
    found = False
    while not found:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"Webcam found at index: {index}")
            found = True
        else:
            index += 1
        cap.release()

find_webcam()