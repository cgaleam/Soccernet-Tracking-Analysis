import cv2
import os

folder = "../SoccerNet/tracking/test/SNMOT-118/img1"
images = sorted(os.listdir(folder))

for img_name in images:
    path = os.path.join(folder, img_name)

    frame = cv2.imread(path)
    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()


