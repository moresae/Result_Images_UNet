import os
import cv2

path = "./results_unet1"
path_save = "./roi"

x_scale = 0.4
y_scale = 1.5

for path_img in os.listdir(path):

    img = cv2.imread(f"{path}/{path_img}")
    img = cv2.resize(img,(int(y_scale*img.shape[0]), int(x_scale*img.shape[1])),interpolation = cv2.INTER_AREA)

    # cv2.rectangle(img,(598,119),(941,691),(255,0,0),2)
    # cv2.rectangle(img,(1009,119),(1352,691),(255,0,0),2)
    roi_ground = img[119:689, 598:940]
    roi_predict = img[119:689, 1009:1351]

    # print(path_img.split("."))

    path_img = path_img.split(".")

    cv2.imwrite(f"{path_save}/{path_img[0]}_ground.png",roi_ground)
    cv2.imwrite(f"{path_save}/{path_img[0]}_predict.png",roi_predict)

    # cv2.imshow("Ground_Truth",roi_ground)
    # cv2.imshow("Ground_Truth",roi_predict)
    # box = cv2.selectROI(img)

    if cv2.waitKey(0) & 0xFF == ord('ñ'):
        break

    