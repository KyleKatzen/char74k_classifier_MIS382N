import preprocessing 
import numpy as np
import cv2

def open_image2(filename, scale_to=[64, 64]):
    """Opens an image, returns the preprocessed image (scaled, masked)"""
    #img = cv2.imread(filename) * cv2.imread(filename.replace('Bmp', 'Msk'))/255
    img = cv2.imread(filename)/255
    #processed_img = np.zeros(list(scale_to)+[3])

    # scaling
    #  img_w, img_h = img.shape[1], img.shape[0]
    #  target_w, target_h = scale_to[1], scale_to[0]
    #  factor = target_w / img_w if img_w/img_h > target_w/target_h else target_h / img_h
    #  img = cv2.resize(img, None, fx=factor, fy=factor)
    img = cv2.resize(img, tuple(scale_to))

    # centering image
    #  x, y = int(target_w/2 - img.shape[1]/2), int(target_h/2 - img.shape[0]/2)
    #  processed_img[y:y+img.shape[0], x:x+img.shape[1]] = img

    # normalising
    processed_img = img.astype(np.float32)
    for c in range(3):
        processed_img[:,:,c] /= np.max(processed_img[:,:,c])

    # to grayscale
    processed_img = cv2.cvtColor(
            (processed_img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    processed_img = np.expand_dims(processed_img, -1)

    # new_image = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    # 	cv2.THRESH_BINARY, 11, 2)

    blur = cv2.GaussianBlur(processed_img, (5,5), 0)
    _, new_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    new_image = np.expand_dims(new_image, -1)
    #new_image = cv2.Laplacian(new_image, cv2.CV_64F)

    return new_image


cv2.imshow("Test Image", open_image2("English/Img/GoodImg/Bmp/Sample021/img021-00061.png"))
cv2.waitKey(0)
cv2.destroyAllWindows()

