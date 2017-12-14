import numpy as np
import cv2

def open_image2(filename, scale_to=[256, 256]):
    """Opens an image, returns the preprocessed image (scaled, masked)"""
    #img = cv2.imread(filename) * cv2.imread(filename.replace('Bmp', 'Msk'))/255
    img = cv2.imread(filename)/255
    
    #img = cv2.resize(img, tuple(scale_to))
    img = cv2.resize(img, tuple(scale_to))
  
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
    # _, new_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # new_image = np.expand_dims(new_image, -1)
    new_image = cv2.Laplacian(blur, cv2.CV_64F)

    return new_image


cv2.imshow("Test Image", open_image2("ghosh.jpeg"))
cv2.waitKey(0)
cv2.destroyAllWindows()

