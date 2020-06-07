import cv2 as cv
import numpy as np
#np.set_printoptions(precision=3)

def get_image(img_source):
    image = cv.imread(img_source)
    return image


def save_image(image, path='/Users/hadi/Downloads/RERE/'):
    cv.imwrite(path, image)

def display_image(image):
    cv.imshow('imagesss', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize_image(img, desired_size):
    
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = desired_size
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    canvas = np.full((desired_size[1], desired_size[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    canvas = canvas[:, :, ::-1].transpose((2, 0, 1)).copy()
    canvas = canvas / 255.0
    canvas = np.expand_dims(canvas, axis=0)
    return canvas