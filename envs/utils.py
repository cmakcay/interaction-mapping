import sys
import torchvision.transforms as transforms
import cv2
import numpy as np


def show_wait(img, T=0, win='image', sz=None, save=None):

    shape = img.shape
    img = transforms.ToPILImage()(img)
    if sz is not None:
        H_new = int(sz/shape[2]*shape[1])
        img = img.resize((sz, H_new))

    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    if save is not None:
        cv2.imwrite(save, open_cv_image)
        return

    cv2.imshow(win, open_cv_image)
    inp = cv2.waitKey(T)
    if inp==27:
        cv2.destroyAllWindows()
        sys.exit(0)