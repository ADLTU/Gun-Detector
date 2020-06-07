from shutil import copyfile
from Image_functions import save_image, display_image, resize_image, get_image
from PIL import Image
import os

test_path = '/Users/hadi/Downloads/comp5_det_test_Pistol.txt'

file = open(test_path, 'r')

test_results_folder = '/Applications/YOLOv3/Test_results/'

list_of_images = []

width = 416
height = 416

for l in file:
    l = l.split(' ')
    if float(l[1]) > 0.4:
        # point_1 = [float(l[2]), float(l[3])]
        # point_2 = [float(l[4]), float(l[5])]
        # x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
        # y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
        # x_width = float(abs(point_2[0] - point_1[0])) / width
        # y_height = float(abs(point_2[1] - point_1[1])) / height
        #
        # result = '0 ' + str(x_center) + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)
        # write_p = (test_results_folder+l[0]+'.txt')
        # frame_result = open(write_p, 'a')
        # frame_result.write(result)
        if l[0] not in list_of_images:
            list_of_images.append(l[0])


# for im in list_of_images:
#     src = '/Users/hadi/Downloads/RERE/' + im + '.jpg'
#     dst = '/Applications/YOLOv3/Detected/' + im + '.jpg'
#     image = Image.open(src)
#     copyfile(src, dst)

#
# images_p = '/Users/hadi/Downloads/WeaponS/'
#
# images = os.listdir(images_p)
#
# p = 0
# for i, j in enumerate(images):
#     if not (j == '.DS_Store'):
#         try:
#             image = get_image(images_p+j)
#             new_ = resize_image(image, (416, 416))
#             saveto = ('/Users/hadi/Downloads/RERE/testimage'+str(i+1)+'.jpg')
#             save_image(new_, path=saveto)
#         except:
#             print(j)

import numpy as np
a = np.ones((32, 13))
print(a.shape)