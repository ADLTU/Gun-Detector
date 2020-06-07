import os

path = '/Applications/YOLOv3/Gun_Images'

move_to = '/Applications/YOLOv3/ImageLabeling/images'
n = 301

for image in os.scandir(path):
    n += 1
    os.rename(image.path, os.path.join(move_to, '{:06}.jpg'.format(n)))
