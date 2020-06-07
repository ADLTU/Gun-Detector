import os

path = '/Users/hadi/Downloads/RERE'

file = open('/Applications/YOLOv3/ImageLabeling/test_paths.txt', 'w')

find_path = '/content/Test/'

images = os.listdir(path)

p = 0
for i in images:
    if not (i == '.DS_Store'):
        file.write(find_path+i+"\n")