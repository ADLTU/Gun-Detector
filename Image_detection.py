from Read_cfg import read_cfg
from Read_classes import load_classes
from Read_weights import load_weights
import argparse
from Image_functions import save_image, display_image, resize_image, get_image
from Net import *
from Interpret_results import write_results
from timeit import default_timer as timer

def get_input():
    arg = argparse.ArgumentParser()

    arg.add_argument("--cfg", dest='cfg_file', default="/Applications/YOLOv3/yolov3.cfg", type=str)
    arg.add_argument("--weights", dest='weights_file', default="/Users/hadi/Downloads/yolov3_4000.weights", type=str)
    arg.add_argument("--image", dest='image', default="/Users/hadi/Downloads/Picture1-ConvertImage.jpg", type=str)
    arg.add_argument("--confidence", dest="confidence", default=0.5)
    arg.add_argument("--nms_thresh", dest="nms_thresh", default=0.4)

    return arg.parse_args()


inputs = get_input()
net_info, modules = read_cfg(inputs.cfg_file)
weights = load_weights(inputs.weights_file)
confidence = float(inputs.confidence)
nms_thesh = float(inputs.nms_thresh)

array_image = []
classes = load_classes("/ImageLabeling/Classes.txt")

image_dims = int(net_info["height"])

assert image_dims % 32 == 0
assert image_dims > 32

try:
    image = get_image(inputs.image)
    ready_image = resize_image(image, (image_dims, image_dims))
except FileNotFoundError:
    print("Image not found")
    exit()

print("Starting Detection")
od_start = timer()

final_image = forward(modules, net_info, weights, ready_image)
print(final_image)
# prediction = write_results(final_image, confidence, len(classes), nms_conf=nms_thesh)
# print(prediction)

# print("Detection done in: ", timer()-od_start)
