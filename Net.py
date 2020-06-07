import numpy as np
from timeit import default_timer as timer
import scipy.ndimage
start = timer()
all_activations = []
from Image_functions import save_image, display_image
np.set_printoptions(precision=4)

'''''
def get_test_masks(mask_source="/Applications/YOLOv3/Masks"):
    masks_file = open(mask_source, 'r')
    masks_file = masks_file.read().split('\n')

    masks = []
    get = False

    for line in masks_file:
        if get:
            line = [int(x) for x in line.split(',')]
            x = int(math.sqrt(len(line)))
            b = []
            for i in range(0, x):
                b.append(line[i * x:(i + 1) * x])
            b = np.array(b)
            masks.append(b)
            get = False
        if line[0] == "[":
            get = True

    return masks
'''''


def yolo(image, start_size, anchors):

    batch_size = image.shape[0]
    stride = start_size // image.shape[2]
    grid_size = image.shape[2]
    bbox_attrs = 5 + 1
    num_anchors = len(anchors)

    image = np.reshape(image, (batch_size, bbox_attrs * num_anchors, grid_size * grid_size))
    image = image.transpose((0, 2, 1))
    image = np.reshape(image, (batch_size, grid_size * grid_size * num_anchors, bbox_attrs))
    anchors = [[a[0]/stride, a[1]/stride] for a in anchors]

    image[:, :, 0] = 1/(1 + np.exp(-image[:, :, 0]))
    image[:, :, 1] = 1/(1 + np.exp(-image[:, :, 1]))
    image[:, :, 4] = 1/(1 + np.exp(-image[:, :, 4]))

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = np.reshape(a, (-1, 1))
    y_offset = np.reshape(b, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), 1)
    x_y_offset = np.tile(x_y_offset, 3)
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, axis=0)

    image[:, :, :2] += x_y_offset

    anchors = np.tile(anchors, (grid_size * grid_size, 1))
    anchors = np.expand_dims(anchors, axis=0)

    image[:, :, 2:4] = np.exp(image[:, :, 2:4]) * anchors

    image[:, :, 5: 5 + 1] = 1/(1 + np.exp(-image[:, :, 5: 5 + 1]))

    image[:, :, :4] *= stride

    return image


def get_masks(conv_weights, num_filters, prev_filter, size):

    filter_size = int(len(conv_weights)) // num_filters
    masks = []
    for i in range(num_filters):
        try:
            use  = conv_weights[i*filter_size:(i+1)*filter_size]
            filters = np.reshape(use, (prev_filter, size, size))
            masks.append(filters)
        except ValueError as e:
            print(e)
            exit()

    return np.array(masks)


def route_layer(image, layers, all_activations):
    m = len(all_activations)
    if len(layers) == 1:
        image = all_activations[m + (layers[0])]
    else:
        layers[1] = layers[1] - m
        map1 = all_activations[m + layers[0]]
        map2 = all_activations[m + layers[1]]
        image = np.concatenate((map1, map2), 1)
    return image


def upsample(image, stride):
    output = np.zeros((image.shape[0], image.shape[1], image.shape[2] * stride, image.shape[3] * stride))

    for img in range(image.shape[0]):
        for f in range(image.shape[1]):
            output[img, f, ...] = scipy.ndimage.zoom(image[img, f, ...], stride, order=0)
    return output


def forward_cnn(images, filters, biases, stride, padding, leaky, bn_weight=None, bn_mean=None, bn_var=None, bn=0):
    num_images, channels, height, width = images.shape
    num_filters, f_channels, f_height, f_width = filters.shape

    dilation = 1

    pad = (f_width - 1) // 2

    images = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    o_dims = int(np.floor(((height + (2 * pad) - dilation * (f_height - 1) - 1) / stride) + 1))

    output = np.zeros((num_images, num_filters, o_dims, o_dims))

    sub_dims = int(np.floor(((height + (2 * pad) - dilation * (f_height - 1) - 1) / 1) + 1))
    sub_shape = (1, sub_dims, sub_dims)
    for img in range(num_images):
        for f in range(num_filters):
            image = images[img]
            weights_ = filters[f, :, :]

            if f_height == 1:
                convolved_matrix = np.sum(image * weights_, axis=0)
            else:
                submatrices_ = np.lib.stride_tricks.as_strided(image, weights_.shape + sub_shape, image.strides * 2)
                convolved_matrix = np.einsum('xyz,xyzklm->klm', weights_, submatrices_)

            if stride == 2:
                stride_shape = (convolved_matrix.strides[0], convolved_matrix.strides[1]*2, convolved_matrix.strides[2]*2)
                convolved_matrix = np.lib.stride_tricks.as_strided(convolved_matrix, (1, o_dims, o_dims), stride_shape)

            if bn:
                bn_convolved = batch_norm(convolved_matrix, bn_weight[f], bn_mean[f], bn_var[f])
                convolved_with_bias = np.squeeze(bn_convolved) + biases[f]

            else:
                convolved_with_bias = np.squeeze(convolved_matrix) + biases[f]

            if leaky:
                convolved_with_bias[convolved_with_bias < 0] = convolved_with_bias[convolved_with_bias < 0] * 0.1
            output[img, f, :, :] = convolved_with_bias
            # display_image(convolved_with_bias)
            # save_image(convolved_with_bias)
    return output


def batch_norm(filtered_image, bn_weight, bn_mean, bn_var):
    filtered_image = filtered_image - bn_mean
    filtered_image = filtered_image / np.sqrt(bn_var + 1e-5)
    filtered_image = filtered_image * bn_weight
    return filtered_image


def forward(modules, net_info, weights, image):
    next_input = image
    position = 0
    prev_num_filters = 3
    write = 0
    print_the = 0
    for m in range(len(modules)):

        module_type = modules[m]["type"]
        if module_type == "convolutional":
            model = modules[m]

            try:
                batch_normalize = int(model["batch_normalize"])
            except KeyError:
                batch_normalize = 0

            num_filters = int(model["filters"])
            conv_size = int(model["size"])
            stride = int(model["stride"])
            pad = int(model["pad"])

            if model["activation"] == "leaky":
                leaky = True
            else:
                leaky = False

            if batch_normalize:

                bn_biase = weights[position:position + num_filters]
                position += num_filters

                bn_weight = weights[position:position + num_filters]
                position += num_filters

                bn_mean = weights[position:position + num_filters]
                position += num_filters

                bn_var = weights[position:position + num_filters]
                position += num_filters

            else:
                conv_biases = weights[position:position + num_filters]
                position += num_filters

            num_weights = (conv_size ** 2) * prev_num_filters * num_filters
            conv_weights = weights[position:position + num_weights]

            position += num_weights

            masks = get_masks(conv_weights, num_filters, prev_num_filters, conv_size)
            if batch_normalize:
                next_input = forward_cnn(next_input, masks, bn_biase, stride, pad, leaky, bn_weight, bn_mean, bn_var, 1)
            else:
                next_input = forward_cnn(next_input, masks, conv_biases, stride, pad, leaky)

        elif module_type == "shortcut":
            model = modules[m]
            get_from = int(model["from"])
            activation = model["activation"]
            next_input = next_input + all_activations[get_from]

        elif module_type == "upsample":
            model = modules[m]
            stride = int(model["stride"])
            next_input = upsample(next_input, stride)


        elif module_type == "route":
            model = modules[m]
            layers = model["layers"].split(",")
            layers = [int(a) for a in layers]
            next_input = route_layer(next_input, layers, all_activations)

        elif module_type == "yolo":
            model = modules[m]
            mask = list(map(int, model["mask"].split(",")))

            anchors = list(map(int, model["anchors"].split(",")))
            anchors = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            inp_dim = int(net_info["height"])

            num_classes = int(model["classes"])

            next_input = yolo(next_input, inp_dim, anchors)

            if not write:
                detections = next_input
                print(detections.shape)
                write = 1

            else:
                detections = np.concatenate((detections, next_input), 1)

        prev_num_filters = next_input.shape[1]
        all_activations.append(next_input)
    return detections
