import numpy as np


def bbox_iou(box1, box2):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    bb1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)

    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    bb2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    x_left = np.maximum(b1_x1, b2_x1)
    y_top = np.maximum(b1_y1, b2_y1)
    x_right = np.minimum(b1_x2, b2_x2)
    y_bottom = np.minimum(b1_y2, b2_y2)

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):

    conf_mask = np.expand_dims((prediction[:, :, 4] > confidence).astype(float), axis=2)

    prediction = prediction * conf_mask

    box_coord = np.zeros(prediction.shape)

    box_coord[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_coord[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_coord[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_coord[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_coord[:, :, :4]

    batch_size = prediction.shape[0]

    write = False

    for b in range(batch_size):
        image_pred = prediction[b]
        max_conf = np.expand_dims(np.max(image_pred[:, 5:5 + num_classes], axis=1), axis=1)
        max_conf_indices = np.expand_dims(np.argmax(image_pred[:, 5:5 + num_classes], axis=1), axis=1)

        seq = (image_pred[:, :5], max_conf, max_conf_indices)

        image_pred = np.concatenate(seq, 1)
        positive_indices = np.squeeze(np.array(np.where(image_pred[:, 4] != 0)))

        try:
            image_pred_ = np.reshape(image_pred[positive_indices, :], (-1, 7))
        except:
            continue
        if image_pred_.shape[0] == 0:
            continue

        img_classes = np.unique(image_pred_[:, -1])

        for class_ in img_classes:

            class_mask = image_pred_ * np.expand_dims((image_pred_[:, -1] == class_), axis=1)
            class_mask_ind = np.squeeze(np.array(np.where(class_mask[:, -2] != 0)))
            image_pred_class = np.reshape(image_pred_[class_mask_ind], (-1, 7))

            conf_sort_index = np.argsort(image_pred_class[:, 4], kind='quicksort')[::-1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.shape[0]

            for i in range(idx):
                try:
                    ious = bbox_iou(np.expand_dims(image_pred_class[i], axis=0), image_pred_class[i + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                iou_mask = np.expand_dims((ious < nms_conf), axis=1)
                image_pred_class[i + 1:] *= iou_mask

                non_zero_ind = np.squeeze(np.array(np.where(image_pred_class[:, 4] != 0)))
                image_pred_class = np.reshape(image_pred_class[non_zero_ind], (-1, 7))

            batch_ind = np.full((image_pred_class.shape[0], 1), class_)
            seq = batch_ind, image_pred_class

            if not write:
                output = np.concatenate(seq, 1)
                write = True
            else:
                out = np.concatenate(seq, 1)
                output = np.concatenate((output, out))

    try:
        return output
    except:
        return 0
