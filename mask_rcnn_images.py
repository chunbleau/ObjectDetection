import numpy as np
import torch
import torchvision
import argparse
from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms

import cv2  # remove

def run_model(image_path):
    print("start")

    # initialize the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                               num_classes=91)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()

    print(device)
    # transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # image_path = "C:\\Users\\CMOS-DEV\\PycharmProjects\\input\\image1.jpg"
    image = Image.open(image_path).convert('RGB')
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()       # remove

    # transform the image
    image = transform(image)
    print(image.shape)
    height = image.shape[1]
    width = image.shape[2]
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(image, model, 0.965)
    masks_out = np.reshape(masks, (-1, width))
    np.savetxt(image_path[:-4] + "_masks.txt", masks_out, delimiter=",", fmt='%d')
    boxes_out = np.reshape(boxes, (-1, 2))
    np.savetxt(image_path[:-4] + "_boxes.txt", boxes_out, delimiter=",", fmt='%d')

    # remove
    result = draw_segmentation_map(orig_image, masks, boxes, labels)
    # visualize the image
    cv2.imshow('Segmented image', result)
    cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()

   # parser.add_argument('--Img', type=str, required=True)
    parser.add_argument('--ImgPath', type=str, default=r"/Users/chunbleau/PycharmProjects/input/image1.jpg")
    parser.add_argument('--threshold', default=0.965, type=float,
                        help='score threshold for discarding detection')
    args = vars(parser.parse_args())
    image_path = args['ImgPath']
    run_model(image_path)

if __name__ == '__main__':
    main()
