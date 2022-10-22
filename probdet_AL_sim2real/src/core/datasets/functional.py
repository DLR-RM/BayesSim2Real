import math, torch, torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as T
import matplotlib.pyplot as plt

def solarize_add(img, addition, threshold):
    img = F.pil_to_tensor(img)
    added_img = img + addition
    added_img = torch.clamp(added_img, 0, 255)
    return F.to_pil_image(torch.where(img < threshold, added_img, img))


def color(img, magnitude):
    return ImageEnhance.Color(img).enhance(magnitude)


def contrast(img, magnitude):
    return ImageEnhance.Contrast(img).enhance(magnitude)


def brightness(img, magnitude):
    return ImageEnhance.Brightness(img).enhance(magnitude)


def sharpness(img, magnitude):
    return ImageEnhance.Sharpness(img).enhance(magnitude)


def crop_and_resize_x(img, bboxs, pixels):
    width = img.size[0] - pixels
    for bbox in bboxs:
        if bbox[0] + pixels < 0:
            bbox[0] = 0
        else:
            bbox[0] = width * (bbox[0] + pixels) / img.size[0]
            bbox[2] = width * (bbox[2] + pixels) / img.size[0]
    return bboxs



def crop_and_resize_y(img, bboxs, pixels):
    height = img.size[1] - pixels
    for bbox in bboxs:
        if bbox[1] + pixels < 0:
            bbox[1] = 0
        else:
            bbox[1] = height * (bbox[1] + pixels) / img.size[1]
            bbox[3] = height * (bbox[3] + pixels) / img.size[1]
    return bboxs


def shear_with_bboxes(img, bboxs, level, replace, shift_horizontal):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    
    shear_bboxs = []
    if shift_horizontal:
        shear_matrix = torch.FloatTensor([[1, -level], 
                                          [0, 1]])
        for bbox in bboxs:
            min_x, min_y, max_x, max_y = bbox
            coords = torch.FloatTensor([[min_x, min_y], 
                                        [min_x, max_y], 
                                        [max_x, max_y], 
                                        [max_x, min_y]])
            shear_coords = torch.matmul(shear_matrix, coords.t()).t()
            x_min, y_min = torch.min(shear_coords, dim=0)[0]
            x_max, y_max = torch.max(shear_coords, dim=0)[0]
            shear_min_x, shear_max_x = torch.clamp(x_min, 0, w), torch.clamp(x_max, 0, w)
            shear_min_y, shear_max_y = torch.clamp(y_min, 0, h), torch.clamp(y_max, 0, h)
            shear_bboxs.append(torch.FloatTensor([shear_min_x, shear_min_y, shear_max_x, shear_max_y]))
    else:
        shear_matrix = torch.FloatTensor([[1, 0], 
                                          [-level, 1]])
        for bbox in bboxs:
            min_x, min_y, max_x, max_y = bbox
            coords = torch.FloatTensor([[min_x, min_y], 
                                        [min_x, max_y], 
                                        [max_x, max_y], 
                                        [max_x, min_y]])
            shear_coords = torch.matmul(shear_matrix, coords.t()).t()
            x_min, y_min = torch.min(shear_coords, dim=0)[0]
            x_max, y_max = torch.max(shear_coords, dim=0)[0]
            shear_min_x, shear_max_x = torch.clamp(x_min, 0, w), torch.clamp(x_max, 0, w)
            shear_min_y, shear_max_y = torch.clamp(y_min, 0, h), torch.clamp(y_max, 0, h)
            shear_bboxs.append(torch.FloatTensor([shear_min_x, shear_min_y, shear_max_x, shear_max_y]))
    return torch.stack(shear_bboxs)


def shear_only_bboxes(img, bboxs, p, level, replace, shift_horizontal):
    img = F.pil_to_tensor(img)
    shear_img = torch.zeros_like(img)
    
    for bbox in bboxs:
        if torch.rand(1) < p and bbox[2] - bbox [0] > 0 and bbox[3] - bbox[1] > 0:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            
            bbox_shear_img = F.to_pil_image(img[:, min_y:max_y+1, min_x:max_x+1])
            if shift_horizontal:
                bbox_shear_img = bbox_shear_img.transform(bbox_shear_img.size, Image.AFFINE, (1,level,0,0,1,0), fillcolor=(replace,replace,replace))
            else:
                bbox_shear_img = bbox_shear_img.transform(bbox_shear_img.size, Image.AFFINE, (1,0,0,level,1,0), fillcolor=(replace,replace,replace))
            shear_img[:, min_y:max_y+1, min_x:max_x+1] = F.pil_to_tensor(bbox_shear_img)

    return F.to_pil_image(torch.where(shear_img != 0, shear_img, img))


def flip_only_bboxes(img, bboxs, p):
    img = F.pil_to_tensor(img)
    flip_img = torch.zeros_like(img)
    
    for bbox in bboxs:
        if torch.rand(1) < p and bbox[2] - bbox [0] > 0 and bbox[3] - bbox[1] > 0:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            flip_img[:, min_y:max_y+1, min_x:max_x+1] = F.hflip(img[:, min_y:max_y+1, min_x:max_x+1])
    
    return F.to_pil_image(torch.where(flip_img != 0, flip_img, img))


def solarize_only_bboxes(img, bboxs, p, threshold):
    img = F.pil_to_tensor(img)
    for bbox in bboxs:
        if torch.rand(1) < p and bbox[2] - bbox [0] > 0 and bbox[3] - bbox[1] > 0:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            solarize_img = img[:, min_y:max_y+1, min_x:max_x+1]
            solarize_img = F.to_pil_image(solarize_img)
            solarize_img = ImageOps.solarize(solarize_img, threshold=threshold)
            solarize_img = F.pil_to_tensor(solarize_img)
            img[:, min_y:max_y+1, min_x:max_x+1] = solarize_img
    return F.to_pil_image(img)


def equalize_only_bboxes(img, bboxs, p):
    img = F.pil_to_tensor(img)
    for bbox in bboxs:
        if torch.rand(1) < p and bbox[2] - bbox [0] > 0 and bbox[3] - bbox[1] > 0:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            equalize_img = img[:, min_y:max_y+1, min_x:max_x+1]
            equalize_img = F.to_pil_image(equalize_img)
            equalize_img = ImageOps.equalize(equalize_img)
            equalize_img = F.pil_to_tensor(equalize_img)
            img[:, min_y:max_y+1, min_x:max_x+1] = equalize_img
    return F.to_pil_image(img)