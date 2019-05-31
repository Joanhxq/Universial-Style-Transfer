# -*- coding: utf-8 -*-


from PIL import Image
import torchvision.transforms.functional as transforms


def load_img(path, new_size):
    img = Image.open(path).convert('RGB')

    if new_size:
        width, height = img.size
        
        if width > height:
            new_shape = (int(new_size*height/width), new_size)  # new_shape=(height,width)
        else:
            new_shape = (new_size, int(new_size*width/height))
            
        img = transforms.resize(img, new_shape, Image.BICUBIC)
    
    return transforms.to_tensor(img)































