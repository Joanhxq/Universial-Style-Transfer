# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
from im_utils import *
import torch

supported_img_formats = ('.png', '.jpg', '.jpeg')

class ContentStyleTripletDataset(Dataset):
    
    def __init__(self, args):
        super(ContentStyleTripletDataset, self).__init__()
        
        self.synthesis = args.synthesis
        self.styleSize = args.styleSize
        self.contentSize = args.contentSize
        self.style0 = args.stylePair.split(',')[0]
        self.style1 = args.stylePair.split(',')[1]
        self.content = args.content
        
        if self.synthesis:
            if not args.content:
                self.triplets_fn = [('texture', self.style0, self.style1)]
            elif args.content.endswith(supported_img_formats):
                self.triplets_fn = [(args.content, self.style0, self.style1)]
            else:
                self.triplets_fn = []
                for c in os.listdir(args.content):
                    path_triplet = (os.path.join(args.content, c), self.style0, self.style1)
                    self.triplets_fn.append(path_triplet)
        elif args.content.endswith(supported_img_formats):
            self.triplets_fn = [(args.content, self.style0, self.style1)]
        else:
            self.triplets_fn = []
            for c in os.listdir(args.content):
                path_triplet = (os.path.join(args.content, c), self.style0, self.style1)
                self.triplets_fn.append(path_triplet)

    def __getitem__(self, index):
        triplet = self.triplets_fn[index]
        
        contentPath = triplet[0]
        style0Path = triplet[1]
        style1Path = triplet[2]
        
        style0_img = load_img(style0Path, self.styleSize)
        style1_img = load_img(style1Path, self.styleSize)
        
        if self.synthesis:
            if self.content:
                content_img = load_img(contentPath, self.contentSize)
            else:
                c_c, c_h, c_w = style0_img.size()
                content_img = torch.zeros((c_c, c_h, c_w)).uniform_()
        else:
            content_img = load_img(contentPath, self.contentSize)
        
        return {'content_img': content_img, 'contentPath': triplet[0], 'style0_img': style0_img, 'style0Path': triplet[1], 'style1_img': style1_img, 'style1Path': triplet[2]}

    def __len__(self):
        return len(self.triplets_fn)




