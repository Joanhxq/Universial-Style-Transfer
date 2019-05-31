# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
from im_utils import *

supported_img_formats = ('.png', '.jpg', '.jpeg')

class ContentStylePairDataset(Dataset):
    
    def __init__(self, args):
        super(ContentStylePairDataset, self).__init__()
        
        self.styleSize = args.styleSize
        self.contentSize = args.contentSize
        
        if args.style.endswith(supported_img_formats) and args.content.endswith(supported_img_formats):
            self.pairs_fn = [(args.content, args.style)]
        elif not args.content.endswith(supported_img_formats) and args.style.endswith(supported_img_formats):
            self.pairs_fn = []
            for c in os.listdir(args.content):
                path_pair = (os.path.join(args.content, c), args.style)
                self.pairs_fn.append(path_pair)
        elif not args.style.endswith(supported_img_formats) and args.content.endswith(supported_img_formats):
            self.pairs_fn = []
            for s in os.listdir(args.style):
                path_pair = (args.content, os.path.join(args.style, s))
                self.pairs_fn.append(path_pair)
        else:
            self.pairs_fn = []
            for c in os.listdir(args.content):
                for s in os.listdir(args.style):
                    path_pair = (os.path.join(args.content, c), os.path.join(args.style, s))
                    self.pairs_fn.append(path_pair)


    def __getitem__(self, index):
        pair = self.pairs_fn[index]
        
        contentPath = pair[0]
        stylePath = pair[1]
        
        content_img = load_img(contentPath, self.contentSize)
        style_img = load_img(stylePath, self.styleSize)
        
        return {'content_img': content_img, 'contentPath': pair[0], 'style_img': style_img, 'stylePath': pair[1]}

    def __len__(self):
        return len(self.pairs_fn)




