# -*- coding: utf-8 -*-

import argparse
import os
import torch
import PairDataset
import TripletDataset
from torch.utils.data import DataLoader
from autoencoder import *
from torchvision.utils import save_image
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', help='Path of content image(or directory containing images) to be transformed')
    parser.add_argument('--style', help='Path of style image(or directory containing images) to be transformed')
    parser.add_argument('--contentSize', type=int, help='Reshape the content image to have new specified maximum size (keeping aspect ratio)')
    parser.add_argument('--styleSize', type=int, help='Reshape the style image to have new specified maximum size (keeping aspect ratio)')
    parser.add_argument('--mask', help='Path of the binary mask image (white on black) to transfer the style pair in the corrisponding areas')
    parser.add_argument('--synthesis', default=False, help='Flag to syntesize a new texture. Must provide a texture style image')
    parser.add_argument('--stylePair', help='Path of two style images(separated by ",") to use in combination')
    parser.add_argument('--gpu', default=True, help='Flag to enables GPU to accelerated computations')
    parser.add_argument('--beta', type=float, default=0.5, help='Hyperparameter balancing the interpolation between the two style images in stylePair')
    parser.add_argument('--alpha', type=float, default=0.2, help='Hyperparameter balancing the original content features and WCT-transformed features')
    parser.add_argument('--outDir', default='outputs', help='Path of the directory store stylized images')
    parser.add_argument('--outPrefix', help='Name prefixd in the saved stylized images')
    parser.add_argument('--singleLevel', default=False, help='Flag to switch to single level stylization')

    return parser.parse_args()
    
def save_img(img, args, content_name, style_name):
    save_image(img.cpu().detach().squeeze(0), 
               os.path.join(args.outDir, ((args.outPrefix+'_') if args.outPrefix else '') + content_name + '_stylized_by_'+ style_name + '_alpha_' + str(args.alpha) + '.png'))
    
def main():
    args = parse_args()
#    ipdb.set_trace()
    os.makedirs(args.outDir, exist_ok=True)
    args.device = torch.device('cuda') if args.gpu else torch.device('cpu')
    
    if args.synthesis:    args.alpha = 1
    
    if args.stylePair:
        dataset = TripletDataset.ContentStyleTripletDataset(args)
    else:
        dataset = PairDataset.ContentStylePairDataset(args)
        
    if args.singleLevel:
        model = singleLevelWCT(args)
    else:
        model = multiLevelWCT(args)
    model.to(args.device)
    model.eval()
    
    dataloader = DataLoader(dataset)
    
    for i, sample in enumerate(dataloader):
        
        if args.stylePair:
            
            content = sample['content_img'].to(device=args.device)
            style0 = sample['style0_img'].to(device=args.device)
            style1 = sample['style1_img'].to(device=args.device)
            
            content_name = str(os.path.basename(sample['contentPath'][0]).split('.')[0])
            style0_name = str(os.path.basename(sample['style0Path'][0]).split('.')[0])
            style1_name = str(os.path.basename(sample['style1Path'][0]).split('.')[0])
            
            if args.synthesis:
                for ii in range(1, 4):
                    content = model(content, style0, True, style1)
                    if args.content:
                        save_img(content, args, content_name+'_iter'+str(ii), style0_name+'_and_'+style1_name)
                    else:
                        save_img(content, args, 'texture_iter'+str(ii), style0_name+'_and_'+style1_name)
            else:
                content_name = str(os.path.basename(sample['contentPath'][0]).split('.')[0])
                out = model(content, style0, True, style1)
                save_img(out, args, content_name, style0_name+'_and_'+style1_name)
        
        else:
            content = sample['content_img'].to(device=args.device)
            style = sample['style_img'].to(device=args.device)
            
            content_name = str(os.path.basename(sample['contentPath'][0]).split('.')[0])
            style_name = str(os.path.basename(sample['stylePath'][0]).split('.')[0])
            
            out = model(content, style)
            save_img(out, args, content_name, style_name)
        

if __name__ == '__main__':
    main()





