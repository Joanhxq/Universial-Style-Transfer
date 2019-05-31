# -*- coding: utf-8 -*-


import torch
import torchvision.transforms.functional as transforms
from PIL import Image
from feature_transforms import *
from encoder_decoder_factory import *
from torch import nn

def stylize(level, content, style0, encoders, decoders, alpha, svd_device, cnn_device, interpolation_beta=None, style1=None, mask_mode=None, mask=None):

    with torch.no_grad():
        if mask_mode:
            fc = encoders[level](content).data.to(device=svd_device).squeeze(0)
            fs0 = encoders[level](style0).data.to(device=svd_device).squeeze(0)
            fs1 = encoders[level](style1).data.to(device=svd_device).squeeze(0)

            fc_channels, fc_width, fc_height = fc.size(0), fc.size(1), fc.size(2)
            mask = transforms.to_tensor(transforms.resize(mask, (fc_height, fc_width), interpolation=Image.NEAREST))

            mask_view = mask.view(-1)
            mask_view = torch.gt(mask_view, 0.5)
            foreground_mask_ix = (mask_view == 1).nonzero().type(torch.LongTensor)
            background_mask_ix = (mask_view == 0).nonzero().type(torch.LongTensor)

            fc_view = fc.view(fc_channels, -1)
            fc_fground_masked = torch.index_select(fc_view, 1, foreground_mask_ix.view(-1))
            fc_bground_masked = torch.index_select(fc_view, 1, background_mask_ix.view(-1))

            csf_fground = WCT_mask(fc_fground_masked, fs0)
            csf_bground = WCT_mask(fc_bground_masked, fs1)
            
            csf = torch.zeros_like(fc_view).t()
            csf.index_copy_(0, foreground_mask_ix.view(-1), csf_fground.t())
            csf.index_copy_(0, background_mask_ix.view(-1), csf_bground.t())
            csf = csf.t()
            csf = csf.view_as(fc)

            csf = alpha * csf + (1.0 - alpha) * fc

        elif interpolation_beta:
            fc = encoders[level](content).data.to(device=svd_device).squeeze(0)
            fs0 = encoders[level](style0).data.to(device=svd_device).squeeze(0)
            fs1 = encoders[level](style1).data.to(device=svd_device).squeeze(0)

            csf = WCT(alpha, fc, fs0, fs1, interpolation_beta)

        else:
            fc = encoders[level](content).data.to(device=svd_device).squeeze(0)
            fs0 = encoders[level](style0).data.to(device=svd_device).squeeze(0)

            csf = WCT(alpha, fc, fs0)
        
        csf = csf.unsqueeze(0).to(device=cnn_device)
        
        return decoders[level](csf)


class singleLevelWCT(nn.Module):
    def __init__(self, args):
        super(singleLevelWCT, self).__init__()
        
        self.svd_device = torch.device('cpu')
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta
        
        if args.mask:
            self.mask_mode = True
            self.mask = Image.open(args.mask).convert('1')
        else:
            self.mask_mode = False
            self.mask = None
        
        self.encoder = Encoder(5)
        self.encoders = [self.encoder]
        self.decoder = Decoder(5)
        self.decoders = [self.decoder]
        
    def forward(self, content_img, style_img, additional_style_flag=False, style1_img=None):
        if additional_style_flag:
            out = stylize(0, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device, self.cnn_device, interpolation_beta=self.beta, style1=style1_img, mask_mode=self.mask_mode, mask=self.mask)
        else:
            out = stylize(0, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device, self.cnn_device)
        return out

class multiLevelWCT(nn.Module):
    def __init__(self, args):
        super(multiLevelWCT, self).__init__()
        
        self.svd_device = torch.device('cpu')
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta
        
        if args.mask:
            self.mask_mode = True
            self.mask = Image.open(args.mask).convert('1')
        else:
            self.mask_mode = False
            self.mask = None
            
        self.e1 = Encoder(1)
        self.e2 = Encoder(2)
        self.e3 = Encoder(3)
        self.e4 = Encoder(4)
        self.e5 = Encoder(5)
        self.encoders = [self.e5, self.e4, self.e3, self.e2, self.e1]
        
        self.d1 = Decoder(1)
        self.d2 = Decoder(2)
        self.d3 = Decoder(3)
        self.d4 = Decoder(4)
        self.d5 = Decoder(5)
        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]
        
    def forward(self, content_img, style_img, additional_style_flag=False, style1_img=None):
        for i in range(len(self.encoders)):
            if additional_style_flag:
                content_img = stylize(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device, self.cnn_device, interpolation_beta=self.beta, style1=style1_img, mask_mode=self.mask_mode, mask=self.mask)
            else:
                content_img = stylize(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device, self.cnn_device)
        return content_img
    
