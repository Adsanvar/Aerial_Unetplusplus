import gradio as gr
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms as T
import os
import cv2
import pandas as pd
import albumentations as album

class ExperimentDataset(torch.utils.data.Dataset):
    def __init__(self, image, augment=None, preprocess=None):
        self.image = image
        self.augment = augment
        self.preprocess = preprocess
        
    def __getitem__(self, i):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if self.augment:
            sample = self.augment(image=image)
            image= sample['image']
        if self.preprocess:
            sample = self.preprocess(image=image)
            image = sample['image']
        return image

def color_convert(image,labelvals):
    colorcodes = np.array(labelvals)
    ccs = colorcodes[image.astype(int)]
    return ccs

def crop_image(image, dims=[1500,1500,3]):
    target_size = dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,]

def to_tensor(x,**kwargs):
    return x.transpose(2,0,1).astype("float32")

def augment_image(): 
    transform = [album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0)]
    return album.Compose(transform)

def preprocessing(preprocessing_fn=None):
    transform = []
    if preprocessing_fn:
        transform.append(album.Lambda(image=preprocessing_fn))
    transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(transform)

def segment(image):
    device = torch.device("cpu")

    best_model = torch.load('../report/unetpp/best_model_upp.pth', map_location=device)
    
    classlabeldict = pd.read_csv("../report/label_class_dict.csv")
    clasnames = classlabeldict['name'].tolist()
    class_rgb_values = classlabeldict[['r','g','b']].values.tolist()
    select_class_indices = [clasnames.index(cls.lower()) for cls in clasnames]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    encoder = "resnet34"
    encoder_weights = "imagenet"
    preprocess_func = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    exp_data = ExperimentDataset(image, augment = augment_image(),
                           preprocess = preprocessing(preprocess_func))

    test_img = exp_data[0]
    x_tensor = torch.from_numpy(test_img).to(device).unsqueeze(0)
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask,(1,2,0))
    pred_mask = crop_image(color_convert(np.argmax(pred_mask,axis=-1), select_class_rgb_values))
    
    return pred_mask

iface = gr.Interface(fn=segment, inputs="image", outputs="image").launch()