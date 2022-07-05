from torchvision.datasets import MNIST
from torchvision import transforms
import cv2
import os
import numpy as np
from tqdm import tqdm

root='datasets/'
subffix='.png'

trans=transforms.Compose([transforms.ToTensor()])
trainset=MNIST(root,train=True,transform=trans,download=True)
testset=MNIST(root,train=False,transform=trans,download=True)
def make_image(directory,set):
    if not os.path.exists(root+directory):
        os.makedirs(root+directory)
    for i,item in tqdm(enumerate(set),desc=directory):
        image=np.uint8(item[0].numpy()[0]*255)
        cv2.imwrite(root+directory+str(i)+subffix,image)

make_image('MNIST/images/train/',trainset)
make_image('MNIST/images/test/',testset)
    