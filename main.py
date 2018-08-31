# Semantic Segmentation
# Code by GunhoChoi

from FusionNet import * 
from UNet import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torchvision
import torch
from image_folder import ImageFolderWithPaths as ImageFolder
from kuzu_get_targets import loadbbox_from_csv
import random

parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="unet",help="choose between fusionnet & unet")

parser.add_argument("--batch_size",type=int,default=1,help="batch size")
parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
args = parser.parse_args()

# hyperparameters

batch_size = args.batch_size
img_size = 512
lr = 0.0002
epoch = 10

# input pipeline

#img_dir = "./maps/"
#img_data = dset.ImageFolder(root=img_dir, transform = transforms.Compose([
#                                            transforms.Scale(size=img_size),
#                                            transforms.CenterCrop(size=(img_size,img_size*2)),
#                                            transforms.ToTensor(),
#                                            ]))
#img_batch = data.DataLoader(img_data, batch_size=batch_size,
#                            shuffle=True, num_workers=2)


kuzu = 'data/images'

transforms = [torchvision.transforms.Resize((img_size,img_size)), torchvision.transforms.ToTensor()]
transforms_keep = [torchvision.transforms.ToTensor()]

kuzu_data = ImageFolder(kuzu, transform=torchvision.transforms.Compose(transforms))
img_batch = torch.utils.data.DataLoader(kuzu_data, batch_size=batch_size, shuffle=True, num_workers=2)


kuzu_keep = ImageFolder(kuzu, transform=torchvision.transforms.Compose(transforms_keep))
img_batch_keep = torch.utils.data.DataLoader(kuzu_keep, batch_size=batch_size, shuffle=True, num_workers=2)

img_sizes = {}

for _,(image,label,file_loc) in enumerate(img_batch_keep):
        img_sizes[file_loc] = image.size()
        print(file_loc, image.size())

del kuzu_keep
del img_batch_keep


# initiate Generator

if args.network == "fusionnet": #64
	generator = nn.DataParallel(FusionGenerator(3,3,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(3,3,64),device_ids=[i for i in range(args.num_gpu)]).cuda()

# load pretrained model

#try:
#    generator = torch.load('./model/{}.pkl'.format(args.network))
#    print("\n--------model restored--------\n")
#except:
#    print("\n--------model not restored--------\n")
#    pass

# loss function & optimizer

recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)

assert(batch_size == 1)

# training

file = open('./{}_mse_loss'.format(args.network), 'w')
for i in range(epoch):
    for _,(image,label,file_loc) in enumerate(img_batch):
        #satel_image, map_image = torch.chunk(image, chunks=2, dim=3) 
        
        satel_image = image*1.0
        #map_image = image*1.0

        img_file = file_loc[0].split('/')[6].rstrip('.jpg')
        #print(img_file)

        map_image = loadbbox_from_csv(img_file, img_size=image.size(), full_size=img_sizes[file_loc])

        gen_optimizer.zero_grad()

        x = Variable(satel_image).cuda(0)
        y_ = Variable(map_image).cuda(0)
        y = generator.forward(x)
        #y = Variable(x*1.0, requires_grad=True)  

        #print('sizes', y.size(), y_.size())

        loss = recon_loss_func(y,y_)
        file.write(str(loss)+"\n")
        loss.backward()
        gen_optimizer.step()


        if True:
            print(i)
            print(loss)
            overlay_real = (y_*0.2 + x*0.8)
            overlay_gen = (y*0.2 + x*0.8)
            #v_utils.save_image(x.cpu().data,"./result/original_image_{}_{}.png".format(i,_))
            #v_utils.save_image(overlay_real.cpu().data,"./result/label_image_{}_{}.png".format(i,_))
            v_utils.save_image(overlay_gen.cpu().data,"./result/gen_image_{}_{}.png".format(i,_))
            torch.save(generator,'./model/{}.pkl'.format(args.network))    




