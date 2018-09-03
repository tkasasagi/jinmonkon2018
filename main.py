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
from get_image_size import get_image_size

parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="unet",help="choose between fusionnet & unet")

parser.add_argument("--batch_size",type=int,default=1,help="batch size")
parser.add_argument("--num_gpu",type=int,default=2,help="number of gpus")
args = parser.parse_args()

# hyperparameters

batch_size = args.batch_size
img_size = 512
lr = 0.0002
epoch = 100

# input pipeline

#img_dir = "./maps/"
#img_data = dset.ImageFolder(root=img_dir, transform = transforms.Compose([
#                                            transforms.Scale(size=img_size),
#                                            transforms.CenterCrop(size=(img_size,img_size*2)),
#                                            transforms.ToTensor(),
#                                            ]))
#img_batch = data.DataLoader(img_data, batch_size=batch_size,
#                            shuffle=True, num_workers=2)

kuzu = './data'

transforms = [torchvision.transforms.Resize((img_size,img_size)), torchvision.transforms.ToTensor()]
transforms_keep = [torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]

kuzu_data = ImageFolder(kuzu, transform=torchvision.transforms.Compose(transforms))
img_batch = torch.utils.data.DataLoader(kuzu_data, batch_size=batch_size, shuffle=True, 
                                        num_workers=2)


kuzu_keep = ImageFolder(kuzu, transform=torchvision.transforms.Compose(transforms_keep))
img_batch_keep = torch.utils.data.DataLoader(kuzu_keep, batch_size=batch_size, shuffle=True, 
                                             num_workers=2)

img_sizes = {}

all_image_lst = []

#This loop is for getting file sizes and a list of all image names
for _,(image,label,file_loc) in enumerate(img_batch_keep):
    #img_sizes[file_loc] = image.size()
    #print('file loc', file_loc)
    size2 = get_image_size(file_loc[0])
    new_size = torch.Size([1,3,size2[1],size2[0]])
    print(file_loc,new_size)
    img_sizes[file_loc] = new_size
    all_image_lst.append(file_loc[0])

print("all image list", all_image_lst)

#This randomly takes 50 images for the "test set".  
test_images = random.sample(all_image_lst, 50)

del kuzu_keep
del img_batch_keep

# initiate Generator

#This initializes the generator object
if args.network == "fusionnet": #64
	generator = nn.DataParallel(FusionGenerator(3,49,64),
                             device_ids=[i for i in range(args.num_gpu)]).cuda()
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(3,49,64), 
                             device_ids=[i for i in range(args.num_gpu)]).cuda()

# load pretrained model

#try:
#    generator = torch.load('./model/{}.pkl'.format(args.network))
#    print("\n--------model restored--------\n")
#except:
#    print("\n--------model not restored--------\n")
#    pass

# loss function & optimizer

#recon_loss_func = nn.MSELoss()
recon_loss_func = nn.BCELoss()

gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)

assert(batch_size == 1)

# training

file = open('./{}_mse_loss'.format(args.network), 'w')
for i in range(epoch):
    for iteration,(image,label,file_loc) in enumerate(img_batch):
        
        satel_image = image*1.0

        img_file = file_loc[0].split('/')[4].rstrip('.jpg')


        if file_loc[0] in test_images:
            is_test = True
        else:
            is_test = False

        map_image = loadbbox_from_csv(img_file, img_size=image.size(), 
                                      full_size=img_sizes[file_loc])

        if map_image is None:
            continue

        gen_optimizer.zero_grad()

        x = Variable(satel_image).cuda(0)
        y_ = Variable(map_image).cuda(0)
        y = generator.forward(x)

        loss = recon_loss_func(y,y_)
        file.write(str(loss)+"\n")

        loss.backward()
        if not is_test:
            gen_optimizer.step()


        if iteration % 5 == 0:
            print(i)
            print(loss)
            x = x.cpu().data
            ysum = y.cpu().data.max(dim=1,keepdim=True)[0].repeat(1,3,1,1)
            ysum_ = y_.cpu().data.max(dim=1,keepdim=True)[0].repeat(1,3,1,1)
            y = y.cpu().data[:,0:3,:,:]
            y_ = y_.cpu().data[:,0:3,:,:]
            overlay_real = (y_*0.3 + x*0.7)
            overlay_gen = (y*0.3 + x*0.7)
            overlay_sum = (ysum*0.3 + x*0.7)
            overlay_sumreal = (ysum_*0.3 + x*0.7)
            v_utils.save_image(overlay_real,"./result/label_image_{}_{}.png".format(i,iteration))
            v_utils.save_image(overlay_gen,"./result/gen_image_{}_{}_{}.png".format(i,iteration,is_test))
            v_utils.save_image(overlay_sum,"./result/sumgen_image_{}_{}.png".format(i,iteration))
            v_utils.save_image(overlay_sumreal,"./result/sumreal_image_{}_{}.png".format(i,iteration))

            torch.save(generator,'./model/{}.pkl'.format(args.network))    




