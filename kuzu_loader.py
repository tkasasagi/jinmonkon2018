import torch
import torchvision
from image_folder import ImageFolderWithPaths as ImageFolder

kuzu = 'home/jinmoncom2018/data/images'

transforms = [torchvision.transforms.Resize((256,256)), torchvision.transforms.ToTensor()]
#transforms = [torchvision.transforms.ToTensor()]

kuzu_data = ImageFolder(kuzu, transform=torchvision.transforms.Compose(transforms))

data_loader = torch.utils.data.DataLoader(kuzu_data)

print(data_loader)

for x in data_loader:
  print(x[2])


