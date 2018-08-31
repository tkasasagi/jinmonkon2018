import torch

fh = open('data/hnsd00000_coordinate.csv')

lines = []

for line in fh:
  lines.append(line.rstrip('\n'))

obj = {}

for line in lines[1:]:
  obj[line.split(',')[1]] = []

for line in lines[1:]:
  if "U+306F" in line:
    linesp = line.split(',')
    obj[linesp[1]].append([int(linesp[2]),int(linesp[3]),int(linesp[6]),int(linesp[7])])

images = list(obj.keys())


#target = '200014740-00004_1'

#xresize = 4.4394
#yresize = 5.9824
#xresize=1.0
#yresize=1.0

def loadbbox_from_csv(target_img, img_size, full_size): 
  bb_lst = obj[target_img]
  target_tensor = torch.zeros(size=img_size)

  xresize = full_size[3]/img_size[3]
  yresize = full_size[2]/img_size[2]

  #print(target_img, bb_lst)
  for bb in bb_lst:
    xpos, ypos, width, height = bb
    xpos = int(xpos/xresize)
    width = int(width/xresize)
    ypos = int(ypos/yresize)
    height = int(height/yresize)

    target_tensor[:,:,ypos:ypos+height,xpos:xpos+width] += 1.0

  #print(target_tensor.size())

  return target_tensor

if __name__ == '__main__':
  loadbbox_from_csv(images[2])


