csv_loc = '/home/tarin/jinmoncom2018/data/'

import torch

fh1 = open(csv_loc + '200014740/200014740_coordinate.csv')
fh2 = open(csv_loc + '200003076/200003076_coordinate.csv')
#target_char = "U+306E"

num_char = 49
char2ind = {}
char2ind['U+306E'] = 0
char2ind['U+306B'] = 1
char2ind['U+3066'] = 2
char2ind['U+3057'] = 3
char2ind['U+306F'] = 4
char2ind['U+3068'] = 5
char2ind['U+304B'] = 6
char2ind['U+3082'] = 7
char2ind['U+3092'] = 8
char2ind['U+308B'] = 9
char2ind['U+306A'] = 10
char2ind['U+3089'] = 11
char2ind['U+305F'] = 12
char2ind['U+3075'] = 13
char2ind['U+304D'] = 14
char2ind['U+304F'] = 15
char2ind['U+308C'] = 16
char2ind['U+3044'] = 17
char2ind['U+3042'] = 18
char2ind['U+307E'] = 19
char2ind['U+304C'] = 20

char2ind['U+3072'] = 21
char2ind['U+3053'] = 22
char2ind['U+3078'] = 23
char2ind['U+7D66'] = 24
char2ind['U+3055'] = 25
char2ind['U+3070'] = 26
char2ind['U+4EBA'] = 27
char2ind['U+3088'] = 28
char2ind['U+3064'] = 29
char2ind['U+309D'] = 30
char2ind['U+3059'] = 31

char2ind['U+305B'] = 32
char2ind['U+3051'] = 33
char2ind['U+3093'] = 34
char2ind['U+3081'] = 35
char2ind['U+307F'] = 36
char2ind['U+3084'] = 37
char2ind['U+305A'] = 38
char2ind['U+306C'] = 39
char2ind['U+898B'] = 40
char2ind['U+3069'] = 41
char2ind['U+6B64'] = 42

char2ind['U+304A'] = 43
char2ind['U+305D'] = 44
char2ind['U+3079'] = 45
char2ind['U+3031'] = 46
char2ind['U+3046'] = 47
char2ind['U+3080'] = 48

lines = []

header = fh1.readline()
for line in fh1:
  lines.append(line.rstrip('\n'))

header = fh2.readline()
for line in fh2:
  lines.append(line.rstrip('\n'))

obj = {}

for line in lines:
  obj[line.split(',')[1]] = []
  
#print(obj)

for line in lines[1:]:
  linesp = line.split(',')
  char = linesp[0]
  if char in char2ind:
    obj[linesp[1]].append([int(linesp[2]),int(linesp[3]),int(linesp[6]),int(linesp[7]),char2ind[char]])

images = list(obj.keys())

#print(images)
#target = '200014740-00004_1'

#xresize = 4.4394
#yresize = 5.9824
#xresize=1.0
#yresize=1.0

def loadbbox_from_csv(target_img, img_size, full_size): 

  target_tensor = torch.zeros(size=(img_size[0],num_char,img_size[2],img_size[3]))
  #target_tensor[:,-1,:,:] += 1.0

  if target_img not in obj:
    print("targets not found")
    #return None
    return target_tensor

  bb_lst = obj[target_img]

  xresize = full_size[3]/img_size[3]
  yresize = full_size[2]/img_size[2]

  #print(target_img, bb_lst)
  for bb in bb_lst:
    xpos, ypos, width, height, char_ind = bb
    xpos = int(xpos/xresize)
    width = int(width/xresize)
    ypos = int(ypos/yresize)
    height = int(height/yresize)

    target_tensor[:,char_ind,ypos:ypos+height,xpos:xpos+width] += 1.0

  #print(target_tensor.size())

  return target_tensor

if __name__ == '__main__':
  loadbbox_from_csv(images[2])






