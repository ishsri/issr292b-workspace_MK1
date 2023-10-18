from PIL import Image
import glob
import os
import random

filelist = []
for name in glob.glob('../data/npm1/non_npm1/*.png'):
    filelist.append(name)


    
print(len(filelist))
#random.seed(4)
#sample_list = random.sample(filelist, 468)
#print(len(sample_list))

#for name in sample_list:
#    os.remove(name)