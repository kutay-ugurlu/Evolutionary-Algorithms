import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2

nrow = 11
ncol = 1

fig = plt.figure(figsize=(4, 10)) 
EXP_NAME = "Results : Default Parameters"

gs = gridspec.GridSpec(nrow, ncol,height_ratios=[1]*11,
         wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 

for i in range(nrow):
    for j in range(ncol):
        im = cv2.imread("OUTPUTs\\Default_Params_ALLCV_increased_min_rad__cv_plt_generation_"+str(1000*i)+".png")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)        
        ax = plt.subplot(gs[i,j])
        plt.ylabel("Iteration\n"+str(i*1000))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(im)

fig = plt.gcf()
fig.suptitle(EXP_NAME, fontsize=14)

#plt.tight_layout() # do not use this!!
plt.show()