import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, 
                    help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, 
                    help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
r_list = np.zeros((seq.shape[2]-1,4))

h,w,f = np.shape(seq)
for frame in range(f-1):
        It  = seq[:, :, frame]
        It1 = seq[:, :, frame+1]
        l = LucasKanade(It, It1, rect, threshold, num_iters)
        rect[0] += l[0] #x1
        rect[1] += l[1] #y1
        rect[2] += l[0] #x2
        rect[3] += l[1] #y2

        r_list[frame] = rect

        if (frame % 100 == 0 or frame == 0):

            plt.figure()
            plt.imshow(seq[:,:,frame], cmap='gray')
            rectangle = patches.Rectangle((int(rect[0]), int(rect[1])), (rect[2]-rect[0]), 
                                    (rect[3]-rect[1]), fill=False, edgecolor='r', linewidth=2)
            
            plt.gca().add_patch(rectangle)
            plt.title('frame %d'%frame)
            plt.savefig('carseqframe' + str(frame) + '.png', bbox_inches='tight')
            plt.show()
            
np.save('carseqrects.npy', r_list)   

