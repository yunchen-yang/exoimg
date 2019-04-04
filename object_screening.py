import cv2
import numpy as np
import sys
import copy

def thresholding_op(img_mat,T,R,S1,S2):
    #S1&S2 divide the image into S1xS2 individual regions
    img_x=img_mat.shape[0]
    img_y=img_mat.shape[1]
    gap_x=int(img_x/S1)
    gap_y=int(img_y/S2)
    img_t=np.zeros(img_mat.shape).astype('uint8')
    for i in range(0,S1):
        for j in range(0,S2):
            if i<S1-1 and j<S2-1:
                ind_x=range(i*gap_x,(i+1)*gap_x)
                ind_y=range(j*gap_y,(j+1)*gap_y)
            elif i==S1-1 and j<S2-1:
                ind_x=range(i*gap_x,img_x)
                ind_y=range(j*gap_y,(j+1)*gap_y)
            elif i<S1-1 and j==S2-1:
                ind_x=range(i*gap_x,(i+1)*gap_x)
                ind_y=range(j*gap_y,img_y)
            elif i==S1-1 and j==S2-1:
                ind_x=range(i*gap_x,img_x)
                ind_y=range(j*gap_y,img_y)
            point=[]
            for x in ind_x:
                for y in ind_y:
                    point.append(np.array([x,y,img_mat[x,y]]))
            ave=np.mean(np.array(list(filter((lambda x:x[2]<T),point)))[:,2])
            point_select=list(filter((lambda x:x[2]>=T),point))
            if point_select!=None:
                for n in range(0,len(point_select)):
                    if point_select[n][2]>=ave*(1+R):
                        #the segmented pixel intensity should be larger than the regional mean instensity as well
                        img_t[point_select[n][0],point_select[n][1]]=img_mat[point_select[n][0],point_select[n][1]]
                    else:
                        continue
            else:
                continue
    return img_t
    
def background_eraser(bwmap):
    bwmap[0, :] = 0
    bwmap[bwmap.shape[0]-1, :] = 0
    bwmap[:, 0] = 0
    bwmap[:, bwmap.shape[1]-1] = 0
    
    trigger_num = 1
    bw_dele = copy.deepcopy(bwmap)
    while trigger_num != 0:
        [edge_x,  edge_y] = np.where(bwmap==1)
        #intialize the vote matirx
        vote_surr = np.zeros((len(edge_y),))
        #start  the voting
        for n in range(0, len(edge_y)):
            vote_surr[n] = np.sum(bwmap[(edge_x[n] - 1): (edge_x[n] + 2), (edge_y[n] - 1): (edge_y[n] + 2)])
        
        #removing edge pixels where voting = 1, 2
        index_2 = np.where(vote_surr < 3)[0]
        for n_2 in range (0, len(index_2)):
            bw_dele[edge_x[index_2[n_2]], edge_y[index_2[n_2]]] = 0
        
        #removing edge pixels where voting = 3, 4
        index_3 = np.where(vote_surr == 3)[0]
        for n_3 in range(0, len(index_3)):
            frame = copy.deepcopy(bwmap[edge_x[index_3[n_3]] - 1: edge_x[index_3[n_3]] + 2, edge_y[index_3[n_3]] - 1: edge_y[index_3[n_3]] + 2])
            frame[1, 1] = 0
            [f_x, f_y] = np.where(frame==1)
            
            if f_x[0] == f_x[1] and abs(f_y[0] - f_y[1]) == 1:
                bw_dele[edge_x[index_3[n_3]], edge_y[index_3[n_3]]] = 0
            if f_y[0] == f_y[1] and abs(f_x[0] - f_x[1]) == 1:
                bw_dele[edge_x[index_3[n_3]], edge_y[index_3[n_3]]] = 0
        
        index_4 = np.where(vote_surr==4)[0]
        for n_4 in range(0, len(index_4)):
            frame = copy.deepcopy(bwmap[edge_x[index_4[n_4]] - 1: edge_x[index_4[n_4]] + 2, edge_y[index_4[n_4]] - 1: edge_y[index_4[n_4]] + 2])
            frame[1, 1] = 0
            sum_1 = np.sum(frame[0: 2, 0: 2])
            sum_2 = np.sum(frame[1: 3, 0: 2])
            sum_3 = np.sum(frame[0: 2, 1: 3])
            sum_4 = np.sum(frame[1: 3, 1: 3])
            
            if np.any([sum_1==3, sum_2==3, sum_3==3, sum_4==3]):
                bw_dele[edge_x[index_4[n_4]], edge_y[index_4[n_4]]] = 0
        
        trigger_num = np.sum(bwmap) - np.sum(bw_dele)
        bwmap = copy.deepcopy(bw_dele)
    bw_dele *= 255
    bw_dele = bw_dele.astype('uint8')    
    return bw_dele

path=sys.path[0]+'/'
#type in the file name here
filename = "Noise1" 
img=cv2.imread(path+filename +".tif", 0)
img_cp = copy.deepcopy(img)

cv2.imshow("original_img", img_cp)

img_seg=thresholding_op(img_cp,20,0.8,5,5)
cv2.imshow("segmented_img", img_seg)
#the regional segmentation did not work very well

edge_img = cv2.Canny(img_cp, 40, 60)
cv2.imshow("edge_img with noise", edge_img)
#cv2.imwrite(path+filename + "_edge with noise.tif", edge_FAM)

edge_screened = background_eraser(edge_img/255)
cv2.imshow("edge_img without noise", edge_screened)
#cv2.imwrite(path+filename + "_edge without noise.tif", edge_FAM_screened)
       
mask=np.zeros((edge_img.shape[0]+2, edge_img.shape[0]+2)).astype('uint8')
mask[1:mask.shape[0]-1, 1:mask.shape[1]-1] = edge_screened[:,:]
img_copy = copy.deepcopy(edge_screened).astype('uint8')
cv2.floodFill(img_copy, mask, (0,0), 255)
img_fill = cv2.bitwise_not(img_copy) + edge_screened
cv2.imshow("edge_img without noise_filled", img_fill)
#cv2.imwrite(path+filename + "_filled.tif", img_fill)                
            
