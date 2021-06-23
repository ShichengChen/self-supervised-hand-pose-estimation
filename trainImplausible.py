import os
import platform
if (platform.node()=='csc-G7-7590'):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
import torch
from cscPy.Nets.AligningHandnet import encoderRGB,decPoseNet
from cscPy.dataloader.MVdataloader import MVDataloader
from cscPy.mano.dataloader.synthesizer import ManoSynthesizer
from cscPy.Loss.utils import pose3d_loss,LossHelper,cloud_dis,cloud_dis2
import numpy as np
from cscPy.mano.network.manoArmLayer import MANO_SMPL
from cscPy.mano.network.biomechanicalLoss import BiomechanicalLayer
from cscPy.globalCamera.camera import CameraIntrinsics,perspective_projection
from cscPy.globalCamera.util import fetch_all_sequences,load_rgb_maps,load_depth_maps,get_cameras_from_dir,visualize_better_qulity_depth_map
from tqdm import tqdm
import trimesh
import cv2
from cscPy.mano.network.utils import *
import io
from PIL import Image
trans0=np.array([[ 0.9314455 , -0.21834283,  0.29109396 , 0.13697226],
     [ 0.34659251 , 0.28870036 ,-0.8924829 , -0.25179651],
     [ 0.11082831,  0.93219017 , 0.34458464 , 0.12970769],
     [ 0.         , 0.        ,  0.         , 1.        ]])
trans1 = np.array([[ 0.95338962,  0.29744177,  0.05076046,  0.06649263],
     [ 0.27769673, -0.79910434, -0.53321363, -0.14445236],
     [-0.11803711,  0.52245635, -0.84445639, -0.20042289],
     [ 0.        ,  0.         , 0.         , 1.        ]])
trans2 = np.array([[ 0.77995301, -0.05338033, -0.62355741, -0.11169595],
     [ 0.48834376, -0.57121169,  0.65972538,  0.17433204],
     [-0.39139965, -0.81906516, -0.41945033, -0.07399003],
     [ 0.        ,  0.        ,  0.        ,  1.        ]])
trans3 = np.array([[ 0.54834784, -0.79624552, -0.25555373, -0.03479103],
     [ 0.01342946,  0.31394009, -0.94934781, -0.33953577],
     [ 0.83614252,  0.51714087,  0.18284152,  0.08128843],
     [ 0.        ,  0.        ,  0.        ,  1.        ]])
trans=[trans0,trans1,trans2,trans3]
biolayer = BiomechanicalLayer(fingerPlaneLoss=True, fingerFlexLoss=True, fingerAbductionLoss=True)
mano_right = MANO_SMPL(manoPath, ncomps=45, oriorder=True,device='cpu',userotJoints=True)
pose=torch.tensor([[0,0,3.14],[0,0,3.14],[0,0,3.14],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype=torch.float32,requires_grad=True)
pose=torch.randn([45],dtype=torch.float32,requires_grad=True)
optimizer = torch.optim.Adam([pose], lr=1e-2,weight_decay=1e-5)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('bio.avi', fourcc, 20.0, (640*4,480))
for epoch in range(300):

    rootr = torch.zeros([3],dtype=torch.float32)*3.14*2-3.14
    vertex_gt, joint_gt = mano_right.get_mano_vertices(rootr.view(1, 1, 3),
                                                       pose.view(1, 45)*3.14-3.14/2,
                                                       torch.zeros([10],dtype=torch.float32).view(1, 10),
                                                       torch.tensor([1],dtype=torch.float32).view(1, 1),
                                                       torch.tensor([[0, 0, 0]],dtype=torch.float32).view(1, 3),
                                                       pose_type='euler', mmcp_center=False)
    vertex_colors = (np.ones([vertex_gt.shape[1],4])).astype(np.uint8)
    vertex_colors[:,2]*=255
    vertex_colors[:,-1]*=150


    def getview(vi):
        v = trimesh.Trimesh(vertices=vertex_gt.detach().numpy()[0], faces=mano_right.faces, vertex_colors=vertex_colors)
        scene = trimesh.Scene(v)
        #print(trans[vi])
        scene.camera_transform = trans[vi].copy()
        data = scene.save_image(resolution=(640, 480))
        image = np.array(Image.open(io.BytesIO(data)))[...,:-1]
        return image


    # if(epoch%100==0):
    #     v = trimesh.Trimesh(vertices=vertex_gt.detach().numpy()[0], faces=mano_right.faces, vertex_colors=vertex_colors)
    #     scene = trimesh.Scene(v)
    #     scene.camera_transform = trans[0]
    #     print('before scene.camera', scene.camera)
    #     print(scene.camera_transform)
    #     scene.show()
    #     print('scene.camera',scene.camera)
    #     print(scene.camera_transform)

    imgs=[]
    for vi in range(4):
        #print(vi)
        imgs.append(getview(vi).copy())

    #v.show()
    imgs=np.hstack(imgs)
    out.write(imgs)
    #cv2.imshow('frame', imgs)
    #print(imgs.shape)
    #cv2.waitKey(1)

    bioloss,eucbio=biolayer(joint_gt,torch.tensor([1],dtype=torch.float32))
    bioloss*=1000
    optimizer.zero_grad()
    bioloss.backward(retain_graph=True)
    optimizer.step()
    print(epoch,float(bioloss))

out.release()
cv2.destroyAllWindows()