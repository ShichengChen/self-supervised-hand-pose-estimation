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

import cv2
from cscPy.mano.network.utils import *

encoderRGB=encoderRGB().cuda()
decoderPose=decPoseNet().cuda()

# checkpoint = torch.load('./pretrain/rgb2pose.pt')
# decoderPose.load_state_dict(checkpoint['decoderCloud'])
# encoderRGB.load_state_dict(checkpoint['encoderRGB'])
onlysyn=True

train_dataset1 = ManoSynthesizer()
train_dataset2 = MVDataloader()
train_dataset = torch.utils.data.ConcatDataset( [train_dataset2])
def _init_fn(worker_id):np.random.seed(worker_id)
def _init_fn2(worker_id):np.random.seed(worker_id**2+2)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,num_workers=4, shuffle=True,worker_init_fn=_init_fn)
print('train_loader',len(train_loader))
manoPath='/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
if not os.path.exists(manoPath):
    manoPath = '/home/shicheng/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
mano_right = MANO_SMPL(manoPath, ncomps=45, oriorder=True,device='cuda',userotJoints=True)

mylist=[]
mylist.append({'params':encoderRGB.parameters()})
mylist.append({'params':decoderPose.parameters()})
optimizer = torch.optim.Adam(mylist, lr=1e-3)
scheduler = MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)

def getLatentLoss(z_mean, z_stddev, goalStd=1.0, eps=1e-9):
    latent_loss = 0.5 * torch.sum(z_mean**2 + z_stddev**2 - torch.log(z_stddev**2)  - goalStd, 1)
    return latent_loss
biolayer = BiomechanicalLayer(fingerPlaneLoss=True,fingerFlexLoss=True, fingerAbductionLoss=True)
losshelp=LossHelper()





for epoch in range(80):

    for idx, inp in enumerate(train_loader):
        img,cloud,pose_gt,scale,root,mask=inp['img'].cuda(),inp['cloud'].cuda(),inp['pose3d'].cuda(),inp['scale'].cuda(),\
                                          inp['root'].cuda(),inp['mask'].cuda().reshape(-1)
        # img, cloud, pose_gt, scale, root, mask = inp['img'], inp['cloud'], inp['pose3d'], inp[
        #     'scale'],inp['root'], inp['mask'].reshape(-1)

        # cur=img[0].permute(1,2,0).cpu().numpy()
        # cur=(cur*255).astype(np.uint8)
        # cv2.imshow('cur',cur)
        # cv2.waitKey(1)
        encoderRGB.train()
        decoderPose.train()
        N=img.shape[0]
        z_rgb, mn_rgb, sd_rgb = encoderRGB(img,training=False)  # encode rgb
        # print('img',img)
        # z_rgb = encoderRGB(img,training=False)  # encode rgb
        # print("z_rgb",z_rgb)
        pose_rgb = decoderPose(z_rgb,).reshape(N,21,3)
        latent_loss_rgb = getLatentLoss(mn_rgb, sd_rgb)
        latent_loss_sum = torch.mean(latent_loss_rgb)
        pose_loss_real,eucLoss_real=pose3d_loss(pose_rgb[mask!=1],pose_gt[mask!=1],scale[mask!=1])

        joints=pose_rgb*scale+root
        wrist_trans,local_trans,outjoints=mano_right.matchTemplate2JointsGreedy(joints)
        _, Greedymatchloss = pose3d_loss(joints, outjoints, scale)

        vertexPre, joint_pre = mano_right.get_mano_vertices(wrist_trans.reshape([N, 1, 3, 3]),
                                                            local_trans.view(N, 15, 3, 3),
                                                            torch.zeros([N, 10], dtype=torch.float32).view(N, 10),
                                                            torch.ones([N, 1], dtype=torch.float32).view(N, 1),
                                                            torch.zeros([N,3], dtype=torch.float32).view(N,3),
                                                            pose_type='rot_matrix', mmcp_center=False,
                                                            external_transition=None)

        mmcp = joint_pre[:, 4:5, :].clone()
        joint_pre = (joint_pre - mmcp) / scale
        vertexPre = (vertexPre - mmcp) / scale


        realBioLoss,realBioEudloss=biolayer(joint_pre[mask!=1],scale.reshape(N)[mask!=1])
        realpose_loss_bone, realeucLoss_bone = pose3d_loss(pose_rgb[mask!=1], joint_pre[mask!=1], scale[mask!=1])
        realcdloss, realcdlossEud = cloud_dis2(vertexPre[mask != 1], cloud[mask != 1], scale[mask != 1])

        loss = 0.0001*latent_loss_sum + pose_loss_real+\
               realpose_loss_bone+\
               realBioLoss+\
               realcdloss
        #loss = 0.0001*latent_loss_sum + cloudRec_loss_sum
        dicloss = {"loss": float(loss),
                   "Greedymatch dis": float(Greedymatchloss) * 1000,
                   "real epe": float(eucLoss_real) * 1000, "real loss": float(pose_loss_real),
                   "real cd dis": float(realcdlossEud) * 1000, "real cd loss": float(realcdloss),
                   "real bio dis": float(realBioEudloss) * 1000, "real bio loss": float(realBioLoss),
                   "real bone dis": float(realeucLoss_bone) * 1000, "real bone loss": float(realpose_loss_bone),
                   }
        losshelp.add(dicloss)

        if(idx%5==0):
            print('epoch:{} iteration:{}'.format(epoch,idx))
            losshelp.showcurrent()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print('scheduler.step()')
    # scheduler.step()
    losshelp.show()
    losshelp.clear()

    print('save model')
    torch.save({
        'epoch': epoch,
        'encoderRGB': encoderRGB.state_dict(),
        'decoderPose': decoderPose.state_dict(),
        'optimizer': optimizer.state_dict()}, './models/' + platform.node() + 'rgb2poseReal.pt')

    #print(epoch, 'epoch mean epeloss', np.mean(epe)*1000)
    # aveloss,epe=[],[]
    # #aveCD,aveEMD=[],[]
    # with torch.no_grad():
    #     for idx,(image, depth, cloud, heatmap, pose_gt, viewRotation, scale) in enumerate(test_loader):
    #         # image, depth, cloud, heatmap, pose_gt, viewRotation, scale = image.cuda(), depth.cuda(), cloud.cuda(), heatmap.cuda(), \
    #         #                                                              pose_gt.cuda(), viewRotation.cuda(), scale.cuda()
    #         pose_gt, scale, image =pose_gt.cuda(), scale.cuda(), image.cuda()
    #         encoderRGB.eval()
    #         decoderPose.eval()
    #         z_rgb, mn_rgb, sd_rgb = encoderRGB(image,training=False)  # encode rgb
    #         pose_rgb = decoderPose(z_rgb, )
    #         pose_loss_rgb, eucLoss_rgb = utils.pose3d_loss(pose_rgb, pose_gt, scale)
    #         epe.append(float(eucLoss_rgb))
    #
    #     print('len(epe)',len(epe[:len(epe)//2]))
    #     print(epoch, 'epoch test mean rgb epeloss', np.mean(epe[:len(epe)//2])*1000)


