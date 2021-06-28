import argparse

parser = argparse.ArgumentParser(description='hyperparameters.')
parser.add_argument('-c','--card', type=int, default=0,
                    help='nvidia smi idx')
parser.add_argument('-ab','--abduction', action='store_true',
                    help='use bio constraint or not')
parser.add_argument('-f','--flex',action='store_true',
                    help='use flex loss of bio constraint')
parser.add_argument('-ck','--checkpoint',action='store_true',
                    help='use checkpoint or not')
parser.add_argument('-co','--comment',type=str,default="",
                    help='use checkpoint or not')
parser.add_argument('-t','--tensorboard',action='store_true',
                    help='use tensorboard or not')
args = parser.parse_args()

import os
import platform
if (platform.node()=='csc-G7-7590'):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.card

import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
import torch
from cscPy.Nets.AligningHandnet import encoderRGB,decPoseNet
from cscPy.dataloader.MVdataloader import MVDataloader
from cscPy.mano.dataloader.synthesizer import ManoSynthesizer
from cscPy.Loss.utils import pose3d_loss,LossHelper,cloud_dis,cloud_dis2,pose2d_loss
import numpy as np
from cscPy.mano.network.manoArmLayer import MANO_SMPL
from cscPy.mano.network.biomechanicalLoss import BiomechanicalLayer
from cscPy.globalCamera.camera import CameraIntrinsics,perspective_projection
from cscPy.globalCamera.util import fetch_all_sequences,load_rgb_maps,load_depth_maps,get_cameras_from_dir,visualize_better_qulity_depth_map


import cv2
from cscPy.mano.network.utils import *

#if not os.path.exists('/mnt/data/shicheng/RHD_published_v2/'):

encoderRGB=encoderRGB().cuda()
decoderPose=decPoseNet().cuda()
lr=1e-4
summary="lr:"+str(lr)+" "+__file__+" "+str(args.card)+" useAB:"+str(args.abduction)+\
        " useflex:"+str(args.flex)+" checkpoint:"+str(args.checkpoint)+" comment:"+args.comment
print('summary',summary)
losshelp=LossHelper(useBar=True,usetb=args.tensorboard,summary=summary)

# checkpointpath='./pretrain/'+platform.node()+'rgb2poseSyn.pt'
# if os.path.exists(checkpointpath):
#     checkpoint = torch.load(checkpointpath)
#     print("use check point", checkpoint['epoch'])
#     decoderPose.load_state_dict(checkpoint['decoderPose'])
#     encoderRGB.load_state_dict(checkpoint['encoderRGB'])

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
optimizer = torch.optim.Adam(mylist, lr=lr)
scheduler = MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)

def getLatentLoss(z_mean, z_stddev, goalStd=1.0, eps=1e-9):
    latent_loss = 0.5 * torch.sum(z_mean**2 + z_stddev**2 - torch.log(z_stddev**2)  - goalStd, 1)
    return latent_loss

losshelp=LossHelper()






for epoch in range(80):
    #torch.autograd.set_detect_anomaly(True)
    for idx, inp in enumerate(train_loader):
        img,cloud,pose_gt,scale,root,mask=inp['img'].cuda(),inp['cloud'].cuda(),inp['pose3d'].cuda(),inp['scale'].cuda(),\
                                          inp['root'].cuda(),inp['mask'].cuda().reshape(-1)
        K,pose2d=inp['K'].cuda(),inp['pose2d'].cuda()
        N = img.shape[0]

        encoderRGB.train()
        decoderPose.train()

        z_rgb, mn_rgb, sd_rgb = encoderRGB(img,training=False)  # encode rgb
        pose_rgb = decoderPose(z_rgb,).reshape(N,21,3)
        latent_loss_rgb = getLatentLoss(mn_rgb, sd_rgb)
        latent_loss_sum = torch.mean(latent_loss_rgb)

        pose3dloss,eucLoss=pose3d_loss(pose_rgb,pose_gt,scale)
        joints=pose_rgb*scale+root

        pose2dpre=perspectiveProjection(joints,K)[...,:-1]
        #print(pose2dpre[0,0],pose2d[0,0])
        pose2dloss,pixelLoss=pose2d_loss(pose2dpre,pose2d)

        wrist_trans,local_trans,outjoints, bioLoss = mano_right.matchTemplate2JointsWithConstraint(joints)
        Greedymatchdis, Greedymatchloss = pose3d_loss(joints, outjoints, scale)



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


        poseloss_bone, eucLoss_bone = pose3d_loss(pose_rgb, joint_pre, scale)


        loss = 0.0001 * latent_loss_sum + pose2dloss*0.001+Greedymatchloss+poseloss_bone+bioLoss

        dicloss = {"loss": float(loss), "epe": float(eucLoss) * 1000, "lepe": float(pose3dloss),
                   "lflex": float(bioLoss['flexloss']), "abduction": float(bioLoss),
                   "dbone": float(eucLoss_bone) * 1000, "lbone": float(poseloss_bone),
                   "dpixel": float(pixelLoss), "lpixel": float(pose2dloss)*0.001,
                   "dGmatch": float(Greedymatchdis) * 1000,
                   }
        losshelp.add(dicloss)

        if(idx%5==0):
            print('epoch:{} iteration:{}'.format(epoch,idx))
            losshelp.showcurrent()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print('scheduler.step()')
    scheduler.step()
    losshelp.finish()

    modelpath='./models/'+platform.node()+'rgb2pose2d.pt'
    print('save model',modelpath)
    torch.save({
        'epoch': epoch,
        'encoderRGB': encoderRGB.state_dict(),
        'decoderPose': decoderPose.state_dict(),
        'optimizer': optimizer.state_dict()}, modelpath)

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
    #         pose_loss_rgb, eucLoss_rgb = utils.pose_loss(pose_rgb, pose_gt, scale)
    #         epe.append(float(eucLoss_rgb))
    #
    #     print('len(epe)',len(epe[:len(epe)//2]))
    #     print(epoch, 'epoch test mean rgb epeloss', np.mean(epe[:len(epe)//2])*1000)


