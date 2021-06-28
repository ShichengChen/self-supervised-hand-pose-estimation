import argparse

parser = argparse.ArgumentParser(description='hyperparameters.')
parser.add_argument('-c','--card', type=int, default=0,
                    help='nvidia smi idx')
parser.add_argument('-b','--bio', action='store_true',
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
#print(args.card,args.bio,args.flex)

import os
import platform
if (platform.node()=='csc-G7-7590'):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.card)

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

#if not os.path.exists('/mnt/data/shicheng/RHD_published_v2/'):

encoderRGB=encoderRGB().cuda()
decoderPose=decPoseNet().cuda()
lr=1e-4
summary="lr:"+str(lr)+" "+__file__+" "+str(args.card)+" usebio:"+str(args.bio)+\
        " useflex:"+str(args.flex)+" checkpoint:"+str(args.checkpoint)+" comment:"+args.comment
print('summary',summary)
losshelp=LossHelper(useBar=True,usetb=args.tensorboard,summary=summary)

checkpointpath='./pretrain/'+platform.node()+'rgb2poseSyn7.pt'
if os.path.exists(checkpointpath) and args.checkpoint:
    checkpoint = torch.load(checkpointpath)
    print("use check point", checkpoint['epoch'])
    decoderPose.load_state_dict(checkpoint['decoderPose'])
    encoderRGB.load_state_dict(checkpoint['encoderRGB'])
onlysyn=False

train_dataset1 = ManoSynthesizer()
train_dataset2 = MVDataloader()
if(onlysyn):train_dataset = torch.utils.data.ConcatDataset( [train_dataset1, train_dataset1])
else:train_dataset = torch.utils.data.ConcatDataset( [train_dataset1, train_dataset2])
def _init_fn(worker_id):np.random.seed(worker_id)
def _init_fn2(worker_id):np.random.seed(worker_id**2+2)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,num_workers=4, shuffle=True,worker_init_fn=_init_fn)
print('train_loader',len(train_loader))
mano_right = MANO_SMPL(manoPath, ncomps=45, oriorder=True,device='cuda',userotJoints=True)

mylist=[]
mylist.append({'params':encoderRGB.parameters()})
mylist.append({'params':decoderPose.parameters()})
optimizer = torch.optim.Adam(mylist, lr=lr)
scheduler = MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)

def getLatentLoss(z_mean, z_stddev, goalStd=1.0, eps=1e-9):
    latent_loss = 0.5 * torch.sum(z_mean**2 + z_stddev**2 - torch.log(z_stddev**2)  - goalStd, 1)
    return latent_loss
#biolayer = BiomechanicalLayer(fingerPlaneLoss=True,fingerFlexLoss=True, fingerAbductionLoss=True)
biolayer = BiomechanicalLayer(fingerPlaneLoss=True, fingerAbductionLoss=True,fingerFlexLoss=args.flex)





from tqdm import tqdm

for epoch in tqdm(range(80)):
    losshelp.initForEachEpoch(len(train_loader),summaryVariable=['epe','loss'])
    for idx, inp in enumerate(train_loader):
        img,cloud,pose_gt,scale,root,mask=inp['img'].cuda(),inp['cloud'].cuda(),inp['pose3d'].cuda(),inp['scale'].cuda(),\
                                          inp['root'].cuda(),inp['mask'].cuda().reshape(-1)
        N = img.shape[0]
        if(torch.sum(mask)==N or torch.sum(mask)==0):continue
        # img, cloud, pose_gt, scale, root, mask = inp['img'], inp['cloud'], inp['pose3d'], inp[
        #     'scale'],inp['root'], inp['mask'].reshape(-1)

        # cur=img[0].permute(1,2,0).cpu().numpy()
        # cur=(cur*255).astype(np.uint8)
        # cv2.imshow('cur',cur)
        # cv2.waitKey(1)
        encoderRGB.train()
        decoderPose.train()

        z_rgb, mn_rgb, sd_rgb = encoderRGB(img,training=False)  # encode rgb
        # print('img',img)
        # z_rgb = encoderRGB(img,training=False)  # encode rgb
        # print("z_rgb",z_rgb)
        pose_rgb = decoderPose(z_rgb,).reshape(N,21,3)
        latent_loss_rgb = getLatentLoss(mn_rgb, sd_rgb)
        latent_loss_sum = torch.mean(latent_loss_rgb)

        pose_loss_syn,eucLoss_syn=pose3d_loss(pose_rgb[mask==1],pose_gt[mask==1],scale[mask==1])
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

        # if (platform.node() == 'csc-G7-7590'):
        #     for i in range(N):
        #         if (mask[i] == 0):
        #             id = int(inp['idx'][i])
        #             image = np.ones([480, 640]) * 2000
        #             image2 = np.ones([480, 640]) * 2000
        #             v = (vertexPre[i] * scale[i] + root[i]).detach().cpu().numpy() * 1000
        #             v2 = (cloud[i] * scale[i] + root[i]).detach().cpu().numpy() * 1000
        #             # import trimesh
        #             # tmesh = trimesh.Trimesh(v,mano_right.faces)
        #             # tmesh.show()
        #             vertex_uvd = perspective_projection(v, train_dataset2.demo.camera[id % 4]).astype(int)
        #             vertex_uvd2 = perspective_projection(v2, train_dataset2.demo.camera[id % 4]).astype(int)
        #             for i in range(vertex_uvd.shape[0]):
        #                 c = 3
        #                 u0 = np.clip(vertex_uvd[i, 0] - c, 0, 640)
        #                 u1 = np.clip(vertex_uvd[i, 0] + c, 0, 640)
        #                 v0 = np.clip(vertex_uvd[i, 1] - c, 0, 480)
        #                 v1 = np.clip(vertex_uvd[i, 1] + c, 0, 480)
        #                 image[v0:v1, u0:u1] = np.minimum(image[v0:v1, u0:u1], vertex_uvd[i, 2])
        #                 u0 = np.clip(vertex_uvd2[i, 0] - c, 0, 640)
        #                 u1 = np.clip(vertex_uvd2[i, 0] + c, 0, 640)
        #                 v0 = np.clip(vertex_uvd2[i, 1] - c, 0, 480)
        #                 v1 = np.clip(vertex_uvd2[i, 1] + c, 0, 480)
        #                 image2[v0:v1, u0:u1] = np.minimum(image2[v0:v1, u0:u1], vertex_uvd2[i, 2])
        #             image = 255 - visualize_better_qulity_depth_map(image)
        #             image2 = 255 - visualize_better_qulity_depth_map(image2)
        #             cv2.imshow("dep", image)
        #             cv2.imshow("dep2", image2)
        #             cv2.waitKey(1)
        #             break



        #print(pose_rgb.shape,joint_pre.shape)
        #joint_pre = joint_pre - joint_pre[:, :1].clone() + pose_rgb[:, :1].clone()

        synBioLoss,synBioEudloss=biolayer(joint_pre[mask==1],scale.reshape(N)[mask==1])
        realBioLoss,realBioEudloss=biolayer(joint_pre[mask!=1],scale.reshape(N)[mask!=1])
        synpose_loss_bone, syneucLoss_bone = pose3d_loss(pose_rgb[mask==1], joint_pre[mask==1], scale[mask==1])
        realpose_loss_bone, realeucLoss_bone = pose3d_loss(pose_rgb[mask!=1], joint_pre[mask!=1], scale[mask!=1])
        syncdloss, syncdlossEud = cloud_dis2(vertexPre[mask == 1], cloud[mask == 1], scale[mask == 1])
        realcdloss, realcdlossEud = cloud_dis2(vertexPre[mask != 1], cloud[mask != 1], scale[mask != 1])

        loss = 0.0001 * latent_loss_sum + pose_loss_syn + \
               synpose_loss_bone + realpose_loss_bone + \
               synBioLoss +\
               realcdloss + syncdloss#\
        if(args.bio):loss+=realBioLoss
               # +pose_loss_real#for test
        # loss = 0.0001*latent_loss_sum + cloudRec_loss_sum
        dicloss = {'epoch':int(epoch),'iter':int(idx),
                 "loss":float(loss),"_epe":float(eucLoss_syn)*1000,"_lepe":float(pose_loss_syn),
                 "_dcd":float(syncdlossEud)*1000,"_lcd":float(syncdloss),
                 "_dbio":float(synBioEudloss)*1000,"_lbio":float(synBioLoss),
                 "_dbone":float(syneucLoss_bone)*1000,"_lbone":float(synpose_loss_bone),
                 "_dgmatch":float(Greedymatchloss)*1000,
                   "epe": float(eucLoss_real) * 1000, "lepe": float(pose_loss_real),
                   "dcd": float(realcdlossEud) * 1000, "lcd": float(realcdloss),
                   "dbio": float(realBioEudloss) * 1000, "lbio": float(realBioLoss),
                   "dbone": float(realeucLoss_bone) * 1000, "lbone": float(realpose_loss_bone),
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

    savepath='./models/'+platform.node()+'rgb2pose.pt'
    print('save model',savepath)
    torch.save({
        'epoch': epoch,
        'encoderRGB': encoderRGB.state_dict(),
        'decoderPose': decoderPose.state_dict(),
        'optimizer': optimizer.state_dict()}, savepath)

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


