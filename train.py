import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
import torch
from cscPy.Nets.AligningHandnet import encoderRGB,decPoseNet
from cscPy.dataloader.MVdataloader import MVDataloader
from cscPy.mano.dataloader.synthesizer import ManoSynthesizer
from cscPy.Loss.utils import pose_loss,LossHelper
import numpy as np
from cscPy.mano.network.manoArmLayer import MANO_SMPL
from cscPy.mano.network.biomechanicalLoss import BiomechanicalLayer
import os
import pptk
from cscPy.mano.network.utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#if not os.path.exists('/mnt/data/shicheng/RHD_published_v2/'):

encoderRGB=encoderRGB().cuda()
decoderPose=decPoseNet().cuda()

onlysyn=True

train_dataset1 = ManoSynthesizer()
train_dataset2 = MVDataloader()
if(onlysyn):
    train_dataset = torch.utils.data.ConcatDataset( [train_dataset1, train_dataset1])
else:train_dataset = torch.utils.data.ConcatDataset( [train_dataset1, train_dataset2])
def _init_fn(worker_id):np.random.seed(worker_id)
def _init_fn2(worker_id):np.random.seed(worker_id**2+2)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,num_workers=4, shuffle=True,worker_init_fn=_init_fn)
print('train_loader',len(train_loader))

mano_right = MANO_SMPL('/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl', ncomps=45, oriorder=True,
                           device='cuda',userotJoints=True)

mylist=[]
mylist.append({'params':encoderRGB.parameters()})
mylist.append({'params':decoderPose.parameters()})
optimizer = torch.optim.Adam(mylist, lr=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)

def getLatentLoss(z_mean, z_stddev, goalStd=1.0, eps=1e-9):
    latent_loss = 0.5 * torch.sum(z_mean**2 + z_stddev**2 - torch.log(z_stddev**2)  - goalStd, 1)
    return latent_loss
biolayer = BiomechanicalLayer(fingerPlaneLoss=True,fingerFlexLoss=True, fingerAbductionLoss=True)
losshelp=LossHelper()


from chamfer_distance import ChamferDistance

chamfer_dist = ChamferDistance()
def cloud_dis(c0, c1,scale):
    dist1, dist2 = chamfer_dist(c0, c1)
    dis0 = torch.mean(dist1, dim=1)
    dis1 = torch.mean(dist2, dim=1)
    cdlossEud = torch.mean(dis0 * scale.reshape(N, 1)) + torch.mean(dis1 * scale.reshape(N, 1))
    cdloss = torch.mean(dis0) + torch.mean(dis1)
    return cdloss,cdlossEud

def cloud_dis2(c0, c1,scale):
    N=c0.shape[0]
    c0=c0.reshape(N,906,1,3)
    c1=c1.reshape(N,1,906,3)
    dis=torch.sqrt(torch.sum((c0-c1)**2,dim=3)).reshape(N,906,906)
    #print(torch.min(dis,dim=1))
    dis0=torch.min(dis,dim=1)[0]
    dis1=torch.min(dis,dim=2)[0]
    cdlossEud=torch.mean(dis0*scale.reshape(N,1))+torch.mean(dis1*scale.reshape(N,1))
    cdloss=torch.mean(dis0)+torch.mean(dis1)
    return cdloss,cdlossEud



for epoch in range(80):

    for idx, inp in enumerate(train_loader):
        img,cloud,pose_gt,scale,root,mask=inp['img'].cuda(),inp['cloud'].cuda(),inp['pose3d'].cuda(),inp['scale'].cuda(),\
                                          inp['root'].cuda(),inp['mask'].cuda().reshape(-1)
        # img, cloud, pose_gt, scale, root, mask = inp['img'], inp['cloud'], inp['pose3d'], inp[
        #     'scale'],inp['root'], inp['mask'].reshape(-1)
        encoderRGB.train()
        decoderPose.train()
        N=img.shape[0]
        z_rgb, mn_rgb, sd_rgb = encoderRGB(img,training=False)  # encode rgb
        pose_rgb = decoderPose(z_rgb,).reshape(N,21,3)
        latent_loss_rgb = getLatentLoss(mn_rgb, sd_rgb)
        latent_loss_sum = torch.mean(latent_loss_rgb)

        pose_loss_syn,eucLoss_syn=pose_loss(pose_rgb[mask==1],pose_gt[mask==1],scale[mask==1])
        pose_loss_real,eucLoss_real=pose_loss(pose_rgb[mask!=1],pose_gt[mask!=1],scale[mask!=1])
        pose_loss_sum = torch.mean(pose_loss_syn)
        #pose_loss_sum = torch.mean(eucLoss_syn)

        joints=pose_rgb*scale+root
        # boneLen = getBoneLen(joints)
        # manotempJ = getTemplateFrom(boneLen, mano_right.J)
        # manotempJ = get32fTensor(manotempJ)
        wrist_trans,local_trans,outjoints=mano_right.matchTemplate2JointsGreedy(joints)
        _, Greedymatchloss = pose_loss(joints, outjoints, scale)
        #print('Greedymatchloss',Greedymatchloss)

        vertexPre, joint_pre = mano_right.get_mano_vertices(wrist_trans.reshape([N, 1, 3, 3]),
                                                            local_trans.view(N, 15, 3, 3),
                                                            torch.zeros([N,10]).view(N, 10),
                                                            torch.ones([N,1]).view(N, 1),
                                                            torch.tensor([[0, 0, 0]]).view(1, 3).repeat(N,1),
                                                            pose_type='rot_matrix', mmcp_center=False,
                                                            external_transition=None)
        mmcp=joint_pre[:,4:5,:].clone()
        joint_pre=(joint_pre-mmcp)/scale
        vertexPre=(vertexPre-mmcp)/scale
        #print(pose_rgb.shape,joint_pre.shape)
        #joint_pre = joint_pre - joint_pre[:, :1].clone() + pose_rgb[:, :1].clone()

        bioLoss,bioEudloss=biolayer(joint_pre,scale.reshape(N))
        pose_loss_bone, eucLoss_bone = pose_loss(pose_rgb, joint_pre, scale)
        pose_loss_bone = torch.mean(pose_loss_bone)


        #print(vertexPre)
        #print(cloud)
        cdloss,cdlossEud=cloud_dis(vertexPre,cloud,scale)

        #loss = 0.0001*latent_loss_sum + pose_loss_sum#+cdloss*1e-5+bioLoss*1e-5+pose_loss_bone
        loss = pose_loss_sum
        #loss = 0.0001*latent_loss_sum + cloudRec_loss_sum
        losshelp.add("loss",float(loss))
        losshelp.add("epeSyn",float(eucLoss_syn)*1000)
        losshelp.add("epeReal",float(eucLoss_real)*1000)
        losshelp.add("cdloss",float(cdlossEud)*1000)
        losshelp.add("bioLoss", float(bioEudloss)*1000)#already scaled to milimeter
        losshelp.add("eucLoss_bone",float(eucLoss_bone)*1000)
        losshelp.add("Greedymatchloss",float(Greedymatchloss)*1000)

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

    # print('save model')
    # torch.save({
    #     'epoch': epoch,
    #     'encoderRGB': encoderRGB.state_dict(),
    #     'decoderPose': decoderPose.state_dict(),
    #     'optimizer': optimizer.state_dict()}, './model/imagePoseTry.pt')

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


