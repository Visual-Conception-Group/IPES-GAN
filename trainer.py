from networks import AdaINGen, MsImageDis
from backbone_network import ft_net, ft_netAB, PCB
from utils import get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import copy
import os
import cv2
import numpy as np
from random_erasing import RandomErasing
import random
import yaml
import pdb
from loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy

#fp16
try:
    from apex import amp
    from apex.fp16_utils import *
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


def to_gray(half=False): #simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward

def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        xx = cv2.Canny(xx, 10, 200) #256x128
        xx = xx/255.0 - 0.5 # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i,:,:] = xx
    out = out.unsqueeze(1) 
    return out.cuda()

def scale2(x):
    if x.size(2) > 128: # do not need to scale the input
        return x
    x = torch.nn.functional.upsample(x, scale_factor=2, mode='nearest')  #bicubic is not available for the time being.
    return x

def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp

def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def update_CNN(model_s, model_t, alpha=0.999):
    for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param_s.data)

def predict_label(CNN_models, inputs, num_class, CNN_style=0):
# CNN_style:
# 0: smooth dynamic label
# 1: Pseudo label, hard dynamic label
# 2: Conditional label, hard static label 
# 3: LSRO, static smooth label
# 4: Dynamic Soft Two-label
# alabel is appearance label
    if CNN_style == 0:
        count = 0
        sm = nn.Softmax(dim=1)
        for CNN_model in CNN_models:
            _, outputs_t1 = CNN_model(inputs) 
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = CNN_model(fliplr(inputs)) 
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
    elif CNN_style == 1:  # dynamic one-hot  label
        count = 0
        sm = nn.Softmax(dim=1)
        for CNN_model in CNN_models:
            _, outputs_t1 = CNN_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())  # change softmax to max
            _, outputs_t2 = CNN_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
        _, dlabel = torch.max(outputs_t.data, 1)
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, dlabel[i]] = 1
    elif CNN_style == 2: # appearance label
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, alabel[i]] = 1
    elif CNN_style == 3: # LSRO
        outputs_t = torch.ones(inputs.size(0), num_class).cuda()
    elif CNN_style == 4: #Two-label
        count = 0
        sm = nn.Softmax(dim=1)
        for CNN_model in CNN_models:
            _, outputs_t1 = CNN_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = CNN_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
        mask = torch.zeros(outputs_t.shape)
        mask = mask.cuda()
        for i in range(inputs.size(0)):
            mask[i, alabel[i]] = 1
            mask[i, slabel[i]] = 1
        outputs_t = outputs_t*mask
    else:
        print('not valid style. CNN-style is in [0-3].')

    s = torch.sum(outputs_t, dim=1, keepdim=True)
    s = s.expand_as(outputs_t)
    outputs_t = outputs_t/s
    return outputs_t

######################################################################
# Load model
#---------------------------
def load_network(network, name):
    save_path = os.path.join('./models/best/net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_config(name):
    config_path = os.path.join('./models',name,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    return config


class IPES_Trainer(nn.Module):
    def __init__(self, hyperparameters, gpu_ids=[0]):
        super(IPES_Trainer, self).__init__()
        lr_g = hyperparameters['lr_g']
        lr_d = hyperparameters['lr_d']
        ID_class = hyperparameters['ID_class']
        if not 'apex' in hyperparameters.keys():
            hyperparameters['apex'] = False
        self.fp16 = hyperparameters['apex']
        # Initiate the networks
        # We do not need to manually set fp16 in the network for the new apex. So here I set fp16=False.
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'], fp16 = False)  # auto-encoder for domain a
        self.gen_b = self.gen_a  # auto-encoder for domain b
      

        if not 'ID_stride' in hyperparameters.keys():
            hyperparameters['ID_stride'] = 2
        if hyperparameters['ID_style']=='PCB':
            self.id_a = PCB(ID_class)
        elif hyperparameters['ID_style']=='AB':
            self.id_a = ft_netAB(ID_class, stride = hyperparameters['ID_stride'], norm=hyperparameters['norm_id'], pool=hyperparameters['pool']) 
        else:
            self.id_a = ft_net(ID_class, norm=hyperparameters['norm_id'], pool=hyperparameters['pool']) # return 2048 now

        self.id_b = self.id_a
        self.dis_a = MsImageDis(3, hyperparameters['dis'], fp16 = False)  # discriminator for domain a
        self.dis_b = self.dis_a # discriminator for domain b
        pdb.set_trace()
        # load CNN
        if hyperparameters['CNN'] != "":
            CNN_name = hyperparameters['CNN']
            print(CNN_name)
            CNN_names = CNN_name.split(',')
            CNN_model = nn.ModuleList()
            CNN_count = 0
            for CNN_name in CNN_names:
                config_tmp = load_config(CNN_name)
                if 'stride' in config_tmp:
                    stride = config_tmp['stride'] 
                else:
                    stride = 2
                model_tmp = ft_net(ID_class, stride = stride)
                CNN_model_tmp = load_network(model_tmp, CNN_name)
                CNN_model_tmp.model.fc = nn.Sequential()  # remove the original fc layer in ImageNet
                CNN_model_tmp = CNN_model_tmp.cuda()
                if self.fp16:
                    CNN_model_tmp = amp.initialize(CNN_model_tmp, opt_level="O1")
                CNN_model.append(CNN_model_tmp.cuda().eval())
                CNN_count +=1
            self.CNN_model = CNN_model
            if hyperparameters['train_bn']:
                self.CNN_model = self.CNN_model.apply(train_bn)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # RGB to one channel
        if hyperparameters['single']=='edge':
            self.single = to_edge
        else:
            self.single = to_gray(False)

        # Random Erasing when training
        if not 'erasing_p' in hyperparameters.keys():
            self.erasing_p = 0
        else:
            self.erasing_p = hyperparameters['erasing_p']
        self.single_re = RandomErasing(probability = self.erasing_p, mean=[0.0, 0.0, 0.0])

        if not 'T_w' in hyperparameters.keys():
            hyperparameters['T_w'] = 1

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) #+ list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) #+ list(self.gen_b.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # id params
        if hyperparameters['ID_style']=='PCB':
            ignored_params = (list(map(id, self.id_a.classifier0.parameters() ))
                            +list(map(id, self.id_a.classifier1.parameters() ))
                            +list(map(id, self.id_a.classifier2.parameters() ))
                            +list(map(id, self.id_a.classifier3.parameters() ))
                            )
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            self.id_opt = torch.optim.SGD([
                 {'params': base_params, 'lr': lr2},
                 {'params': self.id_a.classifier0.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier1.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier2.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier3.parameters(), 'lr': lr2*10}
            ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)
        elif hyperparameters['ID_style']=='AB':
            ignored_params = (list(map(id, self.id_a.classifier1.parameters()))
                            + list(map(id, self.id_a.classifier2.parameters())))
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            self.id_opt = torch.optim.SGD([
                 {'params': base_params, 'lr': lr2},
                 {'params': self.id_a.classifier1.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier2.parameters(), 'lr': lr2*10}
            ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, self.id_a.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            self.id_opt = torch.optim.SGD([
                 {'params': base_params, 'lr': lr2},
                 {'params': self.id_a.classifier.parameters(), 'lr': lr2*10}
            ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.id_scheduler = get_scheduler(self.id_opt, hyperparameters)
        self.id_scheduler.gamma = hyperparameters['gamma2']

        #ID Loss
        self.id_criterion = nn.CrossEntropyLoss()
        
        #self.criterion_tri = TripletLoss(margin=margin)
        
        self.criterion_CNN = nn.KLDivLoss(size_average=False)
        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        # save memory
        if self.fp16:
            # Name the FP16_Optimizer instance to replace the existing optimizer
            assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
            self.gen_a = self.gen_a.cuda()
            self.dis_a = self.dis_a.cuda()
            self.id_a = self.id_a.cuda()

            self.gen_b = self.gen_a
            self.dis_b = self.dis_a
            self.id_b = self.id_a

            self.gen_a, self.gen_opt = amp.initialize(self.gen_a, self.gen_opt, opt_level="O1")
            self.dis_a, self.dis_opt = amp.initialize(self.dis_a, self.dis_opt, opt_level="O1")
            self.id_a, self.id_opt = amp.initialize(self.id_a, self.id_opt, opt_level="O1")

    def to_re(self, x):
        out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
        out = out.cuda()
        for i in range(x.size(0)):
            out[i,:,:,:] = self.single_re(x[i,:,:,:])
        return out

    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))

    def recon_criterion_sqrt(self, input, target):
        diff = input - target
        return torch.mean(torch.sqrt(torch.abs(diff[:])+1e-8))
    
    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def recon_criterion2(self, input, target):
        diff = input - target
        return torch.mean(diff[:]**2)

    def recon_cos(self, input, target):
        cos = torch.nn.CosineSimilarity()
        cos_dis = 1 - cos(input, target)
        return torch.mean(cos_dis[:])
    
    #Label smooth Regularisation loss
    
    def _lsr_loss(self, outputs, targets):
        num_class = outputs.size()[1]
        targets = self._class_to_one_hot(targets.data.cpu(), num_class)
        targets = Variable(targets.cuda())
        outputs = torch.nn.LogSoftmax()(outputs)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def _class_to_one_hot(self, targets, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, 0.9)
        targets_onehot.add_(0.1 / num_class)
        return targets_onehot

    def forward(self,bone_a,bone_b, x_a, x_b, x_a1, x_b1, xp_a, xp_b):
        
        s_a = self.gen_a.encode((x_a))
        s_b = self.gen_b.encode((x_b))
        f_a, p_a = self.id_a(scale2(x_a1))
        f_b, p_b = self.id_b(scale2(x_b1))
        x_ba = self.gen_a.decode(s_b, f_a)
        x_ab = self.gen_b.decode(s_a, f_b)
        x_ba_pose1 = torch.cat([x_ba, bone_b], dim=1)
        x_ab_pose1 = torch.cat([x_ab, bone_a], dim=1)
        s_a_recon = self.gen_a.encode((x_ab_pose1))
        s_b_recon = self.gen_a.encode((x_ba_pose1))
        x_ba_recon = self.gen_a.decode(s_a_recon, f_a)
        x_ab_recon = self.gen_b.decode(s_b_recon, f_b)
        f_a_recon, p_a_recon = self.id_a(scale2(x_ba))
        f_b_recon, p_b_recon = self.id_b(scale2(x_ab))
        x_aba = self.gen_a.decode(s_a_recon, f_a_recon) 
        x_bab = self.gen_b.decode(s_b_recon, f_b_recon)  
        x_a_recon = self.gen_a.decode(s_a, f_a)
        x_b_recon = self.gen_b.decode(s_b, f_b)
        fp_a, pp_a = self.id_a(scale2(xp_a))
        fp_b, pp_b = self.id_b(scale2(xp_b))
        # decode the same person
        x_a_recon_p = self.gen_a.decode(s_a, fp_a)
        x_b_recon_p = self.gen_b.decode(s_b, fp_b)

        # Random Erasing only effect the ID and PID loss.
        if self.erasing_p > 0:
            x_a_re = self.to_re(scale2(x_a1.clone()))
            x_b_re = self.to_re(scale2(x_b1.clone()))
            xp_a_re = self.to_re(scale2(xp_a.clone()))
            xp_b_re = self.to_re(scale2(xp_b.clone()))
            _, p_a = self.id_a(x_a_re)
            _, p_b = self.id_b(x_b_re)
            # encode the same ID different photo
            _, pp_a = self.id_a(xp_a_re) 
            _, pp_b = self.id_b(xp_b_re)

        return x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_ab_recon, x_ba_recon

    def gen_update(self,bone_a,bone_b,x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b, xp_a, xp_b, l_a, hyperparameters, iteration, num_gpu):
        # ppa, ppb is the same person
        self.gen_opt.zero_grad()
        self.id_opt.zero_grad()
 
        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)
        x_ba_pose = torch.cat([x_ba_copy, bone_b], dim=1)
        x_ab_pose = torch.cat([x_ab_copy, bone_a], dim=1)
        x_ba_pose1 = torch.cat([x_ba, bone_b], dim=1)
        x_ab_pose1 = torch.cat([x_ab, bone_a], dim=1)
        rand_num = random.uniform(0,1)
        #################################
        # encode structure
        if hyperparameters['use_encoder_again']>=rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = self.gen_b.enc_content((x_ab_pose))
            s_b_recon = self.gen_a.enc_content((x_ba_pose))
        else:
            # copy the encoder
            self.enc_content_copy = copy.deepcopy(self.gen_a.enc_content)
            self.enc_content_copy = self.enc_content_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_copy((x_ab_pose1))
            s_b_recon = self.enc_content_copy((x_ba_pose1))

        #################################
        # encode appearance
        self.id_a_copy = copy.deepcopy(self.id_a)
        self.id_a_copy = self.id_a_copy.eval()
        if hyperparameters['train_bn']:
            self.id_a_copy = self.id_a_copy.apply(train_bn)
        self.id_b_copy = self.id_a_copy
        # encode again (encoder is fixed, input is tuned)
        f_a_recon, p_a_recon = self.id_a_copy(scale2(x_ba))
        f_b_recon, p_b_recon = self.id_b_copy(scale2(x_ab))

        # KL Loss
        #  Tune the ID model
        log_sm = nn.LogSoftmax(dim=1)
        if hyperparameters['CNN_w'] >0 and hyperparameters['CNN'] != "":
            if hyperparameters['ID_style'] == 'normal':
                _, p_a_student = self.id_a(scale2(x_ba_copy))
                p_a_student = log_sm(p_a_student)
                p_a_CNN = predict_label(self.CNN_model, scale2(x_ba_copy), num_class = hyperparameters['ID_class'], alabel = l_a, slabel = l_b, CNN_style = hyperparameters['CNN_style'])
                self.loss_CNN = self.criterion_CNN(p_a_student, p_a_CNN) / p_a_student.size(0)

                _, p_b_student = self.id_b(scale2(x_ab_copy))
                p_b_student = log_sm(p_b_student)
                p_b_CNN = predict_label(self.CNN_model, scale2(x_ab_copy), num_class = hyperparameters['ID_class'], alabel = l_b, slabel = l_a, CNN_style = hyperparameters['CNN_style'])
                self.loss_CNN += self.criterion_CNN(p_b_student, p_b_CNN) / p_b_student.size(0)
            elif hyperparameters['ID_style'] == 'AB':
                # normal KL loss
                # BA -> LabelA(smooth) + LabelB(batchB)
                _, p_ba_student = self.id_a(scale2(x_ba_copy))# f_a, s_b
                p_a_student = log_sm(p_ba_student[0])
                with torch.no_grad():
                    p_a_CNN = predict_label(self.CNN_model, scale2(x_ba_copy), num_class = hyperparameters['ID_class'], CNN_style = hyperparameters['CNN_style'])
                self.loss_CNN = self.criterion_CNN(p_a_student, p_a_CNN) / p_a_student.size(0)
                #In this part we have duke appearance but we have used CNN model of market. Check whether its helpful or not
                _, p_ab_student = self.id_b(scale2(x_ab_copy)) # f_b, s_a
                p_b_student = log_sm(p_ab_student[0])
                with torch.no_grad():
                    p_b_CNN = predict_label(self.CNN_model, scale2(x_ab_copy), num_class = hyperparameters['ID_class'], CNN_style = hyperparameters['CNN_style'])
                self.loss_CNN += self.criterion_CNN(p_b_student, p_b_CNN) / p_b_student.size(0)

                # branch b loss
                # here we give different label
                loss_B =  self.id_criterion(p_ab_student[1], l_a)
                self.loss_CNN = hyperparameters['T_w'] * self.loss_CNN + hyperparameters['B_w'] * loss_B
        else:
            self.loss_CNN = 0.0

        # auto-encoder image reconstruction
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_xp_a = self.recon_criterion(x_a_recon_p, x_a)
        self.loss_gen_recon_xp_b = self.recon_criterion(x_b_recon_p, x_b)

        # feature reconstruction
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(f_a_recon, f_a) if hyperparameters['recon_f_w'] > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(f_b_recon, f_b) if hyperparameters['recon_f_w'] > 0 else 0

        x_aba = self.gen_a.decode(s_a_recon, f_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(s_b_recon, f_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # ID loss AND Tune the Generated image
        if hyperparameters['ID_style']=='PCB':
            self.loss_id = self.PCB_loss(p_a, l_a) + self.PCB_loss(p_b, l_b)
            self.loss_pid = self.PCB_loss(pp_a, l_a) + self.PCB_loss(pp_b, l_b)
            self.loss_gen_recon_id = self.PCB_loss(p_a_recon, l_a) + self.PCB_loss(p_b_recon, l_b)
            
            
            
            
            
        elif hyperparameters['ID_style']=='AB':
            weight_B = hyperparameters['CNN_w'] * hyperparameters['B_w']
            self.loss_id = self.id_criterion(p_a[0], l_a) \
                         + weight_B * ( self.id_criterion(p_a[1], l_a)  )
            self.loss_pid = self.id_criterion(pp_a[0], l_a)  + weight_B * ( self.id_criterion(pp_a[1], l_a)  )
            
            
            #self.loss_gen_recon_id = self.id_criterion(p_a_recon[0], l_a) 
            
            
            self.loss_gen_recon_id = self._lsr_loss(p_a_recon[0], l_a)
            
           # self.loss_tri = self.criterion_tri(f_a, l_a)
            
            
        else:
            self.loss_id = self.id_criterion(p_a, l_a) + self.id_criterion(p_b, l_b)
            self.loss_pid = self.id_criterion(pp_a, l_a) + self.id_criterion(pp_b, l_b)
            self.loss_gen_recon_id = self.id_criterion(p_a_recon, l_a) + self.id_criterion(p_b_recon, l_b)

        #print(f_a_recon, f_a)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        if num_gpu>1:
            self.loss_gen_adv_a = self.dis_a.module.calc_gen_loss(self.dis_a, x_ba)
            self.loss_gen_adv_b = self.dis_b.module.calc_gen_loss(self.dis_b, x_ab)
        else:
            self.loss_gen_adv_a = self.dis_a.calc_gen_loss(self.dis_a, x_ba) 
            self.loss_gen_adv_b = self.dis_b.calc_gen_loss(self.dis_b, x_ab) 
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
       
        self.loss_gen_recon_kl_a = self.__compute_kl(s_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(s_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(s_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(s_b_recon)
        
        if iteration > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])

        if iteration > hyperparameters['warm_CNN_iter']:
            hyperparameters['CNN_w'] += hyperparameters['warm_scale']
            hyperparameters['CNN_w'] = min(hyperparameters['CNN_w'], hyperparameters['max_CNN_w'])
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_a + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_b + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['id_w'] * self.loss_id + \
                              hyperparameters['pid_w'] * self.loss_pid + \
                              hyperparameters['recon_id_w'] * self.loss_gen_recon_id + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['CNN_w'] * self.loss_CNN
        if self.fp16:
            with amp.scale_loss(self.loss_gen_total, [self.gen_opt,self.id_opt]) as scaled_loss:
                scaled_loss.backward()
            self.gen_opt.step()
            self.id_opt.step()
        else:
            self.loss_gen_total.backward()
            self.gen_opt.step()
            self.id_opt.step()
        print("L_total: %.4f, L_gan: %.4f,  Lx: %.4f, Lxp: %.4f, Lrecycle:%.4f, Lf: %.4f, Ls: %.4f, Recon-id: %.4f, id: %.4f, pid:%.4f, CNN: %.4f"%( self.loss_gen_total, \
                                                        hyperparameters['gan_w'] * (self.loss_gen_adv_a + self.loss_gen_adv_b), \
                                                        hyperparameters['recon_x_w'] * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b), \
                                                        hyperparameters['recon_xp_w'] * (self.loss_gen_recon_xp_a + self.loss_gen_recon_xp_b), \
                                                        hyperparameters['recon_x_cyc_w'] * (self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b), \
                                                        hyperparameters['recon_f_w'] * (self.loss_gen_recon_f_a + self.loss_gen_recon_f_b), \
                                                        hyperparameters['recon_s_w'] * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b), \
                                                        hyperparameters['recon_id_w'] * self.loss_gen_recon_id, \
                                                        hyperparameters['id_w'] * self.loss_id,\
                                                        hyperparameters['pid_w'] * self.loss_pid,\
hyperparameters['CNN_w'] * self.loss_CNN )  )

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def PCB_loss(self, inputs, labels):
       loss = 0.0
       for part in inputs:
           loss += self.id_criterion(part, labels)
       return loss/len(inputs)

    def sample(self, x_a1,x_b1,x_a, x_b,mask_a,mask_b,image_a_b,image_b_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ab1, x_aba, x_bab = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            s_a = self.gen_a.encode(x_a1[i].unsqueeze(0)) 
            s_b = self.gen_b.encode(x_b1[i].unsqueeze(0)) 
            f_a, _ = self.id_a( scale2(x_a[i].unsqueeze(0)))
            f_b, _ = self.id_b( scale2(x_b[i].unsqueeze(0)))
            x_a_recon.append(self.gen_a.decode(s_a, f_a))
            x_b_recon.append(self.gen_b.decode(s_b, f_b))
            x_ba = self.gen_a.decode(s_b, f_a)
            x_ab = self.gen_b.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ab1.append(x_ab)
        self.train()

        return x_a,image_a_b,x_b,image_b_b

    def dis_update(self, x_ab, x_ba, x_a, x_b, hyperparameters, num_gpu):
        self.dis_opt.zero_grad()
        # D loss
        if num_gpu>1:
            self.loss_dis_a, reg_a = self.dis_a.module.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
            self.loss_dis_b, reg_b = self.dis_b.module.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
        else:
            self.loss_dis_a, reg_a = self.dis_a.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
            self.loss_dis_b, reg_b = self.dis_b.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        print("DLoss: %.4f"%self.loss_dis_total, "Reg: %.4f"%(reg_a+reg_b) )
        if self.fp16:
            with amp.scale_loss(self.loss_dis_total, self.dis_opt) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.id_scheduler is not None:
            self.id_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b = self.gen_a
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b = self.dis_a
        # Load ID dis
        last_model_name = get_model_list(checkpoint_dir, "id")
        state_dict = torch.load(last_model_name)
        self.id_a.load_state_dict(state_dict['a'])
        self.id_b = self.id_a
        # Load optimizers
        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])
            self.id_opt.load_state_dict(state_dict['id'])
        except:
            pass
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, num_gpu=1):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        id_name = os.path.join(snapshot_dir, 'id_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict()}, gen_name)
        if num_gpu>1:
            torch.save({'a': self.dis_a.module.state_dict()}, dis_name)
        else:
            torch.save({'a': self.dis_a.state_dict()}, dis_name)
        torch.save({'a': self.id_a.state_dict()}, id_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'id': self.id_opt.state_dict(),  'dis': self.dis_opt.state_dict()}, opt_name)



