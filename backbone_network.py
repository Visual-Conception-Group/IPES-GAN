"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import pdb
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
from utils import weights_init

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
        
def load_network(network):
    save_path = os.path.join('./models/Duke_best/net_last.pth')
    network.load_state_dict(torch.load(save_path),strict=False)
    return network

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        #num_bottleneck = input_dim # We remove the input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck, affine=True)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x
    
class cam_dis(nn.Module):
    def __init__(self):
        super(cam_dis, self).__init__()
        #self.cnns = nn.ModuleList()
        self.model = []
        self.model += [Conv2dBlock(3, 64, 3, 2, 1, norm='in', activation='lrelu', pad_type='reflect')]
        self.model += [Conv2dBlock(64, 128, 3, 1, 1, norm='in', activation='lrelu', pad_type='reflect')]
        self.model += [Conv2dBlock(128, 256, 3, 2, 1, norm='in', activation='lrelu', pad_type='reflect')]
        self.model += [Conv2dBlock(256, 512, 3, 1, 1, norm='in', activation='lrelu', pad_type='reflect')]
        self.model += [nn.AvgPool2d(3)]
        self.model += [nn.Dropout(p=0.999)]
        self.model += [nn.Linear(512,8)]
        self.model = nn.Sequential(*self.model)
        #self.cnns.apply(weights_init('gaussian'))
        #self.cnns.append(self.model)
    def forward(self,x):
        x = self.model(x)
        return x
    
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        # Here I change the stride to 2. 
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        
        self.model += [nn.Conv2d(dim, 128, 1, 1, 0)]
        
        
        self.model = nn.Sequential(*self.model)
        self.classifier = ClassBlock(128, 8)

    def forward(self, x):
        #pdb.set_trace()
        c_bf = self.model(x)
        #print(c_bf.shape)
        #print(x.shape)
        #pdb.set_trace()
        c_bf = c_bf.view(c_bf.shape[0],c_bf.shape[1])
        c_b  = self.classifier(c_bf)
        return c_b
               
 
        
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, fp16 = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16 = fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
        
# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, norm=False, pool='avg', stride=2):
        super(ft_net, self).__init__()
        if norm:
            self.norm = True
        else:
            self.norm = False
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1)) 
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1)) 
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # remove the final downsample
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.model = model_ft   
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)  # -> 512 32*16
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x) # 8 * 2048 4*1
        x = self.model.avgpool(x)  # 8 * 2048 1*1
        
        x = x.view(x.size(0),x.size(1))
        f = f.view(f.size(0),f.size(1)*self.part)
        if self.norm:
            fnorm = torch.norm(f, p=2, dim=1, keepdim=True) + 1e-8
            f = f.div(fnorm.expand_as(f))
        x = self.classifier(x)
        return f, x

class ft_netAB1(nn.Module):

    def __init__(self, class_num, norm=False, stride=2, droprate=0.5, pool='avg'):
        super(ft_netAB1, self).__init__()
        model_tmp = ft_net(702, stride = stride)
        model_ft = load_network(model_tmp)
        model_ft.model.fc = nn.Sequential()
        model_ft.classifier.classifier = nn.Linear(512, 8)
        #model_ft = models.resnet50(pretrained=True)
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft   

        if stride == 1:
            self.model.model.layer4[0].downsample[0].stride = (1,1)
            self.model.model.layer4[0].conv2.stride = (1,1)

        self.classifier1 = ClassBlock(2048, class_num, 0.5)
        self.classifier2 = ClassBlock(2048, class_num, 0.75)
        print("Duke_cam_classes",class_num)

    def forward(self, x):
        x = self.model.model.conv1(x)
        x = self.model.model.bn1(x)
        x = self.model.model.relu(x)
        x = self.model.model.maxpool(x)
        x = self.model.model.layer1(x)
        x = self.model.model.layer2(x)
        x = self.model.model.layer3(x)
        x = self.model.model.layer4(x)
        f = self.model.model.partpool(x)
        f = f.view(f.size(0),f.size(1)*self.part)
        f = f.detach() # no gradient
        x = self.model.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x=[]
        x.append(x1)
        x.append(x2)
        return f, x

# Define the AB Model
class ft_netAB(nn.Module):

    def __init__(self, class_num, norm=False, stride=2, droprate=0.5, pool='avg'):
        super(ft_netAB, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)
#        pdb.set_trace()
        self.classifier1 = ClassBlock(2048, class_num, 0.5)
        self.classifier2 = ClassBlock(2048, class_num, 0.75)
        
#        self.classifier3 = ClassBlock(2048, market_class_num, 0.5)
#        self.classifier4 = ClassBlock(2048, market_class_num, 0.75)
        
#        self.classifier5 = ClassBlock(2048, duke_class_num, 0.5)
#        self.classifier6 = ClassBlock(2048, duke_class_num, 0.75)
    def forward(self, x):
        #pdb.set_trace()
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        #pdb.set_trace()
        #print(x.shape)
#         part = {}
#         p=0
#         for i in range(2):
#             part[i] = x[:,:,p:p+8,:]
#             part[i] = self.model.avgpool(part[i])
#             p=p+4
#             part[i] = part[i].view(part[i].size(0), part[i].size(1))
        
       
       
        #x_2 = self.model.avgpool(part[1])
        #x_3 = self.model.avgpool(part[2])
        #x_4 = self.model.avgpool(part[3])
       # x3 = torch.cat((part[0],part[1],part[2],part[3]),dim=1)

        f = self.model.partpool(x)
        f = f.view(f.size(0),f.size(1)*self.part)
        f = f.detach() # no gradient 
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        
        x1 = self.classifier1(x) 
        x2 = self.classifier2(x)
 
        x=[]
        x.append(x1)
        x.append(x2)
        return f, x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 4 # We cut the pool5 to 4 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.softmax = nn.Softmax(dim=1)
        # define 4 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        f = x
        f = f.view(f.size(0),f.size(1)*self.part)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get part feature batchsize*2048*4
        for i in range(self.part):
            part[i] = x[:,:,i].contiguous()
            part[i] = part[i].view(x.size(0), x.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y=[]
        for i in range(self.part):
            y.append(predict[i])

        return f, y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer3[0].downsample[0].stride = (1,1)
        self.model.layer3[0].conv2.stride = (1,1)

        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y


