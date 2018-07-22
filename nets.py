import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import is_torch_none, is_iterable
from torch.nn.functional import pad

def get_nested_structure_level(x):
  if not is_iterable(x):
    return(0)
  else:
    return(get_nested_structure_level(x[0])+1)

class Criterion(nn.modules.loss._Loss):
  def __init__(self,dt_bound):
    super(Criterion, self).__init__()
    self.bce=nn.BCELoss()
    self.mse=nn.MSELoss()
    self.dt_bound=dt_bound
  def forward(self, input, target):
    loss=self.bce.forward(input[:,0:1,:,:],target[:,0:1,:,:])+self.mse.forward(input[:,1:2,:,:],target[:,1:2,:,:])
    copied_input=input.data.clone()
    seg=copied_input[:,0:1,:,:]
    dist=copied_input[:,1:2,:,:]
    dist[dist>self.dt_bound]=self.dt_bound
    l1_seg=torch.abs(seg-target[:,0:1,:,:].data).mean()
    l1_dist=torch.abs(dist-target[:,1:2,:,:].data).mean()
    return(loss,l1_seg,l1_dist)

def batch_images_labels(batch,use_cuda):
  batch_images=[Variable(x) for x in batch[0]]
  batch_labels=None
  if not is_torch_none(batch[1]):
    batch_labels=Variable(batch[1])
  if use_cuda:
    for i in range(len(batch_images)):
      batch_images[i]=batch_images[i].cuda()
    if not is_torch_none(batch_labels):
     batch_labels=batch_labels.cuda()
  return(batch_images,batch_labels)

class LastActivation(nn.Module):
  def __init__(self):
    super(LastActivation, self).__init__()
    self.sigmoid=nn.Sigmoid()
    self.relu=nn.ReLU()
  def forward(self, input):
    return(torch.cat((self.sigmoid.forward(input[:,0:1,:,:]),self.relu.forward(input[:,1:,:,:])),dim=1))
    
class Model(nn.Module):
  def __init__(self,structure,batchnorm_momentum):
    super(Model, self).__init__()
    self.batchnorm_momentum=batchnorm_momentum
    self.structure=structure
    if get_nested_structure_level(self.structure)==1:
      self.structure=[self.structure]
    print("net structure: ",  self.structure)
    
    self.segments=nn.ModuleList()
    in_channels=1
    cumulative_out_channels=0
    for i,(numof_layers,out_channels,kernel_size) in enumerate(structure):
      cumulative_out_channels+=out_channels
      if kernel_size % 2 != 1:
        raise Exception("Size of the receptive field must be an odd number")
      self.segments.append(nn.ModuleList())
      if i>0:
        in_channels=cumulative_out_channels
      for l in range(numof_layers):
        print("Layer",l,"segment",i,"channels",cumulative_out_channels,sep=" ")
        self.segments[i].append(
          nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1)/2)),
            nn.Conv2d(in_channels, cumulative_out_channels, kernel_size=kernel_size),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(cumulative_out_channels,momentum=batchnorm_momentum)
            )
          )
        in_channels=cumulative_out_channels
      #Last or upsampling layer
      if i==len(structure)-1:
        self.segments[i].append(
          nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1)/2)),
            nn.Conv2d(in_channels, 2, kernel_size=kernel_size),
            LastActivation()))    
      else:
        self.segments[i].append(
          nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(cumulative_out_channels,momentum=batchnorm_momentum)))
  
  def forward(self,x):
    for i,segment in enumerate(self.segments):
      if i==0:
        out=x[i]
      else:
        #print("Forward:")
        #print(x[i].shape)
        #print(out.shape)
        
        if out.data.shape[2]<=x[i].data.shape[2] and out.data.shape[3]<=x[i].data.shape[3]:
          out=pad(out, [0,x[i].data.shape[3]-out.data.shape[3],0,x[i].data.shape[2]-out.data.shape[2]], 'constant', 0)
        #print(out.shape)
        #quit()
        out=torch.cat((out,x[i].repeat(1,self.structure[i][1],1,1)),dim=1)
      for layer in segment:
        out=layer(out)
    return(out)  

  def save(self,filename):
    torch.save({"batchnorm_momentum": self.batchnorm_momentum,
                "structure": self.structure,
                "state_dict": self.state_dict()},filename)
  
def load_model_from_file(filename):
  checkpoint = torch.load(filename,map_location=lambda storage, loc: storage)
  model=Model(checkpoint["structure"],checkpoint["batchnorm_momentum"])
  model.load_state_dict(checkpoint["state_dict"])
  return(model)





