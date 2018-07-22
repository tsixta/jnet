import math
import random
import numpy as np
import scipy.ndimage
import torch
from torch.nn.functional import affine_grid,grid_sample,conv2d
from torch.autograd import Variable



def grid(shape):
  N, C, H, W = shape
  grid = torch.Tensor(N, H, W, 2)
  linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
  grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])
  linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
  grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
  return(Variable(grid,requires_grad=False))

def reflect_grid(grid):
  grid[grid<-1]=-2-grid[grid<-1]
  grid[grid>1]=2-grid[grid>1]


def gaussian_kernel_half_size(sigma,truncate=4.0):
  return(int(truncate * sigma + 0.5))
  
def gaussian_filter_1d(sigma,vertical,truncate=4.0):
  lw = gaussian_kernel_half_size(sigma,truncate)
  ret=torch.Tensor(2*lw+1)
  ret[lw] = 1.0
  exponent_mult=-1.0/(2.0*float(sigma*sigma))
  for ii in range(1, lw + 1):
    tmp = math.exp(exponent_mult*float(ii * ii))
    ret[lw + ii] = tmp
    ret[lw - ii] = tmp
  ret/=ret.sum()
  if vertical:
    return(Variable(ret.unsqueeze(0).unsqueeze(0).unsqueeze(-1),requires_grad=False))
  else:
    return(Variable(ret.unsqueeze(0).unsqueeze(0).unsqueeze(0),requires_grad=False))

def tensor_flip(t,vertical):
  if vertical:
    indices=list(reversed(range(t.data.shape[2])))
    return(t[:,:,indices,:])
  else:
    indices=list(reversed(range(t.data.shape[3])))
    return(t[:,:,:,indices])


#https://discuss.pytorch.org/t/implementation-of-function-like-numpy-roll/964/6
def tensor_roll(tensor, shift, axis):
    if shift == 0:
        return tensor
    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)
    
class RandomCrop:
  def __init__(self,dim3,dim2):
    self.dim2=dim2
    self.dim3=dim3
  def __call__(self,image,gt_labels):
    d2=min(image.data.shape[2],self.dim2)
    d3=min(image.data.shape[3],self.dim3)
    start2=np.random.randint(0,image.data.shape[2]-d2+1)
    start3=np.random.randint(0,image.data.shape[3]-d3+1)
    tmpimage=image[:,:,start2:(start2+d2),start3:(start3+d3)]
    tmplabels=gt_labels[:,:,start2:(start2+d2),start3:(start3+d3)]
    return(image[:,:,start2:(start2+d2),start3:(start3+d3)],gt_labels[:,:,start2:(start2+d2),start3:(start3+d3)])
 
class RandomIntensity:
  def __init__(self,shiftlbound,shiftubound,multlbound,multubound):
    self.shiftlbound=shiftlbound
    self.shiftubound=shiftubound
    self.multlbound=multlbound
    self.multubound=multubound
    
  def __call__(self,image,gt_labels):
    seg_times_image=gt_labels[:,0:1,:,:]*image
    foreground_mean=(seg_times_image).sum()/gt_labels[:,0:1,:,:].sum()
    shift=0 if self.shiftlbound>=self.shiftubound else random.uniform(self.shiftlbound,self.shiftubound)*foreground_mean
    mult=0 if self.multlbound>=self.multubound else random.uniform(self.multlbound,self.multubound)
    return(image+(mult*seg_times_image)+shift*gt_labels[:,0:1,:,:],gt_labels)
    
    
class RandomFlipRotation:
  def __call__(self,image,gt_labels):
    action=np.random.randint(0,6)
    if action==0: #nothing
      return(image,gt_labels)
    elif action==1: #rotation 90°
      retimage=tensor_flip(image,True)
      retgt_labels=tensor_flip(gt_labels,True)
      retimage.data.transpose_(2, 3)
      retgt_labels.data.transpose_(2, 3)
      return(retimage,retgt_labels)
    elif action==2: #rotation 180°
      return(tensor_flip(tensor_flip(image,True),False),
             tensor_flip(tensor_flip(gt_labels,True),False))
    elif action==3: #rotation 270°
      image.data.transpose_(2, 3)
      gt_labels.data.transpose_(2, 3)
      return(tensor_flip(image,True),tensor_flip(gt_labels,True))
    elif action==4: #flip axis 2
      return(tensor_flip(image,True),tensor_flip(gt_labels,True))
    elif action==5: #flip axis 3
      return(tensor_flip(image,False),tensor_flip(gt_labels,False))
    



class RandomRotation:
  def __call__(self,image,gt_labels):
    shape=image.data.shape
    angle=random.uniform(0,np.deg2rad(360))
    for axis in (2,3):
      roll=np.random.randint(0,shape[axis])
      retimage=tensor_roll(image,roll,axis)
      retgt_labels=tensor_roll(gt_labels,roll,axis)
    
    rot_matrix = np.array([[[np.cos(angle),  np.sin(angle),0],
                          [-np.sin(angle), np.cos(angle),0]]])
    ag=affine_grid(torch.Tensor(rot_matrix),shape)
    if image.is_cuda:
      ag=ag.cuda()
    reflect_grid(ag)

    retimage=grid_sample(retimage, ag)
    retgt_labels=grid_sample(retgt_labels, ag)
    return(retimage,retgt_labels)

class RandomElastic:
  """Elastic deformation of images as described in [Simard2003] 
     Simard, Steinkraus and Platt, "Best Practices for
     Convolutional Neural Networks applied to Visual Document Analysis", in
     Proc. of the International Conference on Document Analysis and
     Recognition, 2003.
     Copied from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
  """
  def __init__(self,params):
    self.params=params
    self.last_grid=None
    self.params_p=np.array([x[2] for x in self.params])
    self.params_p=self.params_p/np.sum(self.params_p)
  def get_params_random(self):
    if len(self.params)<1:
      return(-1,-1)
    elif len(self.params)==1:
      return(self.params[0][0],self.params[0][1])
    else:
      i=np.random.choice(range(len(self.params)),p=self.params_p)
      return(self.params[i][0],self.params[i][1])
  def update_last_grid(self,image):
    N, C, H, W = image.data.shape
    if self.last_grid is None or self.last_grid.data.shape!=torch.Size([N, H, W, 2]):
      self.last_grid=grid(image.data.shape)  
      if image.is_cuda:
        self.last_grid=self.last_grid.cuda()
  def __call__(self,image,gt_labels):
    alpha,sigma=self.get_params_random()
    if alpha<=0 or sigma<=0:
      return(image,gt_labels)
    else:
      shape = image.data.shape
      gkhs2=2*gaussian_kernel_half_size(sigma)
      convshape=torch.Size([1,shape[1],shape[2]+gkhs2,shape[3]+gkhs2])
      
      dx=Variable(torch.Tensor(convshape).uniform_(-2/shape[2],2/shape[3]),requires_grad=False)
      dy=Variable(torch.Tensor(convshape).uniform_(-2/shape[2],2/shape[3]),requires_grad=False)
      fh=gaussian_filter_1d(sigma,True)
      fv=gaussian_filter_1d(sigma,False)
      if image.is_cuda:
        dx=dx.cuda()
        dy=dy.cuda()
        fh=fh.cuda()
        fv=fv.cuda()
      dx=torch.nn.functional.conv2d(dx, fh)
      dx=torch.nn.functional.conv2d(dx, fv)
      dx.mul_(alpha)
      dy=torch.nn.functional.conv2d(dy, fh)
      dy=torch.nn.functional.conv2d(dy, fv)
      dy.mul_(alpha)
      distortion=torch.cat([dx[0].unsqueeze(-1),dy[0].unsqueeze(-1)],-1)
      self.update_last_grid(image)
      grid=torch.add(self.last_grid,distortion)
      reflect_grid(grid)
      retimage=grid_sample(image, grid)
      retgt_labels=grid_sample(gt_labels, grid)
      return(retimage,retgt_labels)

    

