import glob
import os
import numpy as np
import scipy
import scipy.ndimage
import torch
from random import shuffle
from torch.utils.data import Dataset
from torch.autograd import Variable
from scipy.ndimage.morphology import distance_transform_cdt
from torch.nn.functional import avg_pool2d




def get_isbi_filenames(root_dir,images_idx={"01":[],"02":[]}):
  images=[]
  gt=[]
  for sequence in sorted(images_idx):
    images_root=os.path.join(root_dir,sequence)
    seg_root=os.path.join(root_dir,sequence+"_GT","SEG")
    if len(images_idx[sequence])==0:
      n=len(glob.glob(os.path.join(images_root,"*.tif")))
      idx=[str(i).zfill(3) for i in range(0,n)]
    else:
      idx=images_idx[sequence]
    images+=[os.path.join(images_root,"t"+x+".tif") for x in idx]
    gt+=[os.path.join(seg_root,"man_seg"+x+".tif") for x in idx]
  return(images,gt)

def get_isbi_dataset_intensity_min_max(dataset_name):
  retmin=0
  retmax=255
  if 'DIC-C2DH-HeLa' in dataset_name:
    retmin=0.0
    retmax=255.0
  if 'Fluo-C2DL-MSC' in dataset_name:
    retmin=2000.0
    retmax=65535.0
  if 'Fluo-N2DH-GOWT1' in dataset_name:
    retmin=0.0
    retmax=255.0
  if 'Fluo-N2DH-SIM' in dataset_name:
    retmin=60.0
    retmax=1000.0
  if 'Fluo-N2DL-HeLa' in dataset_name:
    retmin=32768.0
    retmax=36863.0
  if 'PhC-C2DH-U373' in dataset_name:
    retmin=20.0
    retmax=200.0
  if 'PhC-C2DL-PSC' in dataset_name:
    retmin=0.0
    retmax=255.0
  return(retmin,retmax)




def validation_split(image_filenames,gt_filenames,validation_percentage):
  training_images=[]
  training_gt=[]
  validation_images=[]
  validation_gt=[]
  if validation_percentage<=0:
    training_images=image_filenames
    training_gt=gt_filenames
  elif validation_percentage>=1:
    validation_images=image_filenames
    validation_gt=gt_filenames
  else:
      idx=list(range(0,len(image_filenames)))
      shuffle(idx)
      split=int(validation_percentage*len(image_filenames))
      training_images=[image_filenames[idx[i]] for i in range(split,len(image_filenames))]
      validation_images=[image_filenames[idx[i]] for i in range(0,split)]
      if len(gt_filenames)>0:
        training_gt=[gt_filenames[idx[i]] for i in range(split,len(image_filenames))]
        validation_gt=[gt_filenames[idx[i]] for i in range(0,split)]
  return(training_images,training_gt,validation_images,validation_gt)

def is_torch_none(object):
  ret=object is None
  if object is not None:
    try:
      l=len(object)
      ret=l<=0
    except:
      pass
  return(ret)
  
def is_iterable(x):
  try:
    _=iter(x)
    return(True)
  except TypeError:
    return(False)
    
def rescale_tensor(tensor,factor):
  if factor<0:
    return(avg_pool2d(tensor,2**(-factor)))
  else:
    return(tensor)
  
def do_equalize_histogram(image,histmin=0,histmax=255,histbins=256):
  image=(np.copy(image)-histmin)/histmax
  h,bins= np.histogram(image,bins=histbins,range=(0,1))
  h=h/sum(h) #this step is necessary in situations, when min(image)<histmin or max(image)>histmax
  cumh=np.cumsum(h)
  cumh=cumh*(histmax-histmin)+histmin
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      index=int(image[x,y]*len(cumh))
      if index>=len(cumh): #numerical errors
        index=len(cumh)-1
      image[x,y]=cumh[index]
  return(image)

class Cells(Dataset):
  """Loads images, gt segmentation and gt boundary from a dataset."""
  def __init__(self,image_filenames,gt_filenames=[],dt_bound=1,resolution_levels=[0],load_to_memory=True,equalize_histogram=False,len_multiplier=1,use_cuda=False):
    self.image_filenames=image_filenames
    self.gt_filenames=gt_filenames
    self.images=[]
    self.gt_segmentations=[]
    self.gt_boundaries=[]
    self.dt_bound=dt_bound
    self.len_multiplier=len_multiplier
    self.equalize_histogram=equalize_histogram
    self.use_cuda=torch.cuda.is_available() and use_cuda
    if is_iterable(resolution_levels):
      self.resolution_levels=resolution_levels
    else:
      self.resolution_levels=[0]
    self.transform_schedule=[]
    self.transforms={}
    if load_to_memory:
      for i in range(0,len(self.image_filenames)):
        image,gt_segmentation,gt_boundary=self.load(i)
        self.images.append(image)
        self.gt_segmentations.append(gt_segmentation)
        self.gt_boundaries.append(gt_boundary)
  def set_transform_schedule(self,schedule):
    self.transform_schedule=schedule
  def add_transform(self,key,transform,update_schedule=True):
    self.transforms[key]=transform
    self.transform_schedule.append(key)
  def load(self,index):
    if index<len(self.images) and index<len(self.gt_segmentations) and index<len(self.gt_boundaries):
      image=np.copy(self.images[index])
      gt_segmentation=np.copy(self.gt_segmentations[index])
      gt_boundary=np.copy(self.gt_boundaries[index])
    else:
      image=scipy.misc.imread(self.image_filenames[index],flatten=False).astype(float)
      if self.equalize_histogram:
        histmin,histmax=get_isbi_dataset_intensity_min_max(self.image_filenames[index])
        image=do_equalize_histogram(image,histmin=histmin,histmax=histmax)
      gt_segmentation=None
      gt_boundary=None
      if len(self.gt_filenames)>index:
        gt_segmentation=scipy.misc.imread(self.gt_filenames[index],flatten=False)
        boundary_pixels=np.ones(gt_segmentation.shape,dtype=bool)
        boundary_pixels[0:-1,:]&=(gt_segmentation[0:-1,:]==gt_segmentation[1:,:])
        boundary_pixels[1:,:]&=(gt_segmentation[1:,:]==gt_segmentation[0:-1,:])
        boundary_pixels[:,0:-1]&=(gt_segmentation[:,0:-1]==gt_segmentation[:,1:])
        boundary_pixels[:,1:]&=(gt_segmentation[:,1:]==gt_segmentation[:,0:-1])
        gt_boundary=distance_transform_cdt(boundary_pixels,metric='taxicab').astype(float)
        gt_boundary[gt_boundary>self.dt_bound]=self.dt_bound
        gt_segmentation=(gt_segmentation>0.5) 
        gt_segmentation=gt_segmentation.astype(float)
    return(image,gt_segmentation,gt_boundary)
  def save(self,filename_base,image,gt_segmentation,gt_boundary,use_scipy_clever_convert=True):
    if use_scipy_clever_convert:
      scipy.misc.imsave(filename_base+"_img.png",image)
    else:
      histmin,histmax=get_isbi_dataset_intensity_min_max(self.image_filenames[0])
      int_image=np.copy(image)*255
      int_image[int_image<0]=0
      int_image[int_image>255]=255
      int_image=int_image.astype(np.uint8)
      scipy.misc.imsave(filename_base+"_img.png",int_image)
    scipy.misc.imsave(filename_base+"_gtseg.png", gt_segmentation)
    scipy.misc.imsave(filename_base+"_gtbound.png",gt_boundary)
  def get_filename_basis(self,index):
    base_name=os.path.splitext(os.path.basename(self.image_filenames[index]))[0]
    sequence=os.path.basename(os.path.dirname(self.image_filenames[index]))
    return(sequence+"_"+base_name)


  def __getitem__(self,index):
    image_index=int(index/self.len_multiplier)
    images=[None]*len(self.resolution_levels)
    image,gt_segmentation,gt_boundary=self.load(image_index)
    if is_torch_none(gt_segmentation) or is_torch_none(gt_boundary):
      gt_labels=[]
    else:
      gt_segmentation=np.expand_dims(np.expand_dims(gt_segmentation,0),0)
      gt_boundary=np.expand_dims(np.expand_dims(gt_boundary,0),0)
      gt_labels=Variable(torch.cat((torch.from_numpy(gt_segmentation).float(),torch.from_numpy(gt_boundary).float()),dim=1),requires_grad=False)
      if self.use_cuda:
        gt_labels=gt_labels.cuda()
    
    image=np.expand_dims(np.expand_dims(image,0),0)
    image=Variable(torch.from_numpy(image).float(),requires_grad=False)
    if self.use_cuda:
      image=image.cuda()
      
    if not is_torch_none(gt_labels):
      for tid in self.transform_schedule:
        image,gt_labels=self.transforms[tid](image,gt_labels)
      gt_labels=rescale_tensor(gt_labels,self.resolution_levels[-1])
      gt_labels=gt_labels.squeeze(0).data

    for i,level in enumerate(self.resolution_levels):
      images[i]=rescale_tensor(image,level)
      images[i]=images[i]-images[i].mean()
      images[i]=images[i]/(3*images[i].std())
      images[i]=images[i].squeeze(0).data
    return(images,gt_labels)
    
  def __len__(self):
    return(len(self.image_filenames)*self.len_multiplier)
    
  def __debug_tensors_manual__(self,image,gt_labels,filename_base):
    image=np.squeeze(image.numpy())
    gt_segmentation=np.squeeze(gt_labels[0,:,:].numpy())
    gt_boundary=np.squeeze(gt_labels[1,:,:].numpy())
    print("Image min,max,mean:",image.min(),image.max(),image.mean(),sep=' ')
    print("gt_segmentation min,max,mean:",gt_segmentation.min(),gt_segmentation.max(),gt_segmentation.mean(),sep=' ')
    print("gt_boundary min,max,mean:",gt_boundary.min(),gt_boundary.max(),gt_boundary.mean(),sep=' ')
    image=image/8
    image=image+0.5
    self.save(filename_base,image,gt_segmentation,gt_boundary,use_scipy_clever_convert=False)

  def __debug_tensors__(self,index,filename_base):
    index/=self.len_multiplier
    images,gt_labels=self.__getitem__(index)
    self.__debug_tensors_manual__(images[-1],gt_labels,filename_base)


