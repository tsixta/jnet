import os
import argparse
import ast
from enum import Enum
import torch
from dataset import Cells,get_isbi_filenames,validation_split,is_iterable
from nets import Model,load_model_from_file
from train import train
from eval import eval
from augmentation import RandomRotation,RandomElastic,RandomFlipRotation,RandomIntensity

class Mode(Enum):
  Train = 'train'
  Eval = 'eval'
  Vis = 'vis'
  def __str__(self):
    return(self.value)

def eval_elastic_params(arg):
  if len(arg)>0:
    ret=ast.literal_eval(arg)
    if not is_iterable(ret):
      ret=[ret]
    ret=[x for x in ret if is_iterable(x) and (len(x)==2 or len(x)==3)]
    for x in ret:
      if len(x)==2:
        x.append(1)
    ret=[(int(x[0]),int(x[1]),int(x[2])) for x in ret]
  else:
    ret=[]
  return(ret)

def eval_intensity_params(arg):
  ret=[]
  if len(arg)>0:
    ret=ast.literal_eval(arg)
    if is_iterable(ret) and len(ret)==4:
      try:
        ret=[float(x) for x in ret]
      except ValueError:
        pass
  return(ret)

  
def print_eval_vis_commands(args):
  cuda="--cuda" if args.cuda else ""
  cuda_visible_devices=""
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    cuda_visible_devices = "CUDA_VISIBLE_DEVICES="+os.environ['CUDA_VISIBLE_DEVICES']+" "
    
  common_prefix=cuda_visible_devices+"python3 main.py "+cuda+" --resolution_levels '"+str(args.resolution_levels)+"' --dt_bound "+str(args.dt_bound)
  test_dataset=args.dataset_root.replace("_training/orig","_test")

  print("===Eval command best validation===")
  print(common_prefix+" --images_idx '"+str(args.images_idx)+"' --mode eval --dataset_root "+args.dataset_root+" --model_file "+args.output_dir+"/model_best_train_train --output_dir "+args.output_dir)
  print("===Vis command===")
  print(common_prefix+" --images_idx '{\"01\":[],\"02\":[]}' --mode vis --dataset_root "+test_dataset+" --model_file "+args.output_dir+"/model_best_train_train --output_dir "+args.output_dir)
  print("======")
    
  
  
parser = argparse.ArgumentParser(description='Cell detection')
parser.add_argument('--cuda',  action="store_true",help='Use GPU if available')
parser.add_argument('--dataset_root', required=True, type=str,help='Directory with the dataset (it contains subdirectories 01, 02 and optionally 01_GT and 02_GT)') 
parser.add_argument('--images_idx', required=True, type=str,help='Dictionary with image ids. The keys are datasets (01 or 02), the values lists with three digit indices. Example: {"01":["002","005"],"02":["006","007"]}') 
parser.add_argument('--output_dir', required=True, type=str,help='Output directory')
parser.add_argument('--mode',default=Mode.Train,type=Mode,help="Mode, one of the following: "+str([str(mode) for mode in list(Mode)]))
parser.add_argument('--resolution_levels', required=True, type=str,help='List of resolutions in the pipeline. 0 means the original resolution, -1 downscale by factor 2, -2 downscale by factor 4 etc.')
parser.add_argument('--structure',required=True,type=str,help='Structure of the network [[numof_layers1,numof_channels1,rf_size1],[numof_layers2,additional_numof_channels2,rf_size2],[numof_layers3,additional_numof_channels3,rf_size3]...]')
parser.add_argument('--aug_elastic_params', default='[]', type=str,help='Augmentation elastic, list of admissible [alpha sigma], alpha<=0 or sigma<=0 means no elastic transform')
parser.add_argument('--aug_intensity_params', default='[]', type=str,help='Augmentation intensity, list of [shift_lbound, shift_ubound, mult_lbound, mult_ubound]')
parser.add_argument('--aug_rotation', action='store_true', help='Augmentation rotation.')
parser.add_argument('--aug_rotation_flip', action='store_true', help='Augmentation rotation90 flip')
parser.add_argument('--batch_size', default=1, type=int,help='Batch size')
parser.add_argument('--batchnorm_momentum', default=0.1, type=float,help='Momentum parameter for BatchNorm2d layers')
parser.add_argument('--dt_bound',default=9,type=int,help='Bound for the distance transform')
parser.add_argument('--learning_rate', default=0.0001, type=float,help='Learning rate')
parser.add_argument('--load_dataset_to_ram', default=0, type=int,help='Preload images to RAM instead of loading them on demand')
parser.add_argument('--dataset_len_multiplier', default=1, type=int,help='Use every training image dataset_len_multiplier times')
parser.add_argument('--model_file', default='', type=str,help='Filename of loaded model')
parser.add_argument('--non_decreasing_output_file', default="", type=str,help='Output file for validation experiments')
parser.add_argument('--num_epochs', default=5000, type=int,help='Number of epochs')
parser.add_argument('--num_workers', default=0, type=int,help='Number of workers for the dataloader. If -1, data augmentation is done on GPU')
parser.add_argument('--save_model_frequency',default=200,type=int,help='Save model every --save_model_frequency epoch. -1 means never')
parser.add_argument('--validation_percentage',default=0.0,type=float,help='Percentage of the available training images used for validation.')
args = parser.parse_args()
print("==================================args=============================")
print(args)
print("================================end args===========================")
args.cuda=torch.cuda.is_available() and args.cuda
augmentation_cuda=args.cuda if args.num_workers==-1 else False
if args.num_workers<0:
  args.num_workers=0
print("CUDA:",args.cuda,augmentation_cuda,sep=" ")


if args.mode is Mode.Train:
  print_eval_vis_commands(args)

if not os.path.isdir(args.dataset_root):
  raise Exception("Unable to load images from "+args.dataset_root+": not a directory")

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)
if not os.path.isdir(args.output_dir):
  raise Exception("Unable to save results to "+args.output_dir+": not a directory")

  
#Create or load the network
if len(args.model_file)==0:
  model=Model(ast.literal_eval(args.structure),args.batchnorm_momentum)
else:
  model=load_model_from_file(args.model_file)

#Generate filenames of images in the dataset_root dataset
image_filenames,gt_filenames=get_isbi_filenames(
                               args.dataset_root,
                               ast.literal_eval(args.images_idx))
print(args.mode)


#Train, evaluate or visualize
if args.mode is Mode.Train:
  training_images,training_gt,validation_images,validation_gt=validation_split(image_filenames,gt_filenames,args.validation_percentage)
  train_set=Cells(training_images,training_gt,args.dt_bound,ast.literal_eval(args.resolution_levels),
                  load_to_memory=bool(args.load_dataset_to_ram),
                  len_multiplier=args.dataset_len_multiplier,
                  use_cuda=augmentation_cuda)
  if args.aug_rotation:
    train_set.add_transform("rotation",RandomRotation())
  intensity_params=eval_intensity_params(args.aug_intensity_params)
  if len(intensity_params)==4:
    train_set.add_transform("intensity",RandomIntensity(*intensity_params))
  elastic_params=eval_elastic_params(args.aug_elastic_params)
  if len(elastic_params)>0:
    train_set.add_transform("elastic",RandomElastic(elastic_params))
  if args.aug_rotation_flip:
    train_set.add_transform("fliprot90",RandomFlipRotation())
  
  validation_set=Cells(validation_images,validation_gt,args.dt_bound,ast.literal_eval(args.resolution_levels),load_to_memory=bool(args.load_dataset_to_ram))
  model=train(model,train_set,validation_set,args)

  model.save(os.path.join(args.output_dir,"model_final"))
elif args.mode is Mode.Eval:
  dataset=Cells(image_filenames,gt_filenames,args.dt_bound,ast.literal_eval(args.resolution_levels),load_to_memory=bool(args.load_dataset_to_ram))
  eval(model,dataset,args)
elif args.mode is Mode.Vis:
  dataset=Cells(image_filenames,[],args.dt_bound,ast.literal_eval(args.resolution_levels),load_to_memory=bool(args.load_dataset_to_ram))
  eval(model,dataset,args)
else:
  raise Exception("Unknown mode: "+str(args.mode))

