import os
import time
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nets import Criterion,batch_images_labels
from eval import eval

class ErrorTracker:
  def __init__(self,numof_best_non_decrease=10):
    self.numof_best_non_decrease=numof_best_non_decrease
    self.current_best_non_decrease=0
    self.current_best=-1
  def is_non_decreasing(self):
    return self.current_best_non_decrease>=self.numof_best_non_decrease
  def update(self,error):
    if self.current_best<0 or error<self.current_best:
      self.current_best=error
      self.current_best_non_decrease=0
    else:
      self.current_best_non_decrease+=1

def train(model,train_set,validation_set,args):
 
  optimizer=optim.Adam(model.parameters(),lr=args.learning_rate)
  scheduler=ReduceLROnPlateau(optimizer,'min',factor=0.75,patience=20,cooldown=10)
  et=ErrorTracker(10)

  train_dataloader=DataLoader(train_set, batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
  validation_dataloader=DataLoader(validation_set, batch_size=1,num_workers=args.num_workers,shuffle=True)
  
  criterion=Criterion(args.dt_bound)
  use_cuda=torch.cuda.is_available() and args.cuda
  model.cuda() if use_cuda else model.cpu()

  best_err={'train_train':-1,'validation_eval':-1}
  early_stop=False
  start_time=time.time()
  for epoch in range(0, args.num_epochs):
    if early_stop:
      break
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    model.train()
    
    err={'train_train':np.zeros(3),'validation_eval':np.zeros(3)}
    #Training
    for i,batch in enumerate(train_dataloader):
      batch_images,batch_labels=batch_images_labels(batch,use_cuda)
      optimizer.zero_grad()
      outputs=model(batch_images)
      loss,l1_seg,l1_dist=criterion(outputs, batch_labels)
      loss.backward()
      optimizer.step()
      err['train_train']+=np.array((loss.data[0],l1_seg,l1_dist))/len(train_dataloader)

    #Evaluate the network in eval mode   
    err['validation_eval']=eval(model,validation_set,args,save_images=False,print_output=False)
    et.update(err['validation_eval'][0])
    if et.is_non_decreasing() and len(args.non_decreasing_output_file)>0:
      early_stop=True
      with open(args.non_decreasing_output_file, "a") as myfile:
        myfile.write(str(err['train_train'])+","+str(et.current_best)+","+str(epoch)+","+str(time.time()-start_time)+"\n")
    #Save the model to file
    for set_mode in err:
      if err[set_mode][0]<best_err[set_mode] or best_err[set_mode]<0:
        best_err[set_mode]=err[set_mode][0]
        model.save(os.path.join(args.output_dir,"model_best_"+set_mode))
    model.save(os.path.join(args.output_dir,"model_last"))
    if epoch>0 and args.save_model_frequency>0 and epoch % args.save_model_frequency==0:
      model.save(os.path.join(args.output_dir,"model_"+str(epoch)))
    #Print the loss
    print("Epoch "+str(epoch+1)+"/"+str(args.num_epochs)+", ",err)
    scheduler.step(err['train_train'][0])

  return(model)   

