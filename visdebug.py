#usage: python3 visdebug.py debug_file
import sys
import ast
import matplotlib.pyplot as plt

def print_usage_and_die():
  print("Usage: python3 visdebug.py debug_file error_label")
  quit()

def get_epoch(line):
  ret=-1
  pos=line.find("Epoch")
  if pos>=0:
    comma_pos=line.find(",")
    data=line[pos+len("Epoch"):comma_pos]
    r=data.split('/')
    ret=int(r[0])-1
  return(ret)



def get_loss(line):
  ret={}
  pos_left=line.find("{")
  pos_right=line.find("}")
  if pos_left>=0 and pos_right>=0:
    data=line[pos_left:pos_right+1]
    data=data.replace("array","")
    ret=ast.literal_eval(data)
  return(ret)
# 
#  
#if len(sys.argv)<2:
#  print_usage_and_die()
if len(sys.argv)<3:
  drawn_index=0
  label="loss"
  print("Drawn index not specified, drawing loss")
elif sys.argv[2].lower()=='loss' or sys.argv[2]=='0':
  drawn_index=0
  label="loss"
elif sys.argv[2].lower()=='seg' or sys.argv[2].lower()=='segmentation' or sys.argv[2]=='1':
  drawn_index=1
  label="l1 seg error"
elif sys.argv[2].lower()=='dist' or sys.argv[2].lower()=='distance' or sys.argv[2]=='2':
  drawn_index=2
  label="l1 dist error"

tmp=0
l=[]
with open(sys.argv[1]) as f:
  for line in f:
    e=get_epoch(line)
    if e>=0:
      for i in range(len(l),e+1):
        l.append(0)
      l[e]=get_loss(line)

train_train=[x['train_train'][drawn_index] for x in l]
validation_eval=[x['validation_eval'][drawn_index] for x in l]

start_epoch=0
plt.plot(range(start_epoch,len(validation_eval)),validation_eval,linestyle='-', color=[1,0,0])
plt.plot(range(start_epoch,len(train_train)),train_train,linestyle=':', color=[0,1,0])
plt.legend(['validation (eval mode)', 'train (train mode)'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel(label)
plt.show()


