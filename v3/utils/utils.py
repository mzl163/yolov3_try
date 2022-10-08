from __future__ import division
import time
import platform
import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import subprocess
import random
import imgaug as ia

def provide_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ia.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def worker_seed_set(worker_id):
    uint64_seed=torch.initial_seed()
    ss=np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    worker_seed=torch.initial_seed()%2**32
    random.seed(worker_seed)
    
def to_cpu(tensor):
    return tensor.detach().cpu()

def load_classes(path):
    with open(path,'r') as fp:
        names=fp.read().splitlines()
    return names

def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm2d')!=-1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0.0)

def rescale_boxes(boxes,current_dim,original_shape):
    orig_h,orig_w=original_shape
    pad_x=max(orig_h-orig_w,0)*(current_dim/max(original_shape))
    pad_y=max(orig_w-orig_h,0)*(current_dim/max(original_shape))
    unpad_h=current_dim-pad_y
    unpad_w=current_dim-pad_x
    boxes[:,0]=((boxes[:,0]-pad_x//2)/unpad_w)*orig_w
    boxes[:,1]=((boxes[:,1]-pad_y//2)/unpad_h)*orig_h
    boxes[:,2]=((boxes[:,2]-pad_x//2)/unpad_w)*orig_w
    boxes[:,3]=((boxes[:,3]-pad_y//2)/unpad_h)*orig_h
    return boxes
def xywh2xyxy(x):
    y=x.new(x.shape)
    y[...,0]=x[...,0]-x[...,2]/2
    y[...,1]=x[...,1]-x[...,3]/2
    y[...,2]=x[...,0]-x[...,2]/2
    y[...,3]=x[...,1]-x[...,3]/2
    return y

def ap_per_class(tp,conf,pred_cls,target_cls):
    i=np.argsort(-conf)
    tp,conf,pred_cls=tp[i],conf[i],pred_cls[i]
    unique_classes=np.unique(target_cls)
    ap,p,r=[],[],[]
    for c in tqdm.tqdm(unique_classes,desc="Computing AP"):
        i=pred_cls==c
        n_gt=(target_cls==c).sum()
        n_p=i.sum()
        if n_p==0 and n_gt==0:
            continue
        elif n_p==0 or n_gt==0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            fpc=(1-tp[i]).cumsum()
            tpc=(tp[i]).cumsum()
            recall_curve=tpc/(n_gt+1e-16)
            r.append(recall_curve[-1])
            precision_curve=tpc/(tpc+fpc)
            p.append(precision_curve[-1])
            ap.append(compute_ap(recall_curve,precision_curve))
    p,r,ap=np.array(p),np.array(r),np.array(ap)
    f1=2*p*r/(p+r+1e-16)
    return p,r,ap,f1,unique_classes.astype('int32')
def compute_ap(recall,precision):
    mrec=np.concatenate(([0,0],recall,[1.0]))
    mpre=np.concatenate(([0,0],precision,[0.0]))
    for i in range(mpre.size-1,0,-1):
        mpre[i-1]=np.maximum(mpre[i-1],mpre[i])
    i=np.where(mrec[1:]!=mrec[:-1])[0]
    ap=np.sum((mrec[i+1]-mrec[i])*mpre[i+1])
    return ap

def get_batch_statistics(outputs,targets,iou_threshold):
    batch_metrics=[]
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        output=outputs[sample_i]
        pred_boxes=output[:,:4]
        pred_scores=output[:,4]
        pred_labels=output[:,-1]
        true_positives=np.zeros(pred_boxes.shape[0])
        annotations=targets[targets[:,0]==sample_i][:,1:]
        target_labels=annotations[:0] if len(annotations) else []
        if len(annotations):
            detected_boxes=[]
            target_boxes=annotations
            for pred_i,(pred_box,pred_label) in enumerate(zip(pred_boxes,pred_labels)):
                if len(detected_boxes)==len(annotations):
                    break
                if pred_label not in target_labels:
                    continue
                filtered_target_position,filtered_targets=zip(*filter(lambda x:target_boxes[x[0]]==pred_label,enumerate(target_boxes)))
                iou,box_filtered_index=bbox_iou(pred_box.unsqueeze(0),torch.stack(filtered_targets)).max(0)
                box_index=filtered_target_position[box_filtered_index]
                if iou>=iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i]=1
                    detected_box+=[box_index]
        batch_metrics.append([true_positives,pred_scores,pred_labels])
    return batch_metrics
def bbox_wh_iou(wh1,wh2):
    wh2=wh2.t()
    w1,h1=wh1[0],wh1[1]
    w2,h2=wh2[0],wh2[1]
    inter_area=torch.min(w1,w2)*torch.min(h1,h2)
    union_area=(w1*h1+1e-16)+w2*h2-inter_area
    return inter_area/union_area

def bbox_iou(box1,box2,x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1,b1_x2=box1[:,0]-box1[:2]/2,box1[:0]+box1[:,2]/2
        b1_y1,b1_y2=box1[:,1]-box1[:3]/2,box1[:1]+box1[:,3]/2
        b2_x1,b2_x2=box2[:,0]-box2[:2]/2,box2[:0]+box2[:,2]/2
        b2_y1,b2_y2=box2[:,1]-box2[:3]/2,box2[:1]+box2[:,3]/2
    else:
        b1_x1,b1_y1,b1_x2,b1_y2=\
            box1[:,0],box1[:,1],box1[:,2],box1[:,3]
        b2_x1,b2_y1,b2_x2,b2_y2=\
            box2[:,0],box2[:,1],box2[:,2],box2[:,3]
    inter_rect_x1=torch.max(b1_x1,b2_x1)
    inter_rect_y1=torch.max(b1_y1,b2_y1)
    inter_rect_x2=torch.max(b1_x2,b2_x2)
    inter_rect_y2=torch.max(b1_y2,b2_y2)
    inter_area=torch.clamp(inter_rect_x2-inter_rect_x1+1,min=0)*torch.clamp(
        inter_rect_y2-inter_rect_y1+1,min=0
    )
    b1_area=(b1_x2-b1_x1+1)*(b1_y2-b1_y1+1)
    b2_area=(b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)
    iou=inter_area/(b1_area+b2_area-inter_area+1e-16)
    return iou
def box_iou(box1,box2):
    def box_area(box):
        return (box[2]-box[0])*(box[3]-box[1])
    area1=box_area(box1.T)
    area2=box_area(box2.T)
    inter=(torch.min(box1[:,None,2:],box2[:,2:])-torch.max(box1[:None,:2],box2[:,:2])).clamp(0).prod(2)
    return inter/(area1[:,None]+area2-inter)

def non_max_suppression(prediction,conf_thres=0.25,iou_thres=0.45,classes=None):
    nc=prediction.shape[2]-5
    max_wh=4096
    max_det=300
    max_nms=30000
    time_limit=1.0
    multi_label=nc>1
    t=time.time()
    output=[torch.zeros((0,6),device='cpu')]*prediction.shape[0]
    for xi,x in enumerate(prediction):
        x=x[x[...,4]>conf_thres]
        if not x.shape[0]:
            continue
        x[:,5:]*=x[:,4:5]
        box=xywh2xyxy(x[:,:4])
        if multi_label:
            i,j=(x[:,5:]>conf_thres).nonzero(as_tuple=False).T
            x=torch.cat((box[1],x[i,j+5,None],j[:,None].float()),1)
        else:
            conf,j=x[:,5:].max(1,keepdim=True)
            x=torch.cat((box,conf,j.float()),1)[conf.view(-1)>conf_thres]
        if classes is not None:
            x=x[(x[:,5:6]==torch.tensor(classes,device=x.device)).any(1)]
        n=x.shape[0]
        if not n:
            continue
        elif n>max_nms:
            x=x[x[:4].argsort(descending=True)[:max_nms]]
            
        c=x[:,5:6]*max_wh
        boxes,scores=x[:,:4]+c,x[:,4]
        i = torchvision.ops.nms(boxes,scores,iou_thres)
        if i.shape[0]>max_det:
            i=i[:max_det]
        output[xi]=to_cpu(x[i])
        if (time.time()-t)>time_limit:
            print(f'WARNING:NMS')
            break
        
    return output
def print_environment_info():
    print('Environmen information:')
    print(f'System: {platform.system()} {platform.release()}')
    try:
        print(f"Current Version:")
    except (subprocess.CalledProcessError,FileNotFoundError):
        print("Not using the poatry package")
    try:
        print(f"Current Commit Hash")
    except:
        print("No git or repo found")
t=[[-0.9532,  0.4367, -0.1972],
        [ 2.1078,  0.3750, -0.2939],
        [-0.3682,  1.3246, -0.7197],
        [-0.4119,  0.2093, -0.3431],
        [-1.7094,  0.0638, -0.4597]]
t=np.array(t)
t=torch.FloatTensor(t)
print(t.prod())