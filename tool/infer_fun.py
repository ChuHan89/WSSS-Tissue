import imp
from pdb import set_trace
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
from tool.gradcam import GradCam
def CVImageToPIL(img):
    img = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))
    return img
def PILImageToCV(img):
    img = np.asarray(img)
    img = img[:,:,::-1]
    return img

def fuse_mask_and_img(mask, img):
    mask = PILImageToCV(mask)
    img = PILImageToCV(img)
    Combine = cv2.addWeighted(mask,0.3,img,0.7,0)
    return Combine

def infer(model, dataroot, n_class):
    model.eval()
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    cam_list = []
    gt_list = []    
    bg_list = []
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'img'),transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]; 

        img_path = os.path.join(os.path.join(dataroot,'img'),img_name+'.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=0.25):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam, y = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    y = y.cpu().detach().numpy().tolist()[0]
                    label = torch.tensor([1.0 if j >thr else 0.0 for j in y])
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                    return cam, label

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam-np.min(sum_cam)) / (np.max(sum_cam)-np.min(sum_cam))

        # cam --> segmap
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        if iter%100==0:
            print(iter)
        cam_list.append(seg_map)
        gt_map_path = os.path.join(os.path.join(dataroot,'mask'), img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)
    return iouutils.scores(gt_list, cam_list, n_class=n_class)

      
def create_pseudo_mask(model, dataroot, fm, savepath, n_class, palette, dataset):
    # print(model)
    if fm=='b4_3':
        ffmm = model.b4_3
    elif fm=='b4_5':
        ffmm = model.b4_5
    elif fm=='b5_2':
        ffmm = model.b5_2
    elif fm=='b6':
        ffmm = model.b6
    elif fm=='bn7':
        ffmm = model.bn7
    else:
        print('error')
        return
    print(dataset)
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'train'),transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):      
        img_name = img_name[0]
        img_path = os.path.join(os.path.join(dataroot,'train'),img_name+'.png')
        orig_img = np.asarray(Image.open(img_path))
        grad_cam = GradCam(model=model, feature_module=ffmm, \
                target_layer_names=["1"], use_cuda=True)
        cam = []
        for i in range(n_class):
            target_category = i
            grayscale_cam, _ = grad_cam(img_list, target_category)
            cam.append(grayscale_cam)
        norm_cam = np.array(cam)
        _range = np.max(norm_cam) - np.min(norm_cam)
        norm_cam = (norm_cam - np.min(norm_cam))/_range
        ##  Extract the image-level label from the filename
        ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
        ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
        label_str = img_name.split(']')[0].split('[')[-1]
        if dataset == 'luad':
            label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
        elif dataset == 'bcss':
            label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])

        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None) #此处加入了背景，做修改
        ##  "bg_score" is the white area generated by "cv2.threshold".
        ##  Since lungs are the main organ of the respiratory system. There are a lot of alveoli (some air sacs) serving for exchanging the oxygen and carbon dioxide, which forms some white background in WSIs.
        ##  For LUAD-HistoSeg, we uses it in the pseudo-annotation generation phase to avoid some meaningless areas to participate in the training phase of stage2.
        if dataset == 'luad':
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        ##  Since the white background of images of breast cancer is meaningful (e.g. fat, etc), we do not use it for the training set of BCSS-WSSS.
        elif dataset == 'bcss':
            bg_score = np.zeros((1,224,224))
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score) 
        visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(savepath, img_name+'.png'), format='PNG')

        if iter%100==0:           
            print(iter)
