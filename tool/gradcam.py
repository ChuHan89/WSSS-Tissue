import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms
import importlib
from tool import infer_utils
from PIL import Image
import torch.nn.functional as F

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        # self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            # print(name)
            
            if module == self.feature_module:
                # target_activations, x = self.feature_extractor(x)
                x = module(x)
                if name == "bn7":
                    x = F.relu(x)
                x.register_hook(self.save_gradient)
                target_activations += [x]
           
            else:
                x = module(x)
                if name == "fc8":
                    x = x.view(x.size(0),-1)
                if name == "bn7":
                    x = F.relu(x)

        return target_activations, x #指定层的输出和网络最终的输出

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_BONE)
    heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    '''
        grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)
    '''
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img) #指定层的输出和网络最终的输出

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, input_img.shape[2:])##
        return cam, output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='the_state_of_CAMs_in_training_process/img_1001.png',
                        help='Input image path')
    args = parser.parse_args()
    
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.use_cuda = False
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    for k in range(7):
        args.image_path = 'choose/exp'+str(k)+'/img.png'
        import network.resnet38d
        
        for name_i in range(3,20):
            model = 0
            weights_dict = 0
            weights_dict = torch.load('stage1_save/checkpoints_and_visdom_logs/Baseline_weight_Of_epoch'+str(name_i)+'.pth')
            
            model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(1, 0, 4)
            model.load_state_dict(weights_dict)

            grad_cam = GradCam(model=model, feature_module=model.bn7, \
                            target_layer_names=["1"], use_cuda=args.use_cuda)

            img = cv2.imread(args.image_path, 1)
            img = np.float32(img) / 255
            img = img[:, :, ::-1]
            input_img = preprocess_image(img)
                
            cam = Image.new('RGB', (224,224))
            for i in range(4):
                ##
                target_category = i
                grayscale_cam,_ = grad_cam(input_img, target_category)

                grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
                cam_show = show_cam_on_image(img, grayscale_cam)
                
                ##
                cam_show = Image.fromarray(cv2.cvtColor(cam_show,cv2.COLOR_BGR2RGB))
                cam_show.save("choose/exp"+str(k)+"/Baseline_cam"+str(i)+'_epoch_'+str(name_i)+".jpg")
                test = 1
            # cv2.imwrite("the_state_of_CAMs_in_training_process/cam_"+str(i)+'_epoch_'+str(name_i)+".jpg", cam)
    # cam = np.array(cam)
    # print(cam.shape)

    # image_path = args.image_path
    # orig_img = np.asarray(Image.open(image_path))
    # L = image_path.split('/')[-1].split('.')[0]
    # label = torch.Tensor([int(L[1]),int(L[3]),int(L[5]),int(L[7])])
    # cam_dict = infer_utils.cam_npy_to_cam_dict(cam, label)
    # cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None) 
    # bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
    # seg_map = infer_utils.cam_npy_to_label_map(bgcam_score) 

    # palette = [0]*768
    # palette[0:3] = [192, 128, 128]
    # palette[3:6] = [128, 64, 0]
    # palette[6:9] = [0, 64, 128]
    # palette[9:12] = [64, 128, 0]
    # palette[12:15] = [224, 224, 192]

    # visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
    # visualimg.putpalette(palette) 
    # visualimg.save(image_path.split('.')[0]+'mask.png', format='PNG')

