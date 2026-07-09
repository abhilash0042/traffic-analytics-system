import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True) 
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True) 
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True) 
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True) 
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True) 
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True) 
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
        
        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + r6*(torch.pow(x,2)-x)	
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return enhance_image_1, enhance_image, r

class ZeroDCEEnhancer:
    def __init__(self, weights_path="weights/zero_dce.pth", device='cuda'):
        self.device = device if torch.cuda.is_available() and device != 'cpu' else 'cpu'
        self.weights_path = weights_path
        self._ensure_weights_exist()
        self.model = self._load_model()
        self.model.eval()
    
    def _ensure_weights_exist(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        if not os.path.exists(self.weights_path):
            print(f"Downloading Zero-DCE weights to {self.weights_path}...")
            try:
                # Use public github content URL from official repository
                urllib.request.urlretrieve("https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth", self.weights_path)
                print("Downloaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to auto-download weights ({e}). Please manually download Epoch99.pth from Zero-DCE repo and place it at {self.weights_path}")

    def _load_model(self):
        # Convert integer device to a proper torch.device object to prevent torch.load map_location TypeError
        torch_device = torch.device(self.device) if isinstance(self.device, (int, str)) else self.device
        net = enhance_net_nopool().to(torch_device)
        try:
            net.load_state_dict(torch.load(self.weights_path, map_location=torch_device))
        except FileNotFoundError:
            print(f"Failed to load weights from {self.weights_path}. Check if file exists.")
        return net
    
    def enhance(self, img_bgr):
        # Resize logic if needed, but Zero-DCE handles arbitrary sizes up to memory limit
        # Resize to multiple of 4 to prevent shape issues
        h, w = img_bgr.shape[:2]
        
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, enhanced, _ = self.model(img_tensor)
        
        enhanced = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        return enhanced_bgr

def is_dark_frame(frame, threshold=60):
    """Simple heuristic to detect dark frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold
