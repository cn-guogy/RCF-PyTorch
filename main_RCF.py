import os
import numpy as np
import os.path as osp
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from models import RCF


class CustomDataset(Dataset):
    """自定义数据集，用于加载和处理自己的数据"""
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = sorted([file for file in os.listdir(image_dir) if file.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = osp.join(self.image_dir, self.image_list[idx])
        image = cv2.imread(image_path).astype(np.float32)
        image -= np.array([104.00698793, 116.66876762, 122.67891434])  # 减去均值
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        return torch.tensor(image), self.image_list[idx]


def single_scale_test(model, test_loader, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, (image, filename) in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        all_res = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            all_res[i, 0, :, :] = results[i]
        torchvision.utils.save_image(1 - all_res, osp.join(save_dir, '%s.jpg' % filename[0]))
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ss.png' % filename[0]), fuse_res)
    print('Single-scale test completed.')


def multi_scale_test(model, test_loader, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, (image, filename) in enumerate(test_loader):
        in_ = image[0].numpy().transpose((1, 2, 0))
        _, _, H, W = image.shape
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)
        ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ms.png' % filename[0]), ms_fuse)
    print('Multi-scale test completed.')


# 设置路径
base_dir = "/home/guogy/dataset/DAVSOD"  # 主目录，包含多个视频帧子目录
checkpoint_path = "/home/guogy/RCF-PyTorch/bsds500_pascal_model.pth"  # 模型权重文件路径

# 加载模型
model = RCF().cuda()
if osp.isfile(checkpoint_path):
    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print("=> Checkpoint loaded successfully")
else:
    raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

# 遍历每个子目录并处理
for video_dir in sorted(os.listdir(base_dir)):
    video_path = osp.join(base_dir, video_dir, "Imgs")  # 假设图片在每个子目录的 Imgs 文件夹下
    if not osp.isdir(video_path):
        continue

    print(f"Processing video frames in '{video_path}'")
    
    # 设置保存路径到 edge 文件夹下
    save_dir = osp.join(base_dir, video_dir, "edge")
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    
    # 加载当前子目录数据集
    test_dataset = CustomDataset(image_dir=video_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    
    # 执行测试
    print(f"Running single-scale test for '{video_dir}'...")
    single_scale_test(model, test_loader, save_dir)
    
    print(f"Running multi-scale test for '{video_dir}'...")
    multi_scale_test(model, test_loader, save_dir)

print("All videos have been processed!")
