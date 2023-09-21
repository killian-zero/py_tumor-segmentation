import argparse
import torch
import os
import torch.nn as nn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
from torchvision import transforms
import gzip
import pickle
from PIL import Image
from tqdm import tqdm
from typing import Optional
import math
from PIL import Image
import PIL
import cv2
from matplotlib import pyplot as plt
from VIT import TransformerModel
#from medpy.metric.binary import obj_asd,asd,hd,dc,assd,jc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from medpy.metric.binary import obj_asd,asd,hd,dc,assd,jc
#from pytorch_grad_cam import GradCAM
import pandas as pd
import axialtrans
from axialtrans import MedT128, MedT64


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch dcm training")

    parser.add_argument("--data-path", default="/home/ma-user/work/mri1/", help="root")
    parser.add_argument("--train-rates", default=0.6, type=float, help='train datasets rate')
    parser.add_argument("--trainval-rates", default=0.8, type=float, help='val datasets rate')
    parser.add_argument("--show-number", default=4, type=int, help='show datalodater figures numbers')
    parser.add_argument("--resize", default=256, type=int, help="resize fig")
    parser.add_argument('--save_freq', type=int, default=40)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=41, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')

    args = parser.parse_args(args=[])

    return args



def read_split_data():
    assert os.path.exists(args.data_path), "dataset root: {} does not exist.".format(args.data_path)
    
    datasets = [cla for cla in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, cla))]
    datasets.sort()
    supported = [".jpg", ".JPG", ".png", ".PNG", ".gz"]  # 支持的文件后缀类型

    cla_image = datasets[0]  # datasets[0] = image
    cla_label = datasets[1]  # datasets[1] = label
    cla_path = os.path.join(args.data_path, cla_image)
    cla_label_path = os.path.join(args.data_path, cla_label)
    
    # 获取images和labels的路径
    images = [os.path.join(args.data_path, cla_image, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
    labels = [os.path.join(args.data_path, cla_label, i) for i in os.listdir(cla_label_path) if os.path.splitext(i)[-1] in supported]
    
    # 划分训练集与验证集与测试集
    train_images_path = images[0:int(len(images)*args.train_rates)]
    train_images_label = labels[0:int(len(labels)*args.train_rates)]
    #val_images_path = images[int(len(images)*args.train_rates):]
    val_images_path = images[int(len(images)*args.train_rates):int(len(images)*args.trainval_rates)]
    #val_images_label = labels[int(len(labels)*args.train_rates):]
    val_images_label = labels[int(len(labels)*args.train_rates):int(len(images)*args.trainval_rates)]
    test_images_path = images[int(len(images)*args.trainval_rates):]
    test_images_label = labels[int(len(images)*args.trainval_rates):]
    
    print('train numbers:',len(train_images_path))
    print('val numbers:',len(val_images_path))
    print('test numbers:',len(test_images_path))

    return train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label


def read_image_label(image_path,label_path):
    
    with gzip.open(image_path, 'rb') as unzipped_data:
        image = pickle.load(unzipped_data)
    with gzip.open(label_path, 'rb') as unzipped_data:
        label = pickle.load(unzipped_data)
    image = np.expand_dims(image,axis=0)
    label = np.expand_dims(label,axis=0)
    
    return image,label


def show_loader(train_loader):
    img,label=next(iter(train_loader))
    print('img shape:',img.shape)
    print('label shape:',label.shape)
    print('imge_max and imge_min:',img.max(), img.min())
    print('label_max and label_min:',label.max(), label.min())
    print(img.dtype)
    print(label.dtype)
    plt.figure(figsize=(12,8))
    for i,(img,label) in enumerate(zip(img[:args.show_number],label[:args.show_number])):
        img = np.squeeze(img)
        label = np.squeeze(label)
        img = np.array(img)
        img = PIL.Image.fromarray(img)
        plt.subplot(2,4,i+1)
        plt.imshow(img,cmap='gray')
        plt.subplot(2,4,i+5)
        plt.imshow(label,cmap='gray')


def main_dataloader():
    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data()
    
    data_transform = {"train": transforms.Compose([transforms.ToTensor(),transforms.Resize((512, 512))]),
                  "val": transforms.Compose([transforms.ToTensor(),transforms.Resize((512, 512))]),
                     "test": transforms.Compose([transforms.ToTensor(),transforms.Resize((512, 512))])}
    
    train_data=MyDataSet(train_images_path,train_images_label,data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               collate_fn=train_data.collate_fn)
    
    val_data=MyDataSet(val_images_path,val_images_label,data_transform['val'])
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               collate_fn=train_data.collate_fn)
    test_data=MyDataSet(test_images_path,test_images_label,data_transform['test'])
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               collate_fn=train_data.collate_fn)
    return train_loader, val_loader, test_loader


# In[4]:


class MyDataSet(torch.utils.data.Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_label: list, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image, label = read_image_label(self.images_path[item], self.images_label[item])

        #         image_min = np.min(image, axis=(1,2), keepdims=True)
        #         image_max = np.max(image, axis=(1,2), keepdims=True)
        #         image = (image - image_min) / (image_max - image_min)

        if image.shape[0] == 1:
            image = image[0]
        img_open = Image.fromarray(image)
        img = self.transform(img_open)
        img = img.type(torch.FloatTensor)

        if label.shape[0] == 1:
            label = label[0]
        label_open = Image.fromarray(label)
        label = self.transform(label_open)
        label = label.type(torch.FloatTensor)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels


class LearnedPositionalEncoding0(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding0, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, 2048)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

class PatchEmbed0(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_c=128, embed_dim=2048, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net, self).__init__()
        n1 = 8
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1*16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.patchembed0 = PatchEmbed0()
        #self.patchembed1 = PatchEmbed1()
        #self.patchembed2 = PatchEmbed2()
        #self.patchembed3 = PatchEmbed3()
        self.pos_embed0 = LearnedPositionalEncoding0(64,2048,64)
        #self.pos_embed1 = LearnedPositionalEncoding1(64,4096,64)
        #self.pos_embed2 = LearnedPositionalEncoding2(256,2048,256)
        #self.pos_embed3 = LearnedPositionalEncoding3(256,4096,256)
        self.vit = TransformerModel(2048, 4, 8, 8192)
        #self.ffm1 = TransformerModel(4096, 4, 16, 8192)
        #self.ffm2 = TransformerModel(2048, 4, 8, 4096)
        #self.ffm3 = TransformerModel(4096, 4, 16, 8192)
        self.ffm1 = MedT128()
        #self.ffm2 = MedT64()
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()
    def _reshape_output0(self, x):
        x = x.view(
            x.size(0),
            32,
            32,
            128,
        )
        x = x.permute(0, 3, 1, 2).contiguous()

        return x



    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        #c1 = self.patchembed3(e2)
        #c1 = self.pos_embed3(c1)
        #c1 = self.ffm3(c1)
        #c1 = c1[0]
        #c2 = self._reshape_output3(c1)
        #print(c2.shape)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        c3 = self.ffm1(e3)
        #c3 = self.patchembed2(e3)
        #c3 = self.pos_embed2(c3)
        #c3 = self.ffm2(c3)
        #c3 = c3[0]
        #c4 = self._reshape_output2(c3)
        #print(c4.shape)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        #c4 = self.ffm2(e4)
        #c5 = self.patchembed1(e4)
        #c5 = self.pos_embed1(c5)
        #c5 = self.ffm1(c5)
        #c5 = c5[0]
        #c6 = self._reshape_output1(c5)
        #print(c6.shape)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e6 = self.patchembed0(e5)
        e7 = self.pos_embed0(e6)
        e8 = self.vit(e7)
        e8 = e8[0]
        e9 = self._reshape_output0(e8)
        #print(e9.shape)
        #e5 = self.Conv5(e5)

        d5 = self.Up5(e9)
        d5 = torch.cat((e4, d5), dim=1)
        #print(d5.shape)
        d5 = self.Up_conv5(d5)
        #print(d5.shape)
        d4 = self.Up4(d5)
        d4 = torch.cat((c3, d4), dim=1)
        #print(d4.shape)
        d4 = self.Up_conv4(d4)
        #print(d4.shape)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        #print(d3.shape)
        d3 = self.Up_conv3(d3)
        #print(d3.shape)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        #print(d2.shape)
        d2 = self.Up_conv2(d2)
        #print(d2.shape)

        out = self.Conv(d2)
        #print(out.shape)
        return out

def get_model_and_optimizer():
    model = U_Net()                
    model = model.to('cuda:0')
    optimizer=torch.optim.Adagrad(model.parameters(),lr=0.01)
    return model,optimizer

class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pre, target)
        
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        logp = ce(pre, target)
        p = torch.exp(-logp)
        fl = ((1 - p) ** 1 * logp).mean()
        
        
        smooth = 1e-5
        pre = torch.sigmoid(pre)
        num = target.size(0)
        pre = pre.view(num, -1)
        target = target.view(num, -1)
        intersection = (pre * target)
        dice = (2. * intersection.sum(1) + smooth) / (pre.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + 0.5*fl + dice

BCEDiceLoss = nn.BCEWithLogitsLoss()

# In[7]:


def fit(model, trainloader, valloader):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for epoch in range(1, args.epochs):
        model.train()
        train_loss = 0
        train_loader = tqdm(trainloader)
        for step, data in enumerate(train_loader):
            image, target = data
            image, target = image.to(device), target.to(device)
            #target = target.to(device=device, dtype=torch.int64)
            output_seg = model(image)
            loss = BCEDiceLoss(output_seg, target.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader.desc = "[train epoch {}] loss: {:.5f}".format(epoch,loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loader = tqdm(valloader)
            for step, data in enumerate(val_loader):
                image, target = data
                image, target = image.to(device), target.to(device)
                output_seg = model(image)
                loss = BCEDiceLoss(output_seg, target.float())
                val_loss += loss.item()
                val_loader.desc = "[val epoch {}] loss: {:.5f}".format(epoch,loss)
                
        epoch_train_loss = train_loss / len(trainloader.dataset)
        epoch_val_loss = val_loss / len(valloader.dataset)
        print('epoch: ', epoch,
              'train_loss：', round(epoch_train_loss, 5),
              'val_loss：', round(epoch_val_loss, 5))
        if (epoch % args.save_freq) ==0:
            #torch.save(model.state_dict(),
                   #f'/root/files/weight/{epoch}_T{round(epoch_train_loss, 5)}_V{round(epoch_val_loss, 5)}.pth')

            save_file = {"model": model.state_dict(),"optimizer": optimizer.state_dict(),"epoch": epoch,"args": args}
            torch.save(save_file, f'/home/ma-user/work/weight/lossfunction/mri/{epoch}_T{round(epoch_train_loss, 5)}_V{round(epoch_val_loss, 5)}.pth')

def model_load(path):
    checkpoint = torch.load(path)
    model = U_Net()
    model = model.to('cuda:0')
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adagrad(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


args = parse_args()
train_loader, val_loader, test_loader = main_dataloader()
#model, optimizer = get_model_and_optimizer()
model, optimizer = model_load('/home/ma-user/work/weight/lossfunction/60_T0.00039_V0.00043.pth')
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))
fit(model, train_loader, val_loader)

#model, optimizer = model_load('/root/files/weight/98_T0.10681_V0.23522.pth')





