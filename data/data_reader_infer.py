import numpy as np
import os.path as osp
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from config import parser

class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode 
        self.data_list = opt.data_list
        self.data_path = osp.join(opt.dataroot, opt.datamode) 
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])
        
        # load data list
        im_names = []
        c_names = []
            
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names
    
    def name(self):
        return "VITONDataset for test."

    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = self.c_names[index]
        
        # cloth image
        c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        c = self.transform(c)  # [-1,1]

        # cloth mask
        cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array).unsqueeze(0) # [0,1]

        # person image 
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]

        results = {
            'c_name':       c_name,     # for visualization
            'im_name':      im_name,    # for visualization
            'cloth':        c,          # for input
            'image':        im,         # for input
            'c_mask':    cm,            # for input
            }

        return results

    def __len__(self):
        return len(self.im_names)

class VITONDataLoader(object):
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()
        self.batch_size = opt.batch_size
        self.runmode = opt.runmode
        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            if self.runmode == 'train':
                self.data_iter = self.data_loader.__iter__()
                batch = self.data_iter.__next__()
            if self.runmode == 'test' :
                batch = None
        return batch
    
    def load_data(self):
        return self
    
    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.data_loader):
            if i * self.batch_size >= float("inf"):
                break
            yield data


def main():
    print("Check the dataset!")

    opt = parser()
    dataset = VITONDataset(opt)
    # create dataloader
    data_loader = VITONDataLoader(opt, dataset)
    

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

if __name__ == "__main__":
    main()
