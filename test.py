import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import os.path as osp
import warnings
from tqdm import tqdm
from utils.visualization import load_checkpoint, save_images
from data.data_reader import DGDataset, DGDataLoader
from config import parser
from models.networks import NGD, SIG
from utils.flow_util import flow2color
from utils.losses import flow_warping
torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def funcTryOn(opt, test_loader, model):
    print('----Testing of module {} started----'.format(opt.name))
    model.to(device)
    model.eval()
    
    length = len(test_loader.data_loader)
    step = 0
    pbar = tqdm(total=length)
    
    MRF = NGD(6, 2)
    NGDPath = osp.join(opt.checkpoint_dir, 'NGD', 'epoch_%03d.pth' % (200))
    load_checkpoint(MRF, NGDPath)
    MRF.to(device)
    MRF.eval()
    
    while inputs is not None:
        im_name = inputs['im_name']
        im = inputs['image'].to(device)
        c = inputs['cloth'].to(device)
        # ++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            MRF_flow = MRF(torch.cat([c, im], 1))
            warped_cloth, _ , = flow_warping(c, MRF_flow)
            p_tryon = torch.tanh(model(torch.cat([im, warped_cloth], 1)))
            
        a = im
        b = c
        flow_offset = de_offset(MRF_flow)
        flow_color = flow2color()(flow_offset).cuda()
        c= warped_cloth
        d = p_tryon
        combine = torch.cat([a[0],b[0], flow_color, c[0], d[0]], 2).unsqueeze(0)
      
        save_images(c, im_name, osp.join(opt.save_dir, opt.datamode, 'ClothImg'))
        save_images(im, im_name, osp.join(opt.save_dir, opt.datamode, 'PersonImg'))
        save_images(p_tryon, im_name, osp.join(opt.save_dir, opt.datamode, 'TryOnResults'))
        save_images(combine, im_name, osp.join(opt.save_dir, opt.datamode, 'CResults'))

        inputs = test_loader.next_batch()
        step+=1
        pbar.update(1)
        
def de_offset(s_grid):
    [b,_,h,w] = s_grid.size()

    x = torch.arange(w).view(1, -1).expand(h, -1).float()
    y = torch.arange(h).view(-1, 1).expand(-1, w).float()
    x = 2*x/(w-1)-1
    y = 2*y/(h-1)-1
    grid = torch.stack([x,y], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    offset = grid - s_grid
    offset_x = offset[:,0,:,:] * (w-1) / 2
    offset_y = offset[:,1,:,:] * (h-1) / 2
    offset = torch.cat((offset_y,offset_x),0)
    return  offset

def main():
    opt = parser()
    test_dataset = DGDataset(opt)
    # create dataloader
    test_loader = DGDataLoader(opt, test_dataset)
    model = SIG(6, 3)
    checkpoint_path = osp.join(opt.checkpoint_dir, 'SIG', 'epoch_%03d.pth' % (170))
    load_checkpoint(model, checkpoint_path)
    funcTryOn(opt, test_loader, model)

if __name__ == '__main__':
    main()
