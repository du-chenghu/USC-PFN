import os.path as osp
import os

class parser:
    def __init__(self):
        self.dataroot = 'dataset'
        self.datamode = 'test'                                  # train, test
        self.stage = 'SIG'                                      # NGD, SIG
        self.runmode = self.datamode                              
        self.name = self.stage   
        if self.datamode == 'train':
            self.data_list = 'train_pairs.txt'
        elif self.datamode == 'test':
            # self.data_list = 'measure.txt'                    # paired list
            self.data_list = 'test_pairs.txt'                   # unpaired list
        self.fine_width = 192
        self.fine_height = 256
        self.radius = 4
        # self.grid_path =  osp.join(self.dataroot, 'grid.png')
        if self.datamode == 'train':    #for training keep true, for test keep false
            self.shuffle = True
            self.batch_size = 16          
        else:
            self.shuffle = False
            self.batch_size = 1
        
        self.workers = 8
        self.grid_size = 5
        
        self.lr = 0.0001            # learning rate
        self.keep_epoch = 50        
        self.decay_epoch = 50      
        self.previous_epoch = 0      #if you want to resume training from some steps 
                                      #set previous_step in as per last updated checkpoints                                
        self.save_count = 1
        
        self.checkpoint_dir = osp.join(os.getcwd(), 'checkpoints')
        self.save_dir = osp.join(os.getcwd(), 'outputs')         #for saving output while infernce
        if self.previous_epoch == 0:
            self.checkpoint = ''
        else:
            self.checkpoint = osp.join(self.checkpoint_dir, self.name, 'epoch_%03d.pth' % (self.previous_epoch))



        
