import nibabel as nib
import numpy as np
import torch

from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as augmentor

def tensorize(*args):
    return tuple(torch.Tensor(arg).unsqueeze(0) for arg in args)

def augment(x1, y, x2=None):

    # NOTE: method expects numpy float arrays
    # to_pil_image makes assumptions based on input when mode = None
    # i.e. it should infer that mode = 'F'
    # manually setting mode to 'F' in this function

    # print(x.shape); exit()
    # NOTE: accepts np.ndarray of size H x W x C 
    # x.shape = 64x64
    # torch implicitly expands last dim as below:
    # elif pic.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        # pic = np.expand_dims(pic, 2)
    # BUT!!!!!!!
    # if x was a tensor this would be different:
    # elif pic.ndimension() == 2:
        # if 2D image, add channel dimension (CHW)
        # pic = pic.unsqueeze(0)

    angle = np.random.uniform(-180, 180)
    scale = np.random.uniform(.8, 1.2)
    shear = np.random.uniform(-30, 30)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 3)
    # ax.flat[0].imshow(x)

    x1 = augmentor.to_pil_image(x1, mode='F')
    if x2 is not None:
        x2 = augmentor.to_pil_image(x2, mode='F')

    # ax.flat[1].imshow(x)
    y = augmentor.to_pil_image(y, mode='F')

    x1 = augmentor.affine(x1, 
        angle=angle, translate=(0,0), shear=shear, scale=scale)
    if x2 is not None:
        x2 = augmentor.affine(x2, 
            angle=angle, translate=(0,0), shear=shear, scale=scale)
    y = augmentor.affine(y, 
        angle=angle, translate=(0,0), shear=shear, scale=scale)
    
    x1 = augmentor.to_tensor(x1).float()
    if x2 is not None:
        x2 = augmentor.to_tensor(x2).float()

    # ax.flat[2].imshow(x.squeeze(0))
    # plt.show(); exit()
    y = augmentor.to_tensor(y).float()
    y = (y > 0).float()

    # returns 1xHxW

    if x2 is not None:
        return x1,y,x2
    else:
        return x1,y

class Mixed(torch.utils.data.Dataset):

    def __init__(self, t1_paths, fl_paths, label_paths, 
        augment=False, num_fl=None, agg=False):

        # NOTE: if dataloader does not shuffle
        # and batch size is kept to 5, then 
        # each batch equates to a single subject

        t1_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in t1_paths for f in open(p) for _ in range(5)]
        fl_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in fl_paths for f in open(p) for _ in range(5)]
        lb_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in label_paths for f in open(p) for _ in range(5)]
        assert all([a == b and b == c for a,b,c in zip(t1_paths_order, 
            fl_paths_order, lb_paths_order)])
        # should match up so labels match, but will be randomized
        # so no inherent order to track
        self.order = None
        
        t1_paths = [f.strip() for p in t1_paths 
            for f in open(p).readlines()]
        fl_paths = [f.strip() for p in fl_paths 
            for f in open(p).readlines()]
        label_paths = [f.strip() for p in label_paths 
            for f in open(p).readlines()]
        
        t1_paths = [(p,l) for p,l in zip(t1_paths, label_paths)]
        fl_paths = [(p,l) for p,l in zip(fl_paths, label_paths)]
        
        # random subsamples of flair dataset to sim. missing data
        np.random.shuffle(fl_paths)
        if num_fl is not None:
            fl_paths = fl_paths[:num_fl]

        self.augment = augment
        self.t1_data = []
        self.fl_data = []

        for t1 in t1_paths:
            T1 = self._extract(t1[0])
            Y = self._extract(t1[1])[:, :, 
                (24,25,26,27,28)]
            for sl in range(Y.shape[2]):
                self.t1_data.append({
                    'data' : T1[:, :, sl], 
                    'label' : Y[:, :, sl]})

        for fl in fl_paths:
            FL = self._extract(fl[0])
            Y = self._extract(fl[1])[:, :, 
                (24,25,26,27,28)]
            for sl in range(Y.shape[2]):
                self.fl_data.append({
                    'data' : FL[:, :, sl], 
                    'label' : Y[:, :, sl]})
        
        self.agg = agg

        if self.agg:
            # randomly upsample flair here (for agg) 
            base_len = len(self.fl_data)
            while len(self.fl_data) != len(self.t1_data):
                index = int(np.random.randint(0, base_len))
                self.fl_data += [self.fl_data[index]]
            self.data = [x for x in (self.t1_data + self.fl_data)]
            self.kind = ['T1' for _ in self.t1_data] + ['FL' for _ in self.fl_data]
            self.epoch_len = len(self.data)
        else:
            self.epoch_len = len(self.t1_data)

    def __len__(self):
        return self.epoch_len
    
    def _extract(self, f):
        x = nib.load(f).get_fdata()
        return x.astype('float32')
    
    def __getitem__(self, index):

        if self.agg:
            x = self.data[index]['data']
            y = self.data[index]['label']
            if self.augment:
                x, y = augment(x, y)
            else:
                x, y = tensorize(x, y)
            return {'data' : x, 'label' : y, 'kind' : self.kind[index]}
        else:
            t1 = self.t1_data[index]['data']
            t1_label = self.t1_data[index]['label']
            # randomly upsample flair here (for meta)
            index = int(np.random.randint(0, len(self.fl_data)))
            fl = self.fl_data[index]['data']
            fl_label = self.fl_data[index]['label']
            if self.augment:
                t1, t1_label = augment(t1, t1_label)
                fl, fl_label = augment(fl, fl_label)
            else:
                t1,t1_label,fl,fl_label = \
                    tensorize(t1,t1_label,fl,fl_label)
            return {'T1' : t1, 'FL' : fl, 'T1_label' : t1_label,
                'FL_label' : fl_label}

class Paired(torch.utils.data.Dataset):

    def __init__(self, t1_paths, fl_paths, label_paths, 
        augment = False):

        # NOTE: if dataloader does not shuffle
        # and batch size is kept to 5, then 
        # each batch equates to a single subject

        t1_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in t1_paths for f in open(p) for _ in range(5)]
        fl_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in fl_paths for f in open(p) for _ in range(5)]
        lb_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in label_paths for f in open(p) for _ in range(5)]
        assert all([a == b and b == c for a,b,c in zip(t1_paths_order, 
            fl_paths_order, lb_paths_order)])
        self.order = lb_paths_order

        t1_paths = [f.strip() for p in t1_paths 
            for f in open(p).readlines()]
        fl_paths = [f.strip() for p in fl_paths 
            for f in open(p).readlines()]
        label_paths = [f.strip() for p in label_paths 
            for f in open(p).readlines()]
        
        paths = zip(t1_paths, fl_paths, label_paths)
        self.augment = augment
        self.data = []
        for t1_f, fl_f, label_f in paths:
            T1 = self._extract(t1_f)
            FL = self._extract(fl_f)
            Y = self._extract(label_f)[:, :, 
                (24,25,26,27,28)]
            for sl in range(Y.shape[2]):
                self.data.append({
                    'T1' : T1[:, :, sl], 
                    'FL' : FL[:, :, sl],
                    'label' : Y[:, :, sl]})

    def __len__(self):
        return len(self.data)
    
    def _extract(self, f):
        x = nib.load(f).get_fdata()
        return x.astype('float32')
    
    def __getitem__(self, index):
        t1 = self.data[index]['T1']
        fl = self.data[index]['FL']
        y = self.data[index]['label']
        if self.augment:
            t1,y,fl = augment(t1,y,fl)
        else:
            t1,y,fl = tensorize(t1,y,fl)
        return {'T1' : t1, 'FL' : fl, 'label' : y,
            'subject' : self.order[index]}

class Aggregate(torch.utils.data.Dataset):

    def __init__(self, data_paths, label_paths, augment = False):

        # NOTE: if dataloader does not shuffle
        # and batch size is kept to 5, then 
        # each batch equates to a single subject
        
        data_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in data_paths for f in open(p) for _ in range(5)]
        lb_paths_order = [f.strip().split('/')[-1].split('_')[0] 
            for p in label_paths for f in open(p) for _ in range(5)]
        assert all([a == b for a,b in zip( 
            data_paths_order, lb_paths_order)])
        self.order = lb_paths_order


        data_paths = [f.strip() for p in data_paths 
            for f in open(p).readlines()]
        label_paths = [f.strip() for p in label_paths 
            for f in open(p).readlines()]
        
        paths = zip(data_paths, label_paths)
        self.augment = augment
        self.data = []
        for data_f, label_f in paths:
            X = self._extract(data_f)
            Y = self._extract(label_f)[:, :, 
                (24,25,26,27,28)]
            for sl in range(Y.shape[2]):
                self.data.append({
                    'data' : X[:, :, sl], 
                    'label' : Y[:, :, sl]})

    def __len__(self):
        return len(self.data)
    
    def _extract(self, f):
        x = nib.load(f).get_fdata()
        return x.astype('float32')
    
    def __getitem__(self, index):
        x = self.data[index]['data']
        y = self.data[index]['label']
        if self.augment:
            x,y = augment(x,y)
        else:
            x,y = tensorize(x,y)
        return {'data' : x, 'label' : y,
            'subject' : self.order[index]}
        
DATASETS = {
    'aggregate' : Aggregate,
    'paired' : Paired,
    'mixed' : Mixed
}
