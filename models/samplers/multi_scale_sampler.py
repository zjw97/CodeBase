import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler,RandomSampler,SequentialSampler

class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None, img_sizes=None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = 416
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                print(batch)
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0:
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    class my_dataset(Dataset):
        def __init__(self,input_size = 416):
            super(my_dataset,self).__init__()
            self.input_size =  input_size

        def __len__(self):
            return 10000

        def __getitem__(self, item):
            if type(item) == list or type(item) == tuple:
                index,input_size = item
            else:
                index,input_size = item,self.input_size

            return index,input_size


dataset = my_dataset()

loader_random_sample = DataLoader(dataset=dataset,
                    batch_sampler= BatchSampler(RandomSampler(dataset),
                                 batch_size=10,
                                 drop_last=True,
                                 multiscale_step=1,
                                 img_sizes=list(range(320, 608 + 1, 32))),
                    num_workers=8)

loader_sequential_sample = DataLoader(dataset=dataset,
                    batch_sampler=BatchSampler(SequentialSampler(dataset),
                                 batch_size=10,
                                 drop_last=True,
                                 multiscale_step=1,
                                 img_sizes=list(range(320, 608 + 1, 32))),
                    num_workers=8)

for batch in loader_random_sample:
    # print(batch)
    z = 1
'''random sample
[tensor([ 400, 5006, 9921, 3756, 2826, 6156, 8680, 9827, 4837, 5829]), 
tensor([416, 416, 416, 416, 416, 416, 416, 416, 416, 416])]
[tensor([7319, 4863, 4002, 4321,  838,  736, 9295, 2537, 4451,  492]),
 tensor([352, 352, 352, 352, 352, 352, 352, 352, 352, 352])]
'''
# for batch in loader_sequential_sample:
#     print(batch)
'''sequential sample
[tensor([8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917, 8918, 8919]), 
tensor([544, 544, 544, 544, 544, 544, 544, 544, 544, 544])]
[tensor([8920, 8921, 8922, 8923, 8924, 8925, 8926, 8927, 8928, 8929]), 
tensor([352, 352, 352, 352, 352, 352, 352, 352, 352, 352])]
'''