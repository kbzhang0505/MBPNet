
import torch.utils.data
from data.ours_dataset import OursDataset

#Creating dataset
def create_dataset(opt):

    dataset = OursDataset(opt)  # Initialize add configuration (data path list, clipping and resize parameters)

    print("dataset [%s] was created" % type(dataset).__name__)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads))  # Create a multithreaded data loader
    return dataloader

