#Importing libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

#Setting batch size
batch_size=32

#Setting dataset
norm_dataset = datasets.ImageFolder(root="image-classification-cnn-vs-resnet/data/processed/train", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))

#Setting up the loader, in GPU if it is available or CPU if not
if torch.cuda.is_available():
    device = torch.device("cuda")
    norm_loader = DataLoader(norm_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
else:
    device = torch.device("cpu")
    norm_loader = DataLoader(norm_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

#Compute mean and std
mean=0.
std=0.
total_images=0
for images,_ in norm_loader:                                                #images is a tensor with dimensions [B, C, H, W] = [batch size, color channels (RGB), height, width]
    images = images.to(device, non_blocking=True)                           #Move images to GPU if posible
    batch_samples = images.size(0)                                          #Number of images in each bach. .size() gives [B, C, H, W]
    images = images.view(batch_samples,images.size(1),-1)                   #Resize [B,C,H,W] size tensor to [B,C,H*W] size
    mean += images.mean(2).sum(0)                                           #Get tensor mean of pixel values (H*W) and add the mean of each image of the batch -> mean=[mean_R, mean_G, mean_B]
    std += images.std(2).sum(0)                                             #Same for std
    total_images += batch_samples

mean /= total_images                                                        #Mean of pixels values of images
std /= total_images                                                         #Std of pixels values of images

print(mean,std)


#After executing the mean and std are
#tensor([0.4963, 0.4963, 0.4894]) tensor([0.2304, 0.2314, 0.2542])