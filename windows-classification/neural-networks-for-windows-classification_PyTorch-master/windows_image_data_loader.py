import torch
from torchvision import datasets, transforms
import os

def windows_image_data_loader(data, args):


    # Batch Sizes for dataloaders
    train_batch_size = validation_batch_size = 16  

    train_root = os.path.join(data, 'train')  # this is path to training images folder
    validation_root = os.path.join(data, 'val')  # this is path to validation images folder
    test_root = os.path.join(data, 'test')  # this is path to test images folder

    # The numbers are the mean and std provided in PyTorch documentation to be used for models pretrained on
    # ImageNet data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Create training dataset after applying data augmentation on images
    train_data = datasets.ImageFolder(train_root,
                                        transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                    transforms.ToTensor(),
                                                                    normalize]))
    # Create validation dataset after resizing images
    validation_data = datasets.ImageFolder(validation_root,
                                            transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                            transforms.ToTensor(),
                                                                            normalize]))
    # Create validation dataset after resizing images
    test_data = datasets.ImageFolder(test_root,
                                            transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                            transforms.ToTensor(),
                                                                            normalize]))    


    # Create training dataloader
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                                            num_workers=5)
    # Create validation dataloader
    validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                                batch_size=validation_batch_size,
                                                                shuffle=False, num_workers=5)
    # Create test dataloader
    test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=validation_batch_size,
                                                                shuffle=False, num_workers=5)        



    return train_data_loader, validation_data_loader, test_data_loader
