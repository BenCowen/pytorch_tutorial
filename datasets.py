"""
Creates datasets and dataloaders for various torchvision classes.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from random import shuffle

def load_dataset(namedataset='mnist', batch_size=200,
                  vectorize=False, num_workers=1, valid_size=-1,
                 data_augmentation=False, noise_sigma=0.0,):
    '''
    Arguments:
      namedataset: see below
      batch_size : number samples per batch
      vectorize: flattens input tensors if True
      num_workers: number of GPUs for parallel training
      valid_size : number of samples for validation set; if <=0, returns test set
      (EXAMPLES:)
        noise_sigma: adds noise only to the test set
          (only implemented for noisy_mnist as an example)
        data_augmentation: use data augmentation if True
          (only implemented for cifar10 as an example)

    datasets:
      mnist: standard mnist
      valid_mnist: returns validation set instead of test set
      noisy_mnist: EXPERIMENTAL: adds noisy only to the test set.
      fashion_mnist
      valid_fashion_mnist
      cifar10
    '''

    # Load mnist dataset
    if namedataset == 'mnist':

        DIR_DATASET = '~/data/mnist'

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]

        if vectorize:
            transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1)*x.size(2))))

        transform = transforms.Compose(transform_list)

        trainset = datasets.MNIST(DIR_DATASET, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = datasets.MNIST(DIR_DATASET, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        classes = tuple(range(10))
        n_inputs = 784
    elif namedataset == 'valid_mnist':

        if valid_size < 1:
          raise ValueError('Validation requested with validation size = {}'.format(valid_size))

        DIR_DATASET = '~/data/mnist'

        # Define the transforms applied to every sample.
        # Desired mean and Std Dev of data:
        MEAN = 0.1307
        STD  = 0.3081
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))]
        if vectorize:
            transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1)*x.size(2))))
        transform = transforms.Compose(transform_list)

        # Now load training data and make two dataloaders (train/validation) for it.
        allTrainData = datasets.MNIST(DIR_DATASET, train=True, download=True, transform=transform)
        num_train = len(allTrainData)
        indices   = list(range(num_train))
        # First shuffle the indices:
        shuffle(indices)
        # Assign sampled before split to train; after split to validation.
        split     = num_train - valid_size
        print('nTrain='+str(num_train))
        print('validsize='+str(valid_size))
        print('split='+str(split))
        train_idx, valid_idx = indices[:split], indices[split:]
        # Random samplers for the relevant indices only.
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Finally, instantiate the data loaders.
        train_loader = torch.utils.data.DataLoader(allTrainData,
                                                  sampler     = train_sampler,
                                                  batch_size  = batch_size,
                                                  num_workers = num_workers)
        train_loader.numSamples=len(train_idx)
        # THIS IS THE VALIDATION SET:
        test_loader = torch.utils.data.DataLoader(allTrainData,
                                                  sampler     = valid_sampler,
                                                  batch_size  = batch_size,
                                                  num_workers = num_workers)
        test_loader.numSamples=len(valid_idx)

        print('train size = '+str(train_loader.numSamples))
        print('valid size = '+str(test_loader.numSamples))
        classes = tuple(range(10))
        n_inputs = 784

    elif namedataset == 'noisy_mnist':

        DIR_DATASET = '~/data/mnist'

        # Desired mean and Std Dev of data:
        MEAN = 0.1307
        STD  = 0.3081

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))]
        if vectorize:
            transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1)*x.size(2))))

        transform = transforms.Compose(transform_list)

        trainset = datasets.MNIST(DIR_DATASET, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Add Gaussian noise to the test set.
        transform_list.append(transforms.Lambda(lambda x: x + MEAN
                                                 + noise_sigma*STD*torch.randn(x.size())))

        testset = datasets.MNIST(DIR_DATASET, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        classes = tuple(range(10))
        n_inputs = 784

    # Load mnist dataset
    elif namedataset == 'fashion_mnist':

        DIR_DATASET = '~/data/fashion_mnist'

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]

        if vectorize:
            transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1)*x.size(2))))

        transform = transforms.Compose(transform_list)

        trainset = datasets.FashionMNIST(DIR_DATASET, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = datasets.FashionMNIST(DIR_DATASET, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        classes = tuple(range(10))
        n_inputs = 784

    elif namedataset == 'valid_fashion_mnist':

        if valid_size < 1:
          raise ValueError('Validation requested with validation size = {}'.format(valid_size))

        DIR_DATASET = '~/data/fashion_mnist'

        # Define the transforms applied to every sample.
        # Desired mean and Std Dev of data:
        MEAN = 0.1307
        STD  = 0.3081
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))]

        if vectorize:
            transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1)*x.size(2))))

        transform = transforms.Compose(transform_list)

        # Now load training data and make two dataloaders (train/validation) for it.
        allTrainData = datasets.FashionMNIST(DIR_DATASET, train=True, download=True, transform=transform)
        num_train = len(allTrainData)
        indices   = list(range(num_train))
        # First shuffle the indices:
        shuffle(indices)
        # Assign sampled before split to train; after split to validation.
        split     = num_train - valid_size
        print('nTrain='+str(num_train))
        print('validsize='+str(valid_size))
        print('split='+str(split))
        train_idx, valid_idx = indices[:split], indices[split:]
        # Random samplers for the relevant indices only.
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Finally, instantiate the data loaders.
        train_loader = torch.utils.data.DataLoader(allTrainData,
                                                  sampler     = train_sampler,
                                                  batch_size  = batch_size,
                                                  num_workers = num_workers)
        train_loader.numSamples=len(train_idx)
        # THIS IS THE VALIDATION SET:
        test_loader = torch.utils.data.DataLoader(allTrainData,
                                                  sampler     = valid_sampler,
                                                  batch_size  = batch_size,
                                                  num_workers = num_workers)
        test_loader.numSamples=len(valid_idx)

        print('train size = '+str(train_loader.numSamples))
        print('valid size = '+str(test_loader.numSamples))
        classes = tuple(range(10))
        n_inputs = 784

    # Load cifar10 (preprocessing from https://github.com/kuangliu/pytorch-cifar)
    elif namedataset == 'cifar10':

        DIR_DATASET = '~/data/cifar10'

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

        if vectorize:
            transform_list.append(
                transforms.Lambda(lambda x: x.view(x.size(0)*x.size(1)*x.size(2))))

        transform_test = transforms.Compose(transform_list)

        if data_augmentation:
            transform_train_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

            if vectorize:
                transform_train_list.append(
                    transforms.Lambda(lambda x: x.view(x.size(0)*x.size(1)*x.size(2))))

            transform_train = transforms.Compose(transform_train_list)

        else:
            transform_train = transform_test

        trainset = datasets.CIFAR10(DIR_DATASET, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = datasets.CIFAR10(DIR_DATASET, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        n_inputs = 3*32*32

    else:
        raise ValueError('Dataset {} not recognized'.format(namedataset))

    return train_loader, test_loader, n_inputs, len(classes)
