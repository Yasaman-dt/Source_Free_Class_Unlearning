# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from models.allcnn import AllCNN
import csv
from models.ViT import ViT_16_mod
from tools.opts import OPT as opt
import os 
from models.swin_transformer import swin_tiny_patch4_window7_224
from torchvision.transforms import InterpolationMode
import wandb

run = wandb.init()

mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'TinyImageNet': (0.485, 0.456, 0.406),
        }

std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'TinyImageNet': (0.229, 0.224, 0.225),
        }

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
])

transform_train_tiny = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])

transform_test_tiny = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])



if opt.model in ('ViT', 'swint'):
    transform_train = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset]),
    ])


def get_forget_tag(class_to_remove):
    if class_to_remove is None:
        return "all"

    if isinstance(class_to_remove, list):
        if len(class_to_remove) > 10:
            return f"multi_{len(class_to_remove)}_classes"
        elif len(class_to_remove) == 1:
            return str(class_to_remove[0])
        else:
            return "_".join(map(str, class_to_remove))

    return str(class_to_remove)


def log_epoch_to_csv(epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, mode, dataset, model, class_to_remove, seed):
    os.makedirs(f'results/{mode}/epoch_logs', exist_ok=True)

    if isinstance(class_to_remove, list):
        class_name = '_'.join(map(str, class_to_remove))
    else:
        class_name = class_to_remove if class_to_remove is not None else 'all'

    csv_path = f'results/{mode}/epoch_logs/{dataset}_{model}_epoch_results_{class_name}.csv'
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'mode', 'Forget Class', 'seed', 'train_acc', 'train_loss', 'val_acc', 'val_loss', 'test_acc', 'test_loss'])
        writer.writerow([epoch, mode, class_name, seed, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss])

def log_summary_across_classes(best_epoch, train_acc, train_loss, best_acc, val_loss, mode, dataset, model, class_to_remove, seed):
    os.makedirs('results', exist_ok=True)
    summary_path = f'results/{mode}/{dataset}_{model}_unlearning_summary.csv'
    file_exists = os.path.isfile(summary_path)

    if isinstance(class_to_remove, list):
        class_name = '_'.join(map(str, class_to_remove))
    else:
        class_name = class_to_remove if class_to_remove is not None else 'all'

    with open(summary_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'Forget Class', 'seed', 'mode', 'dataset', 'model', 'train_acc', 'train_loss', 'best_val_acc', 'val_loss'])
        writer.writerow([best_epoch, class_name, seed, mode, dataset, model, train_acc, train_loss, best_acc, val_loss])


def get_oracle_loaders(class_to_remove, seed):
    """Build Oracle loaders using the same deterministic 90/10 split as the original model."""
    if opt.dataset == 'TinyImageNet' and opt.model not in ('ViT', 'swint'):
        train_transform = transform_train_tiny
        eval_transform = transform_test_tiny
    else:
        train_transform = transform_train
        eval_transform = transform_test

    if opt.dataset == 'cifar10':
        train_aug = torchvision.datasets.CIFAR10(opt.data_path, train=True, download=True, transform=train_transform)
        train_eval = torchvision.datasets.CIFAR10(opt.data_path, train=True, download=True, transform=eval_transform)
        test_data = torchvision.datasets.CIFAR10(opt.data_path, train=False, download=True, transform=eval_transform)
    elif opt.dataset == 'cifar100':
        train_aug = torchvision.datasets.CIFAR100(opt.data_path, train=True, download=True, transform=train_transform)
        train_eval = torchvision.datasets.CIFAR100(opt.data_path, train=True, download=True, transform=eval_transform)
        test_data = torchvision.datasets.CIFAR100(opt.data_path, train=False, download=True, transform=eval_transform)
    elif opt.dataset == 'TinyImageNet':
        train_path = os.path.join(opt.data_path, 'TinyImageNet', 'train')
        val_path = os.path.join(opt.data_path, 'TinyImageNet', 'val')
        train_aug = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
        train_eval = torchvision.datasets.ImageFolder(train_path, transform=eval_transform)
        # TinyImageNet's labeled val directory is the available held-out test set.
        test_data = torchvision.datasets.ImageFolder(val_path, transform=eval_transform)
    else:
        raise ValueError(f'Unsupported dataset: {opt.dataset}')

    split_generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_aug), generator=split_generator).tolist()
    train_size = int(0.9 * len(indices))
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    targets = np.asarray(train_aug.targets)
    if opt.mode == 'CR':
        forgotten = np.isin(targets, np.atleast_1d(class_to_remove))
    else:
        num = 10000 if opt.dataset == 'TinyImageNet' else 5000
        forget_file = f'{opt.root_folder}forget_id_files/forget_idx_{num}_{opt.dataset}_seed_{seed}.txt'
        forget_indices = np.loadtxt(forget_file).astype(np.int64).reshape(-1)
        forgotten = np.zeros(len(train_aug), dtype=bool)
        forgotten[forget_indices] = True

    retained_train_indices = [idx for idx in train_indices if not forgotten[idx]]
    retained_val_indices = [idx for idx in val_indices if not forgotten[idx]]
    train_loader = DataLoader(Subset(train_aug, retained_train_indices), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(Subset(train_eval, retained_val_indices), batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # Keep the entire held-out test split for reporting, including forgotten
    # classes. It is never used to select the checkpoint.
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader

        
def trainer(class_to_remove, seed):
    # Initialize the model
    if opt.model == 'resnet18':
        model= torchvision.models.resnet18(pretrained=True).to('cuda')
    elif opt.model=='resnet34':
        model= torchvision.models.resnet34(pretrained=True).to('cuda')
    elif opt.model=='resnet50':
        model = torchvision.models.resnet50(pretrained=True).to('cuda')
    elif opt.model=='AllCNN':
        model = AllCNN(n_channels=3, num_classes=opt.num_classes).to('cuda')
    elif opt.model == 'ViT':
        model = ViT_16_mod(n_classes=opt.num_classes).to('cuda')
    elif opt.model == "swint":
        model = swin_tiny_patch4_window7_224(pretrained=True, num_classes=opt.num_classes).to('cuda')   
    else:
        raise ValueError(f"Unknown model: {opt.model}")
    
            
    train_retain_loader, val_loader, test_loader = get_oracle_loaders(class_to_remove, seed)

    if opt.dataset == 'cifar10':
        os.makedirs('./weights/chks_cifar10', exist_ok=True)
        if 'resnet' in opt.model:    
            model.fc = nn.Linear(model.fc.in_features, opt.num_classes).to('cuda')


    elif opt.dataset == 'cifar100':
        os.makedirs('./weights/chks_cifar100', exist_ok=True)
        if 'resnet' in opt.model:    
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
            model.maxpool = nn.Identity()
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to('cuda')

    elif opt.dataset == 'TinyImageNet':
        os.makedirs('./weights/chks_TinyImageNet', exist_ok=True)
        #dataloader
        if 'resnet' in opt.model:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
            model.maxpool = nn.Identity()
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to('cuda')

    epochs=300
    criterion = nn.CrossEntropyLoss(label_smoothing=0.4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-5)
    if opt.dataset == 'TinyImageNet':
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #learning rate decay
    else:
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)


    # Train the network
    best_acc = 0.0
    patience_counter = 0  # Counter to track epochs without improvement
    patience = 30 
    best_epoch = -1
    best_train_acc = 0.0
    best_train_loss = 0.0
    best_val_loss = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        correct, total = 0, 0
        for i, data in enumerate(train_retain_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_retain_loader)
        train_scheduler.step()
        save_dir = f'weights/chks_{opt.dataset}/retrained'
        os.makedirs(f'weights/chks_{opt.dataset}/retrained', exist_ok=True)
        # if opt.mode == 'CR':
        #     torch.save(model.state_dict(), f'weights/chks_{opt.dataset}/retrained/best_checkpoint_without_{class_to_remove}.pth')
        # elif opt.mode == 'HR':
        #     torch.save(model.state_dict(), f'weights/chks_{opt.dataset}/retrained/chks_{opt.dataset}_seed_{seed}.pth')

        if epoch % 1 == 0:
            model.eval()

            def evaluate(loader):
                loss_sum, correct, count = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in loader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        outputs = model(inputs)
                        loss_sum += criterion(outputs, labels).item() * labels.size(0)
                        correct += (outputs.argmax(dim=1) == labels).sum().item()
                        count += labels.size(0)
                return loss_sum / count, 100.0 * correct / count

            val_loss, val_acc = evaluate(val_loader)
            test_loss, test_acc = evaluate(test_loader)

    
                # Check if this is the best validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_train_acc = train_acc
                best_train_loss = train_loss
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience counter
                os.makedirs(f'weights/chks_{opt.dataset}/retrained', exist_ok=True)
                forget_tag = get_forget_tag(class_to_remove)
                torch.save(
                    model.state_dict(),
                    f'weights/chks_{opt.dataset}/retrained/best_checkpoint_{opt.model}_without_{forget_tag}.pth'
                )
                print(f"New best model saved with Val Acc: {best_acc:.3f}")
            else:
                patience_counter += 1  # No improvement

            print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Test Acc: {test_acc:.3f}, Best Val Acc: {best_acc:.3f}')
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc})

            # Save a summary across all unlearning runs
            log_epoch_to_csv(
                epoch=epoch,
                train_acc=round(train_acc, 4),
                train_loss=round(train_loss, 4),
                val_acc=round(val_acc, 4),
                val_loss=round(val_loss, 4),
                test_acc=round(test_acc, 4),
                test_loss=round(test_loss, 4),
                mode='retrained',
                dataset=opt.dataset,
                model=opt.model,
                class_to_remove=class_to_remove,
                seed=seed)
            

            # Check for early stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break  # Exit training loop    return best_acc

    log_summary_across_classes(
        best_epoch=best_epoch,
        train_acc=round(best_train_acc, 4),
        train_loss=round(best_train_loss, 4),
        best_acc=round(best_acc, 4),
        val_loss=round(best_val_loss, 4),
        mode='retrained',
        dataset=opt.dataset,
        model=opt.model,
        class_to_remove=class_to_remove,
        seed=seed)

    print(f"Best epoch: {best_epoch}, Train Acc: {best_train_acc:.2f}, Val Acc: {best_acc:.2f}")


if __name__ == '__main__':
    for i in opt.seed:
        if opt.mode == 'CR':
            for class_to_remove in opt.class_to_remove:
                best_acc = trainer(class_to_remove=class_to_remove,seed=i)
        else:
            best_acc = trainer(seed=i,class_to_remove=None)
