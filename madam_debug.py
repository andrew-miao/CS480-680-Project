import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import torch.optim as optim
from MAdam import MAdam

def timeSince(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)


def evaluateImageModel(model, data, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for image, labels in data:
            image, labels = image.to(device), labels.to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            loss = criterion(output, labels)
            total_loss += loss.item()

    return total_loss / len(data), 1 - correct / total


def trainImageModel(model, train_data, val_data, n_epochs, criterion, optimizer, device, path):
    model.train()
    train_error_list = []
    val_error_list = []
    min_error = None
    step = 0
    print_every = len(train_data)
    start = time.time()
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for image, labels in train_data:
            optimizer.zero_grad()
            step += 1
            image, labels = image.to(device), labels.to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1)
            running_correct += (pred == labels).sum().item()
            running_total += len(labels)
            loss = criterion(output, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step % print_every == 0:
                val_loss, val_error = evaluateImageModel(model, val_data, criterion, device)
                print(('%d/%d (%s) train loss: %.3f, train error: %.2f%%, val loss: %.3f, val error: %.2f%%') %
                      (epoch + 1, n_epochs, timeSince(start), running_loss / len(train_data),
                       100 * (1 - running_correct / running_total), val_loss, 100 * val_error))
                train_error_list.append(1 - running_correct / running_total)
                val_error_list.append(val_error)
                if min_error is None or min_error > val_error:
                    if min_error is None:
                        print(('Validation error rate in first epoch: %.2f%%') % (100 * val_error))
                    else:
                        print(('Validation error rate is decreasing: %.2f%% --> %.2f%%') %
                              (100 * min_error, 100 * val_error))
                    min_error = val_error
                    print('Saving model...')
                    torch.save(model, path)

                model.train()
                running_loss = 0.0
                running_correct = 0
                running_total = 0

    return train_error_list, val_error_list

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    batch_size = 100
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    model = models.resnet34()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    n_epochs = 100
    nadam_train_list = None
    nadam_test_list = None
    optimizer = MAdam(model.parameters())
    model.to(device)
    path = 'sgdshuffle.pt'
    nadam_train_list, nadam_test_list = trainImageModel(model, trainloader, testloader,
                                                        n_epochs, criterion, optimizer,
                                                        device, path)
    torch.save(nadam_train_list, 'sgdres_train.pt')
    torch.save(nadam_test_list, 'sgdres_test.pt')