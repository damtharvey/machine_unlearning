import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import time


class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.ensemble_size = len(models)
        self.ensemble = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack(tuple(model(x) for model in self.ensemble)).mean(0)

# class HomogeneousEnsemble(Ensemble):
#     def __init__(self, individual: type(nn.Module), ensemble_size: int = 1, **kwargs):
#         super().__init__([individual(**kwargs).to(device) for _ in range(ensemble_size)])


def train_model(model, optimizer, criterion, train_loader, epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        for examples, labels in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            
            examples = examples.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(examples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        validation_loss = 0
        n_correct = 0
        n_examples = 0
        for examples, labels in test_loader:
            examples = examples.to(device)
            labels = labels.to(device)
            logits = model(examples)
            loss = criterion(logits, labels)
            n_correct += (logits.max(1).indices == labels).sum().item()

            validation_loss += loss.item()
            n_examples += labels.shape[0]
        print(f">> Mean Loss: {validation_loss / n_examples:.5f}, Accuracy: {n_correct / n_examples:.4f}")

# def test_model(models, ):

if __name__ == '__main__':
    start = time.time()
    batch_size = 512
    chunks = 5
    subset_ratio = 0.2
    num_models = 5
    # deletion_size = 5
    deletion_ratio = 0.0001
    epochs = 1
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''''Get data'''
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    train_set_chunks_data = (
        (torch.tensor(train_set.data).permute(0,3,1,2) / 255
        - torch.tensor([0.4914, 0.4821, 0.4465])[None, :, None, None])
        / torch.tensor([0.2470, 0.2435, 0.2616])[None, :, None, None]
        ).chunk(chunks)
    train_set_chunks_targets = torch.tensor(train_set.targets).chunk(chunks)
    
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            # shuffle=True, num_workers=8)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False, num_workers=8)
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    models = [None] * num_models
    valid = [False] * num_models
    data_usages = [None] * num_models
    all_deleted_indices = set()
    working_train_set_data = train_set_chunks_data[0]
    working_train_set_targets = train_set_chunks_targets[0]
    print("Chunk:", 1)
    for i in range(0, chunks):
        training_size = len(working_train_set_data)
        sample_weights = torch.ones(training_size) / (training_size - len(all_deleted_indices))
        for deleted_index in all_deleted_indices:
            sample_weights[deleted_index] = 0
        sample_size = int(subset_ratio * training_size)
        deletion_size = int(deletion_ratio * training_size)
        '''Model training phase'''
        for j in range(num_models):
            print(f"Model {j + 1}, Valid:", valid[j]) 
            '''Bagging training samples'''
            train_sample_indices = sample_weights.multinomial(num_samples=sample_size, replacement=True)
            train_set_samples = data.TensorDataset(working_train_set_data[train_sample_indices], working_train_set_targets[train_sample_indices])
            '''Creating the model'''
            if not valid[j]:
                model = torchvision.models.resnet18(num_classes=10).to(device)
                models[j] = model
                valid[j] = True
                data_usages[j] = set(np.unique(train_sample_indices.numpy()))
            else:
                model = models[j]
                data_usages[j].update(np.unique(train_sample_indices.numpy()).tolist())
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_loader = torch.utils.data.DataLoader(train_set_samples, batch_size=batch_size,
                                                    shuffle=True, num_workers=8)
            # train_model(model, optimizer, criterion, train_loader, epochs)
            '''Setting model to ensemble'''
            data_usages[j] = set(np.unique(train_sample_indices.numpy()))
            
        
        # '''Model evaluation phase'''
        # print("Ensemble")
        # model = Ensemble(models)
        # model.eval()
        # validation_loss = 0
        # n_correct = 0
        # n_examples = 0
        # for examples, labels in test_loader:
        #     examples = examples.to(device)
        #     labels = labels.to(device)
        #     logits = model(examples)
        #     loss = criterion(logits, labels)
        #     n_correct += (logits.max(1).indices == labels).sum().item()

        #     validation_loss += loss.item()
        #     n_examples += labels.shape[0]
        # print(f">> Mean Loss: {validation_loss / n_examples:.5f}, Accuracy: {n_correct / n_examples:.4f}")

        '''Deletion request phase: completed in the next loop'''
        if i == chunks - 1: 
            break
        print("Data Deletions:", deletion_size)
        deletion_indices = sample_weights.multinomial(num_samples=deletion_size, replacement=False)
        for deletion_index in deletion_indices:
            all_deleted_indices.add(int(deletion_index))
            # data_deletions = [ [] for _ in range(num_models) ]
            for j in range(num_models):
                if int(deletion_index) in data_usages[j]:
                    valid[j] = False
                    # data_deletions[j].append(int(deletion_index))
                    # print(f"Model {j + 1} used data at {deletion_index}")
        # for k in range(num_models):
        #     print(f"Model {k + 1} Data Deletions: {data_deletions[k]}")
        working_train_set_data = torch.cat((working_train_set_data, train_set_chunks_data[i + 1]))
        working_train_set_targets = torch.cat((working_train_set_targets, train_set_chunks_targets[i + 1]))
        print("Chunk:", i + 2, "New Data Size", len(working_train_set_data))
    end = time.time()
    print("Time:", end - start)
        