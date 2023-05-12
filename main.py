import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# # Plot example
# img, label = training_data[0]
# figure = plt.figure(figsize=(8, 8))
# figure.add_subplot(1, 1, 1)
# plt.title(label)
# plt.axis("off")
# plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# # Show example image
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

input_dim = 784
hidden_dim = 100
output_dim = 10


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="sigmoid")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.nn.functional.relu(x)  # ReLU seems to overfit less than sigmoid
        x = self.layer_2(x)
        x = torch.nn.functional.softmax(x, dim=1)

        return x


model = NN(input_dim, hidden_dim, output_dim)
print(model)

# Train the model

learning_rate = 0.1
momentum = 0.5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

epochs = 10
loss_values = []
log_interval = 64
train_counter = []
test_counter = [i * len(train_dataloader.dataset) for i in range(epochs + 1)]
test_losses = []


# examples = enumerate(test_dataloader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_data.shape)

def train(epoch):
    model.train()
    for indx, (X, y) in enumerate(train_dataloader):
        X = torch.flatten(X, start_dim=1)
        # y = torch.nn.functional.one_hot(y)
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)
        if indx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, indx * len(X), len(train_dataloader.dataset),
                       100. * indx / len(train_dataloader), loss.item()))
        loss_values.append(loss.item())
        train_counter.append((indx * 64) + ((epoch - 1) * len(train_dataloader.dataset)))
        loss.backward()
        optimizer.step()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data = torch.flatten(data, start_dim=1)

            output = model(data)
            test_loss += loss_fn(output, target).item()
            counter += 1
            pred = output.data.max(1, keepdim=True)[1]  # [1] gets the index of the max
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= counter  # Not sure why i need to use counter and not train_dataloader.dataset
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


test()
for epoch in range(1, epochs + 1):
    train(epoch)
    test()

print("Training Complete")

fig = plt.figure()
plt.plot(train_counter, loss_values, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
