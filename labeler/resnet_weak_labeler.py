import numpy as np
import torch

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNetWeakLabeler(torch.nn.Module):

    # Need to input input dims and output dims
    # Can input dict_training_param: {learning rate, num batches, num epoches, fc1 size, fc2 size}
    def __init__(self, in_dim_h=28, in_dim_w=28, in_dim_c=1, out_dim=10, dict_training_param=None):
        super(ResNetWeakLabeler, self).__init__()

        self.in_dim_h = in_dim_h
        self.in_dim_w = in_dim_w
        self.in_dim_c = in_dim_c
        self.out_dim = out_dim

        if dict_training_param is not None:
            self.learning_rate = dict_training_param["learning_rate"]
            self.num_batches = dict_training_param["num_batches"]
            self.num_epochs = dict_training_param["num_epochs"]
        else:  # If training params not specified, use random params as follows
            self.learning_rate = 1e-3
            self.num_batches = 20
            self.num_epochs = 10

        # architecture parameters
        block = BasicBlock
        num_blocks = [3, 3, 3]
        num_classes = 10
        
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        torch.cuda.empty_cache()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # update training params
    def update_training_params(self, dict_training_param: dict):
        self.learning_rate = dict_training_param["learning_rate"]
        self.num_batches = dict_training_param["num_batches"]
        self.num_epochs = dict_training_param["num_epochs"]
        return

    # compute shape of tensor output from previous params to determine first fc size
    # assuming h and w are the same
    def compute_fc_size(self):
        h_conv1 = compute_next_shape(prev_shape=self.in_dim_h, padding=0, dilation=1, kernel_size=self.k_conv1, stride=1)
        h_conv2 = compute_next_shape(prev_shape=h_conv1, padding=0, dilation=1, kernel_size=self.k_conv2, stride=1)
        h_pool1 = compute_next_shape(prev_shape=h_conv2, padding=0, dilation=1, kernel_size=self.k_pool1, stride=self.s_pool1)
        h_conv3 = compute_next_shape(prev_shape=h_pool1, padding=0, dilation=1, kernel_size=self.k_conv3, stride=1)
        h_conv4 = compute_next_shape(prev_shape=h_conv3, padding=0, dilation=1, kernel_size=self.k_conv4, stride=1)
        h_pool2 = compute_next_shape(prev_shape=h_conv4, padding=0, dilation=1, kernel_size=self.k_pool2,
                                     stride=self.s_pool2)
        fc_size = self.out_c_conv2 * h_pool2 * h_pool2
        return fc_size

    # retrieve variable tensors
    def get_parameters(self):
        conv1_w = self.conv1.weight
        conv1_b = self.conv1.bias
        conv2_w = self.conv2.weight
        conv2_b = self.conv2.bias
        fc1_w = self.fc1[0].weight
        fc1_b = self.fc1[0].bias
        fc2_w = self.fc2[0].weight
        fc2_b = self.fc2[0].bias
        fc3_w = self.fc3[0].weight
        fc3_b = self.fc3[0].bias
        return {"conv1_w": conv1_w, "conv1_b": conv1_b, "conv2_w": conv2_w, "conv2_b": conv2_b,
                "fc1_w": fc1_w, "fc1_b": fc1_b, "fc2_w": fc2_w, "fc2_b": fc2_b, "fc3_w": fc3_w, "fc3_b": fc3_b}

    # retrieve size of the model in bytes
    def get_size(self):
        dict_params = self.get_parameters()
        size = 0
        for key in dict_params.keys():
            size += torch.numel(dict_params[key])
        return size * 4

    """# input float tensor output prob of class 1 as numpy array
    def marginal(self, X):
        X_tensor = torch.from_numpy(X).to(DEVICE)
        out = self(X_tensor).cpu().detach().numpy()
        return out[:, 1]
d
    # input float tensor output entire prob matrix as numpy array
    def prob_matrix(self, X):
        X_tensor = torch.from_numpy(X).to(DEVICE)
        out = self(X_tensor).cpu().detach().numpy()
        return out"""

    # input float tensor output prob of class 1 as numpy array
    def marginal(self, X):
        X_split = np.split(X, 100, axis=0)
        out_all = []
        for i in range(100):
            X_tensor = torch.from_numpy(X_split[i]).to(DEVICE)
            out = self(X_tensor).cpu().detach().numpy()
            out_all.append(out)
        out_all = np.concatenate(out_all)
        return out_all[:, 1]

    # input float tensor output entire prob matrix as numpy array
    def prob_matrix(self, X):
        X_split = np.split(X, 100, axis=0)
        out_all = []
        for i in range(100):
            X_tensor = torch.from_numpy(X_split[i]).to(DEVICE)
            out = self(X_tensor).cpu().detach().numpy()
            out_all.append(out)
        out_all = np.concatenate(out_all)
        return out_all

    # train the model
    def train_cnn(self, X, y, verbose=False):

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        loss_list = []
        acc_list = []
        total_step = self.num_batches

        batches = batch_loader(X, y, num_batches=self.num_batches)

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(batches):
                images_tensor = torch.from_numpy(images).to(DEVICE)
                labels_tensor = torch.from_numpy(labels).long().to(DEVICE)
                # Run the forward pass
                outputs = self(images_tensor)
                loss = criterion(outputs, labels_tensor)
                loss_list.append(loss.item())
                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    # Track the accuracy
                    total = labels.shape[0]
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels_tensor).sum().item()
                    acc_list.append(correct / total)
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))
        return self


# compare if two models have the same parameters
def cnn_equal(model1, model2):
    params1 = model1.get_parameters()
    params2 = model2.get_parameters()
    for key in params1.keys():
        if key not in params2.keys():
            return False
        if not torch.equal(params1[key], params2[key]):
            return False
    return True


# batch loader of data, input X, y are numpy arrays
def batch_loader(X, y, num_batches):
    shuffler = np.random.permutation(X.shape[0])
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    X_split = np.array_split(X_shuffled, num_batches)
    y_split = np.array_split(y_shuffled, num_batches)

    data = []
    for i in range(num_batches):
        data.append((X_split[i], y_split[i]))

    return data

# compute shape of next layer in one dimension
def compute_next_shape(prev_shape, padding, dilation, kernel_size, stride):
    return (prev_shape + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


