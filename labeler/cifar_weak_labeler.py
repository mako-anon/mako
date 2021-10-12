import numpy as np
import torch
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


class CifarWeakLabeler(torch.nn.Module):

    # Need to input input dims and output dims
    # Can input dict_training_param: {learning rate, num batches, num epoches, fc1 size, fc2 size}
    def __init__(self, in_dim_h=32, in_dim_w=32, in_dim_c=3, out_dim=2, dict_training_param=None):
        super(CifarWeakLabeler, self).__init__()

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
            self.num_batches = 5
            self.num_epochs = 100

        # architecture parameters
        self.out_c_conv1 = 4
        self.k_conv1 = 3
        self.out_c_conv2 = 4
        self.k_conv2 = 3
        self.k_pool2 = 2
        self.s_pool2 = 2
        self.out_c_conv3 = 16
        self.k_conv3 = 3
        self.out_c_conv4 = 32
        self.k_conv4 = 3
        self.k_pool4 = 2
        self.s_pool4 = 2
        self.out_c_conv5 = 32
        self.k_conv5 = 3
        self.d_fc1 = self.compute_fc_size()
        self.d_fc2 = 45
        # self.d_fc3 = 21

        self.conv1 = torch.nn.Conv2d(in_channels=self.in_dim_c, out_channels=self.out_c_conv1, kernel_size=self.k_conv1)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=self.out_c_conv1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out_c_conv1, out_channels=self.out_c_conv2, kernel_size=self.k_conv2)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=self.out_c_conv2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=self.k_pool2, stride=self.s_pool2)
        self.dropout2 = torch.nn.Dropout2d(p=0.5)
        self.conv3 = torch.nn.Conv2d(in_channels=self.out_c_conv2, out_channels=self.out_c_conv3, kernel_size=self.k_conv3)
        self.batchnorm3 = torch.nn.BatchNorm2d(num_features=self.out_c_conv3)
        self.conv4 = torch.nn.Conv2d(in_channels=self.out_c_conv3, out_channels=self.out_c_conv4, kernel_size=self.k_conv4)
        self.batchnorm4 = torch.nn.BatchNorm2d(num_features=self.out_c_conv4)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=self.k_pool4, stride=self.s_pool4)
        self.dropout4 = torch.nn.Dropout2d(p=0.2)
        self.conv5 = torch.nn.Conv2d(in_channels=self.out_c_conv4, out_channels=self.out_c_conv5, kernel_size=self.k_conv5)
        self.batchnorm5 = torch.nn.BatchNorm2d(num_features=self.out_c_conv5)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(self.d_fc1, self.d_fc2), torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(self.d_fc2, self.out_dim), torch.nn.Softmax())
        # self.fc3 = torch.nn.Sequential(torch.nn.Linear(self.d_fc3, self.out_dim), torch.nn.Softmax())

    def forward(self, x, training=False):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.batchnorm1(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.batchnorm2(out)
        out = self.maxpool2(out)
        out = self.dropout2(out)
        out = self.conv3(out)
        out = torch.relu(out)
        out = self.batchnorm3(out)
        out = self.conv4(out)
        out = torch.relu(out)
        out = self.batchnorm4(out)
        out = self.maxpool4(out)
        out = self.dropout4(out)
        out = self.conv5(out)
        out = torch.relu(out)
        out = self.batchnorm5(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        return out

    # compute shape of tensor output from previous params to determine first fc size
    def compute_fc_size(self):
        h_conv1 = compute_next_shape(prev_shape=self.in_dim_h, padding=0, dilation=1, kernel_size=self.k_conv1, stride=1)

        h_conv2 = compute_next_shape(prev_shape=h_conv1, padding=0, dilation=1, kernel_size=self.k_conv2, stride=1)
        h_pool2 = compute_next_shape(prev_shape=h_conv2, padding=0, dilation=1, kernel_size=self.k_pool2, stride=self.s_pool2)
        h_conv3 = compute_next_shape(prev_shape=h_pool2, padding=0, dilation=1, kernel_size=self.k_conv3, stride=1)
        h_conv4 = compute_next_shape(prev_shape=h_conv3, padding=0, dilation=1, kernel_size=self.k_conv4, stride=1)
        h_pool4 = compute_next_shape(prev_shape=h_conv4, padding=0, dilation=1, kernel_size=self.k_pool4, stride=self.s_pool4)
        h_conv5 = compute_next_shape(prev_shape=h_pool4, padding=0, dilation=1, kernel_size=self.k_conv5, stride=1)

        w_conv1 = compute_next_shape(prev_shape=self.in_dim_w, padding=0, dilation=1, kernel_size=self.k_conv1, stride=1)
        w_conv2 = compute_next_shape(prev_shape=w_conv1, padding=0, dilation=1, kernel_size=self.k_conv2, stride=1)
        w_pool2 = compute_next_shape(prev_shape=w_conv2, padding=0, dilation=1, kernel_size=self.k_pool2, stride=self.s_pool2)
        w_conv3 = compute_next_shape(prev_shape=w_pool2, padding=0, dilation=1, kernel_size=self.k_conv3, stride=1)
        w_conv4 = compute_next_shape(prev_shape=w_conv3, padding=0, dilation=1, kernel_size=self.k_conv4, stride=1)
        w_pool4 = compute_next_shape(prev_shape=w_conv4, padding=0, dilation=1, kernel_size=self.k_pool4, stride=self.s_pool4)
        w_conv5 = compute_next_shape(prev_shape=w_pool4, padding=0, dilation=1, kernel_size=self.k_conv5, stride=1)

        fc_size = self.out_c_conv5 * h_conv5 * w_conv5
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

    # input float tensor output prob of class 1 as numpy array
    def marginal(self, X):
        X_tensor = torch.from_numpy(X).to(DEVICE)
        out = self(X_tensor).cpu().detach().numpy()
        return out[:, 1]

    # input float tensor output entire prob matrix as numpy array
    def prob_matrix(self, X):
        X_tensor = torch.from_numpy(X).to(DEVICE)
        out = self(X_tensor).cpu().detach().numpy()
        return out

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


if __name__ == '__main__':

    # Generate label matrices from Snuba weak labelers for Snorkel
    def generate_label_matrices(X_l, X_u, lfs):
        L_l = []
        L_u = []
        for lf in lfs:
            prob_l = lf.prob_matrix(X_l.astype('float32'))
            y_l_hat = np.argmax(prob_l, axis=1)
            L_l.append(y_l_hat)
            prob_u = lf.prob_matrix(X_u.astype('float32'))
            y_u_hat = np.argmax(prob_u, axis=1)
            L_u.append(y_u_hat)
        return np.array(L_u).T, np.array(L_l).T

    X_l = np.load('D:\phd\mako\mako_iclr\\sample_task_data\cifar10\X_l.npy')
    y_l = np.load('D:\phd\mako\mako_iclr\\sample_task_data\cifar10\y_l.npy')
    X_u = np.load('D:\phd\mako\mako_iclr\\sample_task_data\cifar10\X_u.npy')
    y_u = np.load('D:\phd\mako\mako_iclr\\sample_task_data\cifar10\y_u.npy')

    from utils.bootstrapping import bootstrap_xy

    lfs = []

    for i in range(20):
        X_boot, y_boot = bootstrap_xy(X_l, y_l, size=30)
        lf = CifarWeakLabeler()
        lf.train()
        lf.train_cnn(X=X_boot, y=y_boot, verbose=False)
        lf.eval()

        logit_l_prime = lf.prob_matrix(X=X_l)
        y_l_prime = np.argmax(logit_l_prime, axis=1)
        num_correct_l = np.sum(y_l == y_l_prime)
        empirical_acc_l = float(num_correct_l) / y_l.shape[0]
        print("acc_l: ", empirical_acc_l)

        logit_u_prime = lf.prob_matrix(X=X_u)
        y_u_prime = np.argmax(logit_u_prime, axis=1)
        num_correct_u = np.sum(y_u == y_u_prime)
        empirical_acc_u = float(num_correct_u) / y_u.shape[0]
        print("acc u: ", empirical_acc_u)

        lfs.append(lf)

    L_u, L_l = generate_label_matrices(X_l, X_u, lfs)
    L = np.concatenate((L_l, L_u), axis=0)

    from snorkel.labeling.model.label_model import LabelModel
    snorkel_model = LabelModel(cardinality=2, verbose=False)
    snorkel_model.fit(L)
    y_snorkel_u, logit_u = snorkel_model.predict(L_u, return_probs=True)
    y_snorkel_l, logit_l = snorkel_model.predict(L_l, return_probs=True)
    num_correct_l = np.sum(y_l == y_snorkel_l)
    learned_acc_l = float(num_correct_l) / y_l.shape[0]
    num_correct_u = np.sum(y_u == y_snorkel_u)
    learned_acc_u = float(num_correct_u) / y_u.shape[0]
    print("final acc l: ", learned_acc_l)
    print("final acc u: ", learned_acc_u)


