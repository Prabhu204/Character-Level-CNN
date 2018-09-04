"""
@author: Prabhu

"""
import torch.nn as nn

class characterlevel(nn.Module):
    def __init__(self, n_classes = 14, input_dim = 68,input_length= 1014, n_convolutional_filter = 256, n_fc_neurons = 1024):
        super(characterlevel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_convolutional_filter, kernel_size= 7, padding= 0),
                                   nn.ReLU,
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_convolutional_filter,n_convolutional_filter,kernel_size= 7, padding= 0),
                                   nn.ReLU,
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_convolutional_filter,n_convolutional_filter, kernel_size= 3, padding= 0),
                                   nn.ReLU)
        self.conv4 = nn.Sequential(nn.Conv1d(n_convolutional_filter,n_convolutional_filter,kernel_size= 3, padding= 0),
                                   nn.ReLU)
        self.conv5 = nn.Sequential(nn.Conv1d(n_convolutional_filter, n_convolutional_filter, kernel_size= 3, padding= 0),
                                   nn.ReLU)
        self.conv6 = nn.Sequential(nn.Conv1d(n_convolutional_filter, n_convolutional_filter, kernel_size=3, padding=0),
                                   nn.ReLU,
                                   nn.MaxPool1d(3))

        dim = int((input_length-96)/27*n_convolutional_filter)
        self.fc1 = nn.Sequential(nn.Linear(dim, n_fc_neurons))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons),
                                 nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(n_fc_neurons, n_classes))


        self.create_weights(mean = 0.0, std = 0.05)

    def create_weights(self, mean = 0.0, std = 0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)


    def forward(self, input):
        input = input.transpose(1,2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0),-1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
