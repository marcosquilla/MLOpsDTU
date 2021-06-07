from torch import nn

# Conv array [{nFilters, kernel, stride, padding}, 
#               {nFilters, kernel, stride, padding}, ...]
# Hidden layers [neurons_layer_1, 
#                neurons_layer_2, ...]

class MyAwesomeModel(nn.Module):
    def __init__(self, num_classes, channels, height, width, convolutions = [{'nFilters':32, 'kernel':4, 'stride':1, 'padding':1}], hidden_layers = [128]):
        super(MyAwesomeModel, self).__init__()
        
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.channels = channels
        
        self.layers_conv = []
        self.layers_hidden = []
        
# Generation of convolution layers        
        for conv in convolutions:
            self.in_features = self.channels * self.height * self.width

            self.layers_conv.append(nn.Conv2d(in_channels=self.channels,
                                out_channels=conv['nFilters'],
                                kernel_size=conv['kernel'],
                                stride=conv['stride'],
                                padding=conv['padding']))
            
            self.height = (self.height - conv['kernel'] + 2 * conv['padding']) // conv['stride'] + 1
            self.width = (self.width - conv['kernel'] + 2 * conv['padding']) // conv['stride'] + 1
            self.channels = conv['nFilters']

            self.layers_conv.append(nn.ReLU())
            self.layers_conv.append(nn.BatchNorm2d(self.channels))
            self.layers_conv.append(nn.Dropout2d(p=0.5))
        
        self.layers_conv = nn.Sequential(*self.layers_conv)
        
# Generation of fully connected hidden layers

        self.in_features = int(self.channels * self.height * self.width)        
        self.in_hidden_features = self.in_features
        
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.in_features, neurons))
            self.in_features = neurons
            self.layers_hidden.append(nn.ReLU())
            self.layers_hidden.append(nn.BatchNorm1d(neurons))
            self.layers_hidden.append(nn.Dropout2d(p=0.5))

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], self.num_classes))
        
        self.layers_hidden = nn.Sequential(*self.layers_hidden)
        
    def forward(self, x):
        x = self.layers_conv(x)
        x = x.view(-1, self.in_hidden_features)
        x = self.layers_hidden(x)
        
        return nn.functional.softmax(x, dim = 1)