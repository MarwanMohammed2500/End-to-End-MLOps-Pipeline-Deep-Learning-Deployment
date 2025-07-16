from torch import nn

class CNNModelV0(nn.Module):
    """
    CNN Model that follows roughly the TinyVGG Architecture

    Arguments:
    in_features:int, the number of input features to the fully connected layer
    out_features:int, the number of output features from the fully connected layer
    hidden_feauters:int=16, the number of hidden features in the convolutional layers
    kernel_size:int=3, the kernel size used in the convulational layers
    stride:int=1, The stride used in the conv layers
    padding:int=0, Image padding used in the conv layers

    returns:
    x:torch.Tensor, the output of the model/the model's prediction
    """
    def __init__(self, in_features:int, out_features:int, hidden_feauters:int=16, kernel_size:int=3, stride:int=1, padding:int=0):
        super().__init__()
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_feauters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_feauters, out_channels=hidden_feauters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_feauters, out_channels=hidden_feauters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_feauters, out_channels=hidden_feauters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = self.fully_connected(x)
        return x
    
MODEL_REGISTRY = {
    "CNNModelV0": CNNModelV0,
}