import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from collections import namedtuple

VGG16_LossOutput = namedtuple("VGG16Output", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
VGG19_LossOutput = namedtuple("VGG19Output", ["relu1_2", "relu2_2", "relu3_4", "relu4_4"])

VGG16_Layer = {
    '3': "relu1_2", '8': "relu2_2", '15': "relu3_3", '22': "relu4_3"
}
VGG16_bn_Layer = {
    '5': "relu1_2", '12': "relu2_2", '22': "relu3_3", '32': "relu4_3"
}
VGG19_Layer = {
    '3': "relu1_2", '8': "relu2_2", '17': "relu3_4", '26': "relu4_4"
}
VGG19_bn_Layer = {
    '5': "relu1_2", '12': "relu2_2", '25': "relu3_4", '38': "relu4_4"
}

class LossNetwork(nn.Module):
    def __init__(self, model, layer_name_mapping):
        super(LossNetwork, self).__init__()
        
        self.layers = model.features
        self.layer_name_mapping = layer_name_mapping
        
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        """
        Parameters
        ----------
        x, y : torch.Tensor
            x is the output tensor, y is the label tensor
        """
        loss = None

        for name, module in self.layers._modules.items():
            # Pull the tensor layer by layer
            x, y = module(x), module(y)

            if name in self.layer_name_mapping:
                if loss is None:
                    loss = self.loss(x, y)
                else:
                    loss += self.loss(x, y)
                
        return loss

def main():
    # Paper: Perceptual Losses for Real-Time Style Transfer and Super-Resolution
    vgg16 = models.vgg16(pretrained=True)
    vgg16_bn = models.vgg16_bn(pretrained=True)

    # Paper: Super Resolution GAN (SRGAN)
    vgg19 = models.vgg19(pretrained=True)
    vgg19_bn = models.vgg19_bn(pretrained=True)

    # Paper: None
    resnet50 = models.resnet50(pretrained=True)

    criteria = LossNetwork(vgg16, VGG16_Layer)
    criteria.eval()

    return

if __name__ == "__main__":
    main()
