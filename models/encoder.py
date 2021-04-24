import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(*(list(torchvision.models.resnet50(pretrained=False).children())[:-2]) +
                                     [nn.AdaptiveAvgPool2d((output_size, output_size))])
        self.fine_tune()
    def forward(self, input_):
        x = self.model(input_)  # [BATCH_SIZE * 2048 *  output_size * output_size]
        return x.permute(0, 2, 3, 1)  # [BATCH_SIZE * output_size * output_size * 2048]  => 나중에 가운데 이미지차원 평균으로 squeeze하기

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.model.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune