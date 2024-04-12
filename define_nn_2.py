import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Ref
# https://tutorials.pytorch.kr/recipes/recipes/defining_a_neural_network.html
# https://reniew.github.io/12/
# https://yjjo.tistory.com/8


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 첫번째 2D 합성곱 계층
        # 1개의 입력 채널(이미지)을 받아들이고, 사각 커널 사이즈가 3인 합성곱 특징들을 출력합니다.
        self.conv1 = nn.Conv2d(1, 36, 3, 1)

    # x는 데이터를 나타냅니다.
    def forward(self, x):
        print("Input data")
        print(x.size())
        print(x)

        # 데이터가 conv1을 지나갑니다.
        print(f"Pass the conv1:{self.conv1}")
        x = self.conv1(x)
        print(x.size())
        print(x)

        return x


# 임의의 10x10 이미지로 맞춰줍니다.
image_heght = 3
image_width = 3
channel_cnt = 1
kernel_height = 3
kernel_width = 3
stride = 1
padding = 0

random_data = torch.rand((1, 1, image_heght, image_width))  # Sample, Channel, H, W

print(f"image size:{image_heght * image_width}")
max_feature_map_h = math.floor((image_heght - kernel_height + 2 * padding) / stride) + 1
max_feature_map_w = math.floor((image_width - kernel_width + 2 * padding) / stride) + 1
max_feature_map = max_feature_map_h * max_feature_map_w
print(f"Max feature map: {max_feature_map}")

my_nn = Net()
result = my_nn(random_data)

print("output")
print(result)

# How can the number of feature maps be extracted larger than the image size?
