import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

n_epoch = 20
batch_size = 256
workers = 1  # # of thread for data loader
image_channels = 3  # RGB = 3
image_height = 1280
image_width = 720
n_classes = 10
n_feat = 128
learning_rate = 1e-4
device = "cuda:0" if torch.cuda.is_available() else "cpu"
save_model = True
save_dir = "learning_space/model/"
train_image_dir = "data/"
ws = 2.0  # strength of generative guidance

# tensor to image
trans = transforms.Compose(
    [
        transforms.Resize((1280, 1280)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
# There must be a class in the image path: Ex. fashion-small-size > [black_jean, gray_jean] > image.jpg
trainset = torchvision.datasets.ImageFolder(root=train_image_dir, transform=trans)
dataloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=workers
)

target_image1 = trainset.__getitem__(0)[0]  # torch.Size([3, 1280, 1280])
target_image2 = trainset.__getitem__(1)[0]  # torch.Size([3, 1280, 1280])

print(target_image1.shape)
save_image(target_image1, "origin1.jpg")
print(target_image2.shape)
save_image(target_image2, "origin2.jpg")


class SynthesisConvBlock(nn.Module):
    def __init__(self) -> None:
        super(SynthesisConvBlock, self).__init__()

        self.synthesis = nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.ReLU())

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1_down = DownSampling()
        input1_down_tensor = input1_down(input1)

        input2_down = DownSampling()
        input2_down_tensor = input2_down(input2)
	stack = torch.cat((input1_down_tensor, input2_down_tensor), 1)
        print(stack.shape)

        syntehsis_tensors = self.synthesis(stack)
        print(syntehsis_tensors.shape)

        syntehsis_tensors_up = UpSampling()
        syntehsis_up_tensor = syntehsis_tensors_up(syntehsis_tensors)

        print(syntehsis_up_tensor.shape)
        return syntehsis_up_tensor


class DownSampling(nn.Module):
    def __init__(self) -> None:
        super(DownSampling, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(3, 640, 3, 1, 0),
            nn.BatchNorm2d(640),
            nn.GELU(),
            nn.Conv2d(640, 320, 3, 1, 1),
            nn.BatchNorm2d(320),
            nn.GELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.down(input)
        print(result.shape)
        return result


class UpSampling(nn.Module):
    def __init__(self) -> None:
        super(UpSampling, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(320, 640, 3, 1, 1),
            nn.BatchNorm2d(640),
            nn.GELU(),
            nn.ConvTranspose2d(640, 3, 3, 1),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.up(input)
        print(result.shape)
        return result


sample_class = SynthesisConvBlock()
result_sample = sample_class(target_image1.unsqueeze(0), target_image2.unsqueeze(0))
save_image(result_sample, "result_sample.jpg")
