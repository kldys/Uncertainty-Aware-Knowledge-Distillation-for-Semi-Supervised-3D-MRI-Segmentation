import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class Gabor3DConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, sigma, theta, lambd, gamma, psi, padding_size):
#         super(Gabor3DConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#         self.theta = theta
#         self.lambd = lambd
#         self.gamma = gamma
#         self.psi = psi
#         self.padding_size = padding_size
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # Generate Gabor filter
#         self.gabor_filter = self._generate_gabor_filter()
#         # print(self.gabor_filter)
#         # Convert the Gabor filter to a PyTorch tensor
#         self.gabor_filter = torch.FloatTensor(self.gabor_filter).to(self.device)
#         self.gabor_filter = self.gabor_filter.view(1, 1, *self.gabor_filter.shape)  # Add batch and channel dimensions
#
#         # Create convolutional layer with the Gabor filter as weights
#         self.conv_layer = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, self.padding_size, bias=False)
#
#         # Set the weights of the convolutional layer to the Gabor filter
#         self.conv_layer.weight.data = self.gabor_filter
#         self.conv_layer.weight.requires_grad = True  # Freeze the weights
#
#     def _generate_gabor_filter(self):
#         x, y, z = np.meshgrid(np.arange(-self.kernel_size[0] // 2 + 1, self.kernel_size[0] // 2 + 1),
#                               np.arange(-self.kernel_size[1] // 2 + 1, self.kernel_size[1] // 2 + 1),
#                               np.arange(-self.kernel_size[2] // 2 + 1, self.kernel_size[2] // 2 + 1))
#
#         x_theta = x * np.cos(self.theta) + y * np.sin(self.theta)
#         y_theta = -x * np.sin(self.theta) + y * np.cos(self.theta)
#
#         envelope = np.exp(-((x_theta ** 2 + (y_theta ** 2 * self.gamma ** 2) + z ** 2) / (2 * self.sigma ** 2)))
#         sinusoid = np.cos(2 * np.pi * x_theta / self.lambd + self.psi)
#         # print(envelope)
#         # print(sinusoid)
#         gabor_filter = envelope * sinusoid
#
#         gabor_filter /= np.sum(np.abs(gabor_filter))  # Normalize the filter
#
#         gabor_filter = gabor_filter.astype(np.float32)
#
#         return gabor_filter
#
#     def forward(self, x):
#         return self.conv_layer(x)

class Gabor3DConvolution(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0, frequency=0.2, theta=0.0):
        super(Gabor3DConvolution, self).__init__()

        # Construct the 3D Gabor kernel
        self.weight = self.construct_3d_gabor_kernel(kernel_size, sigma, frequency, theta)

    def gabor_kernel_2d(self, size, sigma=1.0, frequency=0.2, theta=0.0):
        x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
        x = x - (size - 1) / 2
        y = y - (size - 1) / 2

        rotx = x * np.cos(theta) + y * np.sin(theta)
        roty = -x * np.sin(theta) + y * np.cos(theta)

        gabor = np.exp(-(rotx ** 2 + roty ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * frequency * rotx)

        return gabor / np.sum(np.abs(gabor))

    def construct_3d_gabor_kernel(self, size, sigma=1.0, frequency=0.2, theta=0.0):
        # Generate 2D Gabor kernel
        kernel_2d = self.gabor_kernel_2d(size, sigma, frequency, theta)

        # Repeat to create 3x3x3 kernel
        kernel_3d = torch.from_numpy(kernel_2d).unsqueeze(0)  # 1x3x3
        kernel_3d = kernel_3d.repeat(3, 1, 1)  # 3x3x3

        # Convert to a 3D convolution weight tensor
        weight = kernel_3d.unsqueeze(0).unsqueeze(0)  # 1x3x3x3

        return weight.float().cuda()

    def forward(self, x):
        # Apply 3D convolution using F.conv3d
        return F.conv3d(x, self.weight, padding=1)

class Gaussian3DConvolution(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(Gaussian3DConvolution, self).__init__()

        # Construct the 3D Gaussian kernel
        self.weight = self.construct_3d_gaussian_kernel(kernel_size, sigma)

    def gaussian_kernel_2d(self, size, sigma=1.0):
        kernel = torch.from_numpy(
            np.fromfunction(
                lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                    -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)
                ),
                (size, size)
            )
        )
        return kernel / torch.sum(kernel)

    def construct_3d_gaussian_kernel(self, size, sigma=1.0):
        # Generate 2D Gaussian kernel
        kernel_2d = self.gaussian_kernel_2d(size, sigma)

        # Repeat to create 3x3x3 kernel
        kernel_3d = kernel_2d.unsqueeze(0)  # 1x3x3
        kernel_3d = kernel_3d.repeat(3, 1, 1)  # 3x3x3

        # Convert to a 3D convolution weight tensor
        weight = kernel_3d.unsqueeze(0).unsqueeze(0)  # 1x3x3x3

        return weight.float().cuda()

    def forward(self, x):
        # Apply 3D convolution using F.conv3d
        return F.conv3d(x, self.weight, padding=1)

# # Example usage with padding
# in_channels = 1
# out_channels = 1
# kernel_size = (3, 3, 3)
# sigma = 1.0
# theta = np.pi / 4
# lambd = 5.0
# gamma = 1.0
# psi = 0.0
# padding = 1  # Set padding to a non-zero value

# gabor_conv = Gabor3DConv(in_channels, out_channels, kernel_size, sigma, theta, lambd, gamma, psi, padding)

# # Generate a random input tensor
# input_tensor = torch.randn(1, in_channels, 10, 10, 10)

# # Apply Gabor convolution
# output = gabor_conv(input_tensor)
# print("Output shape:", output.shape)
