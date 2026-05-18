from typing import Optional

import numpy as np
import torch
# from mmcv.runner import load_checkpoint
from mmengine.runner import load_checkpoint

from torch import Tensor, nn
from typing import Union
import numpy

# from .builder import FILTERS
# from .filter import TemporalFilter
from pytorch3d.transforms import (
        axis_angle_to_matrix,
        axis_angle_to_quaternion,
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        matrix_to_rotation_6d,
        quaternion_to_axis_angle,
        quaternion_to_matrix,
        rotation_6d_to_matrix,
    )


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self,
                 rotation: Union[torch.Tensor, numpy.ndarray],
                 convention: str = 'xyz',
                 **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, numpy.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError(
                'Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation
def aa_to_rotmat(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def rotmat_to_rot6d(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def rotmat_to_aa(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)



def rot6d_to_rotmat(
    rotation_6d: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)     
class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.
    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (*, in_channels)
        Output: (*, in_channels)
    """

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out

class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.

    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5

    Shape:
        Input: (*, in_channels)
        Output: (*, in_channels)
    """

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class SmoothNet(nn.Module):
    """SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.

    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .

    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)

    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5

    Shape:
        Input: (N, C, T) the original pose sequence
        Output: (N, C, T) the smoothed pose sequence
    """

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        num_windows = T - self.window_size + 1

        assert T >= self.window_size, (
            'Input sequence length must be no less than the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Unfold x to obtain input sliding windows
        # [N, C, num_windows, window_size]
        x = x.unfold(2, self.window_size, 1)

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, num_windows, output_size]

        # Accumulate output ensembles
        out = x.new_zeros(N, C, T)
        count = x.new_zeros(T)

        for t in range(num_windows):
            out[..., t:t + self.output_size] += x[:, :, t]
            count[t:t + self.output_size] += 1.0

        return out.div(count)


class SmoothNetFilter:

    def __init__(
        self,
        window_size: int,
        output_size: int,
        checkpoint: Optional[str] = None,
        hidden_size: int = 512,
        res_hidden_size: int = 512,
        num_blocks: int = 5,
        device: str = 'cpu',
    ):
        super(SmoothNetFilter, self).__init__()
        self.window_size = window_size
        self.device = device
        self.smoothnet = SmoothNet(window_size, output_size, hidden_size,
                                   res_hidden_size, num_blocks)
        self.smoothnet.to(device)
        if checkpoint:
            load_checkpoint(
                self.smoothnet, checkpoint, map_location=self.device)
        self.smoothnet.eval()

        for p in self.smoothnet.parameters():
            p.requires_grad_(False)

    def __call__(self, x: np.ndarray, type: str):
        assert type in ['rot', 'pos']
        x_type = 'tensor'
        if not isinstance(x, torch.Tensor):
            x_type = 'array'

        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        T, K, C = x.shape

        assert C == 3 or C == 6 or C == 9

        if T < self.window_size:
            # Skip smoothing if the input length is less than the window size
            smoothed = x
        else:
            if x_type == 'array':
                dtype = x.dtype

            # Convert to tensor and forward the model
            with torch.no_grad():
                if x_type == 'array':
                    x = torch.tensor(
                        x, dtype=torch.float32, device=self.device)
                if type == 'rot' :
                    if C == 9:
                        input_type = 'matrix'
                        x = rotmat_to_rot6d(x.reshape(-1, 3, 3)).reshape(T, K, -1)
                    elif C == 3:
                        input_type = 'axis_angles'
                        x = rotmat_to_rot6d(aa_to_rotmat(x.reshape(-1,
                                                                   3))).reshape(
                                                                       T, K, -1)
                    else:
                        input_type = 'rotation_6d'
                elif type ==' pos':
                    pass
                print(x.shape[:])#torch.Size([1001, 55, 6])
                x = x.view(1, T, -1).permute(0, 2, 1)  # to [1, KC, T] #torch.Size([1, 330, 1001])
                print(x.shape[:])
                smoothed = self.smoothnet(x)  # in shape [1, KC, T] #torch.Size([1, 330, 1001])
                print(smoothed.shape[:])
             

            # Convert model output back to input shape and format
            smoothed = smoothed.permute(0, 2, 1).view(T, K, -1)  # to [T, K, C]

            if type == 'rot':
                if input_type == 'matrix':
                    smoothed = rot6d_to_rotmat(smoothed.reshape(-1, 6)).reshape(
                        T, K, C)
                elif input_type == 'axis_angles':
                    smoothed = rotmat_to_aa(
                        rot6d_to_rotmat(smoothed.reshape(-1, 6))).reshape(T, K, C)

            if x_type == 'array':
                smoothed = smoothed.cpu().numpy().astype(
                    dtype)  # to numpy.ndarray
      
        return smoothed
