import paddle

"""
Baseline Models.
"""
from collections import OrderedDict
from logging import getLogger

from neuralop.models import FNO3d

from .embedder import get_embedder, patchify
from .transformer import TransformerDataEncoder

logger = getLogger()
ACTIVATION = {
    "gelu": paddle.nn.GELU(),
    "tanh": paddle.nn.Tanh(),
    "sigmoid": paddle.nn.Sigmoid(),
    "relu": paddle.nn.ReLU(),
    "leaky_relu": paddle.nn.LeakyReLU(negative_slope=0.1),
    "softplus": paddle.nn.Softplus(),
    "ELU": paddle.nn.ELU(),
    "silu": paddle.nn.SiLU(),
}


class FNO(paddle.nn.Module):
    """
    Wrapper for FNO model (1to1).
    """

    def __init__(self, config, max_output_dim):
        super().__init__()
        self.config = config
        self.fno = FNO3d(
            n_modes_height=config.n_modes_height,
            n_modes_width=config.n_modes_width,
            n_modes_depth=config.n_modes_depth,
            hidden_channels=config.hidden_channels,
            in_channels=max_output_dim,
            out_channels=max_output_dim,
            n_layers=config.n_layers,
        )

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, output_times, **kwargs):
        """
        Inputs:
            data_input:          Tensor     (bs, input_len, x_num, x_num, data_dim)
            output_times:        Tensor     (bs/1, output_len, 1)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        assert data_input.size(1) >= output_times.size(
            1
        ), "FNO3D requires input_len >= output_len"
        data_input = data_input.permute(0, 4, 1, 2, 3)
        output_len = output_times.size(1)
        data_output = self.fno(data_input)
        data_output = data_output.permute(0, 2, 3, 4, 1)
        data_output = data_output[:, :output_len, :, :, :]
        return data_output


class UNet(paddle.nn.Module):
    def __init__(self, config, max_output_dim):
        super().__init__()
        self.config = config
        self.features = config.width
        self.act = ACTIVATION[config.act]
        self.in_channels = max_output_dim
        self.out_channels = max_output_dim
        self.__name__ = "UNet"
        self.encoder1 = self._block(self.in_channels, self.features, name="enc1")
        self.pool1 = paddle.nn.MaxPool3D(kernel_size=2, stride=2)
        self.encoder2 = self._block(self.features, self.features * 2, name="enc2")
        self.pool2 = paddle.nn.MaxPool3D(kernel_size=2, stride=2)
        self.encoder3 = self._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = paddle.nn.MaxPool3D(kernel_size=2, stride=2)
        self.bottleneck = self._block(
            self.features * 4, self.features * 8, name="bottleneck"
        )
        self.upconv3 = paddle.nn.Conv3DTranspose(
            in_channels=self.features * 8,
            out_channels=self.features * 4,
            kernel_size=2,
            stride=2,
        )
        self.decoder3 = self._block(
            self.features * 4 * 2, self.features * 4, name="dec3"
        )
        self.upconv2 = paddle.nn.Conv3DTranspose(
            in_channels=self.features * 4,
            out_channels=self.features * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = self._block(
            self.features * 2 * 2, self.features * 2, name="dec2"
        )
        self.upconv1 = paddle.nn.Conv3DTranspose(
            in_channels=self.features * 2,
            out_channels=self.features,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = self._block(self.features * 2, self.features, name="dec1")
        self.conv = paddle.nn.Conv3d(
            in_channels=self.features, out_channels=self.out_channels, kernel_size=1
        )

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, output_times, **kwargs):
        """
        Inputs:
            data_input:          Tensor     (bs, input_len, x_num, x_num, data_dim)
            output_times:        Tensor     (bs/1, output_len, 1)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        x = data_input.permute(0, 4, 1, 2, 3)
        assert x.size(2) == 10
        x = paddle.compat.nn.functional.pad(x, (0, 0, 0, 0, 0, 6))
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = paddle.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = paddle.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = paddle.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        x = self.conv(dec1)
        x = x.permute(0, 2, 3, 4, 1)
        output_len = output_times.size(1)
        data_output = x[:, :output_len, :, :, :]
        return data_output

    def _block(self, in_channels, features, name):
        return paddle.nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        paddle.nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", paddle.nn.BatchNorm3D(num_features=features)),
                    (name + "tanh1", self.act),
                    (
                        name + "conv2",
                        paddle.nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", paddle.nn.BatchNorm3D(num_features=features)),
                    (name + "tanh2", self.act),
                ]
            )
        )


class ViT(paddle.nn.Module):
    """
    Wrapper for vision transformer (1to1).
    """

    def __init__(self, config, x_num, max_output_dim):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim
        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.encoder = TransformerDataEncoder(config.encoder)

    def summary(self):
        s = "\n"
        s += f"""	Embedder:        {sum([p.numel() for p in self.embedder.parameters() if not p.stop_gradient]):,}
"""
        s += f"\tEncoder:         {sum([p.numel() for p in self.encoder.parameters() if not p.stop_gradient]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, input_times, output_times, **kwargs):
        """
        Inputs:
            data_input:          Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:         Tensor     (bs/1, input_len, 1)
            output_times:        Tensor     (bs/1, output_len, 1)

        Output:
            data_output:     Tensor     (bs, 1, x_num, x_num, data_dim)
        """
        output_len = output_times.size(1)
        assert (
            input_times.size(1) >= output_len
        ), "For this ViT, input length should >= output length"
        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, input_len, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = input_len * patch_num * patch_num
        """
        data_input = self.embedder.encode(data_input, input_times)
        """
        Step 2: Encode
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_encoded = self.encoder(data_input)
        """
        Step 3: Decode data
        """
        data_output = self.embedder.decode(data_encoded)
        return data_output[:, :output_len]


act = paddle.nn.Tanh()


class OneInputBasis(paddle.nn.Module):
    def __init__(self, num_sensors, dim1):
        super().__init__()
        self.num_sensors = num_sensors
        self.dim1 = dim1
        bo_b = True
        bo_last = False
        self.l1 = paddle.compat.nn.Linear(self.num_sensors, 100, bias=bo_b)
        self.l4 = paddle.compat.nn.Linear(100, self.dim1, bias=bo_last)

    def forward(self, v):
        v = act(self.l1(v))
        v = self.l4(v)
        return v


class branch(paddle.nn.Module):
    def __init__(self, basis_dim, num_sensors, dim1):
        super().__init__()
        self.basis_dim = basis_dim
        self.num_sensors = num_sensors
        self.dim1 = dim1
        self.set_lay = paddle.nn.ModuleList(
            [OneInputBasis(self.num_sensors, self.dim1) for _ in range(self.basis_dim)]
        )

    def forward(self, v):
        w = self.set_lay[0](v)
        for ii in range(self.basis_dim - 1):
            w = paddle.cat((w, self.set_lay[ii + 1](v)), dim=1)
        return w


class MeshNetwork(paddle.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        bo_b = True
        bo_last = False
        self.l3 = paddle.compat.nn.Linear(3, 100, bias=bo_b)
        self.l4 = paddle.compat.nn.Linear(100, 100, bias=bo_b)
        self.l5 = paddle.compat.nn.Linear(100, 100, bias=bo_b)
        self.l6 = paddle.compat.nn.Linear(100, 100, bias=bo_b)
        self.l7 = paddle.compat.nn.Linear(100, hidden_dim, bias=bo_last)

    def forward(self, w):
        w = act(self.l3(w))
        w = act(self.l4(w))
        w = act(self.l5(w))
        w = act(self.l6(w))
        w = self.l7(w)
        return w


class DeepONet(paddle.nn.Module):
    def __init__(
        self, model_config, data_config, input_len, x_num=128, output_t_num=10
    ):
        super().__init__()
        self.x_dim = data_config.x_num
        self.patch_num = model_config.patch_num
        self.num_t = input_len
        self.dim = model_config.emb_dim
        self.basis_dim = model_config.basis_dim
        self.output_channel = data_config.max_output_dimension
        self.input_channel = self.output_channel
        self.patch_resolution = self.x_dim // self.patch_num
        self.patch_dim = (
            self.input_channel * self.patch_resolution * self.patch_resolution
        )
        self.singlebranch = model_config.singlebranch
        self.branch_dim = 1 if self.singlebranch else self.output_channel
        self.top = branch(
            self.basis_dim, self.num_t * self.patch_num**2, self.branch_dim
        )
        self.bottom = paddle.nn.ModuleList(
            [MeshNetwork(self.basis_dim) for _ in range(self.output_channel)]
        )
        self.pre_proj = paddle.nn.Sequential(
            paddle.compat.nn.Linear(self.patch_dim, self.dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(self.dim, 1),
        )
        x = paddle.linspace(0, 128, x_num, dtype=paddle.float32)[None]
        y = paddle.linspace(0, 128, x_num, dtype=paddle.float32)[None]
        t = paddle.linspace(0, 10, output_t_num, dtype=paddle.float32)[None]
        t_mesh, x_mesh, y_mesh = paddle.meshgrid(
            t.squeeze(), x.squeeze(), y.squeeze(), indexing="ij"
        )
        meshgrid = paddle.stack((t_mesh, x_mesh, y_mesh), dim=-1).reshape(-1, 3)
        self.register_buffer("meshgrid", meshgrid, persistent=False)

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, output_times, **kwargs):
        meshgrid = self.meshgrid
        bs = data_input.shape[0]
        data = patchify(data_input, self.patch_num)
        data = self.pre_proj(data)
        data = data.view(-1, 1, self.num_t * self.patch_num**2)
        k1 = self.top(data)
        output = 0
        for i in range(self.output_channel):
            branch_output = self.bottom[i](meshgrid)
            duplicated_tensor = branch_output.unsqueeze(0).repeat(bs, 1, 1)
            if self.singlebranch:
                k = k1
            else:
                k = k1[..., i : i + 1]
            e = paddle.bmm(duplicated_tensor, k)
            if i == 0:
                output = e
            else:
                output = paddle.cat([output, e], dim=-1)
        output = output.view(bs, -1, self.x_dim, self.x_dim, self.output_channel)
        return output[:, : output_times.size(1)]
