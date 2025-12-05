import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, Tensor
# from .hypernet import HyperNetwork


def adaptive_conv_factory(num_choice, conv_para):
    ada_conv = None
    if num_choice == 0:
        ada_conv = Depthwise(conv_para['out_dim'], conv_para['total_in_channels'])
    elif num_choice == 1:
        ada_conv = SliceParam(conv_para['total_in_channels'], conv_para['out_dim'])
    elif num_choice == 2:
        ada_conv = TemplateMixing(conv_para['total_in_channels'], conv_para['out_dim'],
                                  conv_para['num_templates'])
    elif num_choice == 3:
        ada_conv = HyperNet(conv_para['total_in_channels'], conv_para['z_dim'],
                            conv_para['hidden_dim'], conv_para['out_dim'])
    return ada_conv


def conv1x1(in_dim: int, out_dim: int, stride: int = 1) -> nn.Conv2d:
    """return 1x1 conv"""
    return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)


class Depthwise(nn.Module):
    def __init__(self, kernels_per_channel, total_in_channels):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()

        self.kernels_per_channel = kernels_per_channel

        ## all channels in this order (alphabet): ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']

        self.conv1depthwise_param_bank = nn.Parameter(
            torch.zeros(total_in_channels * self.kernels_per_channel, 1, 3, 3)
        )

        nn.init.kaiming_normal_(
            self.conv1depthwise_param_bank, mode="fan_in", nonlinearity="relu"
        )

    def slice_params_first_layer(self, num_dim):
        # assert chunk in self.mapper, f"Invalid data_channel: {chunk}"

        ## conv1depthwise_param_bank's shape: (c_total * kernels_per_channel, 1, 3, 3)
        param_list = []
        for c in num_dim:
            param = self.conv1depthwise_param_bank[
                    c * self.kernels_per_channel: (c + 1) * self.kernels_per_channel, ...
                    ]
            param_list.append(param)
        params = torch.cat(param_list, dim=0)

        return params

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]

        ## slice params of the first layers
        conv1depth_params = self.slice_params_first_layer(list(range(c)))

        # assert len(self.mapper[chunk]) == c
        # assert conv1depth_params.shape == (c * self.kernels_per_channel, 1, 3, 3)

        out = F.conv2d(x, conv1depth_params, bias=None, stride=1, padding=1, groups=c)
        out = rearrange(out, 'b (c k) h w -> b c k h w', k=self.kernels_per_channel)
        out = out.mean(dim=1)
        return out


class SliceParam(nn.Module):
    def __init__(self, total_in_channels, out_dim):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()

        # total_in_channels = len(config.in_channel_names)

        ## all channels in this order (alphabet): ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']
        self.mapper = {
            "Allen": [5, 2, 6],
            "HPA": [3, 6, 5, 0],
            "CP": [5, 0, 7, 1, 4],
        }

        self.class_emb_idx = {
            "Allen": [0, 1, 2],
            "HPA": [3, 4, 5, 6],
            "CP": [7, 8, 9, 10, 11],
        }
        total_diff_class_channels = 12

        # out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        # self.stride = model.stem[0].stride
        # self.padding = model.stem[0].padding
        # self.dilation = model.stem[0].dilation
        # self.groups = model.stem[0].groups
        kh, kw = 3, 3

        self.conv1_param_bank = nn.Parameter(
            torch.zeros(out_dim, total_in_channels, kh, kw)
        )
        self.init_slice_param_bank_(None, None)

        self.class_emb = None

        ## Make a list to store reference for easy access

    def init_slice_param_bank_(
            self, total_in_channels: int, conv1_weight: Tensor
    ) -> None:
        """
        Initialize the first layer of the model
        conv1_weight: pre-trained weight, shape (original_out_dim, original_in_dim, kh, kw)
        """
        nn.init.kaiming_normal_(
            self.conv1_param_bank, mode="fan_in", nonlinearity="relu"
        )

    def slice_params_first_layer(self, dim_len):
        # assert chunk in self.mapper, f"Invalid data_channel: {chunk}"

        params = self.conv1_param_bank[:, :dim_len]
        if self.class_emb is not None:
            params = params + self.class_emb[:, :dim_len]
        return params

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_params = self.slice_params_first_layer(x.shape[1])
        x = F.conv2d(
            x,
            conv1_params,
            bias=None,
            stride=1,
            padding=1,
        )

        return x


class TemplateMixing(nn.Module):
    def __init__(self, total_channels, out_dim, num_templates):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()

        # num_templates = config.num_templates

        ## all channels in this order (alphabet): config.in_channel_names = ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']

        # self.mapper = {
        #     "Allen": [5, 2, 6],
        #     "HPA": [3, 6, 5, 0],
        #     "CP": [5, 0, 7, 1, 4],
        # }

        self.mapper = {
            "Allen": [(0, [5, 7]), (1, []), (2, [4])],
            "HPA": [(3, []), (4, [2]), (5, [0, 7]), (6, [8])],
            "CP": [(7, [0, 5]), (8, [6]), (9, []), (10, []), (11, [])],
        }

        kh, kw = 3, 3

        # total_channels = sum([len(v) for v in self.mapper.values()])
        assert (
                num_templates % total_channels == 0
        ), "num_templates must be divisible by total_channels"
        self.n_templ_per_channel = num_templates // total_channels

        # First conv layer
        self.conv1_param_bank = nn.Parameter(
            torch.zeros(out_dim, num_templates, kh, kw)
        )

        self.conv1_coefs = nn.Parameter(torch.zeros(total_channels, num_templates))

        nn.init.kaiming_normal_(
            self.conv1_param_bank, mode="fan_in", nonlinearity="relu"
        )

        self._init_conv1_coefs_()

        ## Make a list to store reference for easy access later on
        self.adaptive_interface = nn.ParameterList(
            [self.conv1_param_bank, self.conv1_coefs]
        )

        ## shared feature_extractor

    def _init_conv1_coefs_(self):
        ## generate random weight with normal distribution

        nn.init.normal_(self.conv1_coefs, mean=0.5, std=0.1)

        # for v in self.mapper.values():
        #     for c, shared_c_list in v:
        #         self.conv1_coefs.data[
        #         c, c * self.n_templ_per_channel: (c + 1) * self.n_templ_per_channel
        #         ] = 0.9
        #         for shared_c in shared_c_list:
        #             self.conv1_coefs.data[
        #             c,
        #             shared_c
        #             * self.n_templ_per_channel: (shared_c + 1)
        #                                         * self.n_templ_per_channel,
        #             ] = 0.1
        return None

    def mix_templates_first_layer(self, dim_len) -> Tensor:
        """
        @return: return a tensor, shape (out_channels, in_channels, kernel_h, kernel_w)
        """
        # assert chunk in self.mapper, f"Invalid chunk: {chunk}"
        # idx = [c for c, _ in self.mapper[chunk]]
        # idx = list(range(idx[0] * self.n_templ_per_channel, (idx[-1] + 1) * self.n_templ_per_channel))

        coefs = self.conv1_coefs[:dim_len]

        coefs = rearrange(coefs, "c t ->1 c t 1 1")
        templates = repeat(
            self.conv1_param_bank, "o t h w -> o c t h w", c=dim_len
        )
        params = torch.sum(coefs * templates, dim=2)
        return params

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_params = self.mix_templates_first_layer(x.shape[1])
        x = F.conv2d(
            x,
            conv1_params,
            bias=None,
            stride=1,
            padding=1,
        )

        return x


class HyperNet(nn.Module):
    def __init__(self, total_in_channels, z_dim, hidden_dim, out_dim):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()

        # total_in_channels = len(config.in_channel_names)

        ## all channels in this order (alphabet): ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']
        self.mapper = {
            "Allen": [5, 2, 6],
            "HPA": [3, 6, 5, 0],
            "CP": [5, 0, 7, 1, 4],
        }
        # out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        # self.stride = model.stem[0].stride
        # self.padding = model.stem[0].padding
        # self.dilation = model.stem[0].dilation
        # self.groups = model.stem[0].groups

        # First conv layer
        kh = 3

        self.conv1_emb = nn.Embedding(total_in_channels, z_dim)
        self.separate_emb = False
        self.hypernet = HyperNetwork(z_dim, hidden_dim, kh, out_dim, 1)

        ## Make a list to store reference to `conv1_emb` and `hypernet` for easy access

    def generate_params_first_layer(self, dim_len) -> Tensor:
        # assert chunk in self.mapper, f"Invalid chunk: {chunk}"
        if self.separate_emb:
            z_emb = self.conv1_emb[dim_len]
        else:
            z_emb = self.conv1_emb(
                torch.tensor(
                    list(range(dim_len)),
                    dtype=torch.long,
                    device=self.conv1_emb.weight.device,
                )
            )

        kernels = self.hypernet(z_emb)
        return kernels

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_params = self.generate_params_first_layer(x.shape[1])
        x = F.conv2d(
            x,
            conv1_params,
            bias=None,
            stride=1,
            padding=1,
        )

        return x


if __name__ == '__main__':
    # model = Depthwise(5, 40)
    # model = HyperNet(40, 64, 120, 64)
    model = HyperNet(40, 64, 120, 64)
    x = torch.rand((1, 20, 224, 224))
    y = model(x)
    print(y.shape)
    # mapper = {
    #     "Allen": [(0, [5, 7]), (1, []), (2, [4])],
    #     "HPA": [(3, []), (4, [2]), (5, [0, 7]), (6, [8])],
    #     "CP": [(7, [0, 5]), (8, [6]), (9, []), (10, []), (11, [])],
    # }
    # idx = [c for c, _ in mapper["CP"]]
    # print(idx)
