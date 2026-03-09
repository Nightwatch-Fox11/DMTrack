import torch
import math
import torch.nn as nn


class Mlp_2(nn.Module):
    """Multilayer perceptron."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)

        # nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        # x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        # x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        # print("return adap x", x_up.size())
        return x_up


class Tem_adapter(nn.Module):
    def __init__(self, dim=12, kernel_size=3):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
        )

        # nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # self.act = QuickGELU()
        self.dropout = nn.Dropout(0.2)
        self.dim = dim

    def forward(self, x, len_x=256, len_z=64):
        B, N, C = x.shape

        T = (N - len_x) // len_z
        # Hz = Wz = round(math.sqrt(len_z))
        x_down = self.adapter_down(x)

        x_x = x_down[:, len_z * T:]
        x_z = x_down[:, :len_z * T].view(B, T, len_z, -1).permute(0, 2, 3, 1).contiguous().flatten(0,
                                                                                                   1)  # (B, TN, C) -> (B, T, N, C) -> (B, N, C, T) -> (BN, C, T)
        # x_down = self.act(x_down)
        x_z = x_z + self.adapter_mid(x_z)
        x_z = x_z.view(B, len_z, -1, T).permute(0, 3, 1, 2).flatten(1, 2)

        x_down = torch.cat([x_z, x_x], dim=1)

        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        # print("return adap x", x_up.size())
        return x_up


class G_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)

        # Gemini part
        self.cross_heads = 8
        self.cross_attn_0_to_1 = nn.MultiheadAttention(dim, self.cross_heads, dropout=0.0, batch_first=False)
        self.cross_attn_1_to_0 = nn.MultiheadAttention(dim, self.cross_heads, dropout=0.0, batch_first=False)

        self.relation_judger = nn.Sequential(Mlp_2(dim * 2, dim, dim), torch.nn.Softmax(dim=-1))

        self.k_noise = nn.Embedding(2, dim)
        self.v_noise = nn.Embedding(2, dim)

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x, xi):
        x_down = self.adapter_down(x)
        xi_down = self.adapter_down(xi)

        """Parallized Gemini"""
        ## 1. 0_to_1 cross attn and skip connect
        B, N, C = x_down.shape
        q = x_down  # (B, N, C)

        judger_input = torch.cat([x_down, xi_down], dim=-1)  # (B, N, 2C)

        relation_score = self.relation_judger(judger_input)  # (B, N, C)

        noise_k = self.k_noise.weight[0] + q  # (B, N, C)
        noise_v = self.v_noise.weight[0] + q

        k = torch.cat([noise_k.unsqueeze(1), torch.mul(q, relation_score).unsqueeze(1)], dim=1)  # (B, 2, N, C)

        v = torch.cat([noise_v.unsqueeze(1), xi_down.unsqueeze(1)], dim=1)  # (B, 2, N, C)

        cross_0_to_1 = torch.vmap(self.cross_attn_0_to_1)(q.unsqueeze(1), k, v)[0].squeeze(1)

        ## 2. 1_to_0 cross attn and skip connect
        q = xi_down  # (B, N, C)

        judger_input = torch.cat([xi_down, x_down], dim=-1)  # (B, N, 2C)

        relation_score = self.relation_judger(judger_input)  # (B, N, C)

        noise_k = self.k_noise.weight[1] + q  # (B, N, C)
        noise_v = self.v_noise.weight[1] + q

        k = torch.cat([noise_k.unsqueeze(1), torch.mul(q, relation_score).unsqueeze(1)], dim=1)  # (B, 2, N, C)
        v = torch.cat([noise_v.unsqueeze(1), x_down.unsqueeze(1)], dim=1)  # (B, 2, N, C)

        cross_1_to_0 = torch.vmap(self.cross_attn_1_to_0)(q.unsqueeze(1), k, v)[0].squeeze(1)

        cross_0_to_1 = self.dropout(cross_0_to_1)
        cross_1_to_0 = self.dropout(cross_1_to_0)

        cross_0_to_1 = self.adapter_up(cross_0_to_1)
        cross_1_to_0 = self.adapter_up(cross_1_to_0)
        return cross_0_to_1, cross_1_to_0