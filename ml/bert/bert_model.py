from torch import nn
from transformers import AutoModel


class CustomBertModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone_name = backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        self.pool = ClsPool()

    def forward(self, x):
        x = self.backbone(**x)["last_hidden_state"]
        x = self.pool(x)

        return x


class ClsPool(nn.Module):
    def forward(self, x):
        # batch * num_tokens * num_embedding
        return x[:, 0, :]
