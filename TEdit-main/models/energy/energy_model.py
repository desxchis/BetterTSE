import torch
from torch import nn
import torch.nn.functional as F

from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PatchEmbedding
from models.encoders.attr_encoder import AttributeEncoder


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class EnergyModel(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs): #default patch_len=16, stride=8
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.instance_norm = configs["instance_norm"]
        self.seq_len = configs["seq_len"]
        patch_len = configs["patch_len"]
        stride = configs["stride"]
        padding = stride

        # diffusion step embeddings
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=configs["num_steps"],
            embedding_dim=configs["d_model"],
        )

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs["d_model"], patch_len, configs["n_var"], stride, padding, configs["dropout"])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs["factor"], attention_dropout=configs["dropout"],
                                      output_attention=configs["output_attention"]), configs["d_model"], configs["n_heads"]),
                    configs["d_model"],
                    configs["d_ff"],
                    dropout=configs["dropout"],
                    activation=configs["activation"]
                ) for l in range(configs["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(configs["d_model"])
        )

        # attribute encoder
        self.device = configs["device"]
        configs["attrs"]["device"] = configs["device"]
        self.attr_encoder = AttributeEncoder(configs["attrs"])
        self.n_attrs = len(configs["attrs"]["num_attr_ops"])
        self.n_attr_ops = configs["attrs"]["num_attr_ops"]

        # Prediction Head
        self.head_nf = configs["d_model"] * \
                       int((configs["seq_len"] - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs["dropout"])
        self.projection = nn.Linear(self.head_nf, configs["d_model"]*self.n_attrs)

        self.proj_a = nn.Linear(configs["d_model"], configs["d_model"])
        self.proj_a_out = nn.Linear(configs["d_model"], configs["d_model"])

        self.project1 = nn.ModuleList([nn.Linear(configs["d_model"], configs["d_model"]) for i in range(self.n_attrs)])
        
        self.d_model = configs["d_model"]
        self.CE = nn.CrossEntropyLoss(reduction="none")
        self.norm_in_loss = configs["norm_in_loss"]

    def forward(self, x, attrs):
        """
        x: (B,V,L)
        attrs: (B,K)
        """
        x_emb = self.encode_ts(x.permute(0,2,1))  # (B,N,d)
        a_emb = self.attr_encoder(attrs)  # (B,N,d)
        return x_emb, a_emb

    def get_tap_loss(self, x, attrs):
        """
        x: (B,V,L)
        attrs: (B,N)
        """
        B, N = attrs.shape

        x_emb, a_emb = self.forward(x, attrs)  # (B,N,d), (B,N,d)

        if self.norm_in_loss:
            x_emb = x_emb/torch.norm(x_emb, dim=-1, keepdim=True)
            a_emb = a_emb/torch.norm(a_emb, dim=-1, keepdim=True)
        x_emb = x_emb.permute(1,0,2)  # (N,B,d)
        a_emb = a_emb.permute(1,2,0) # (B,N,d) -> (N,d,B)

        sim = torch.bmm(x_emb, a_emb)  # (N,B,B)

        labels = torch.arange(sim.shape[1], device=sim.device)  # (B)
        labels = labels.expand([sim.shape[0], sim.shape[1]])  # (N,B)
        labels = torch.reshape(labels, [-1])  # (N*B)
        loss_x2a = self.CE(torch.reshape(sim, [N*B,-1]), labels).reshape(N,B)
        sim = sim.permute(0, 2, 1)
        loss_a2x = self.CE(torch.reshape(sim, [N*B,-1]), labels).reshape(N,B)

        loss_x2a = torch.mean(loss_x2a, dim=-1)
        loss_a2x = torch.mean(loss_a2x, dim=-1)

        loss_x2a = torch.mean(loss_x2a)
        loss_a2x = torch.mean(loss_a2x)

        return loss_x2a, loss_a2x
    
    def get_ts2all_attr_scores_embs(self, x):
        """
        Cosine similarity.
        """
        x_emb = self.encode_ts(x.permute(0,2,1)) # (B,N,d)
        a_emb = self.attr_encoder.get_all_embs()  # list of attributes [(N1,d),(N2,d)...]
        cos_list = []
        for i, a_e in enumerate(a_emb):
            a_e = torch.unsqueeze(a_e, 0)  # (1,Nk,d)
            x_e = x_emb[:, i:i+1]  # (B,1,d)
            x_e = x_e/torch.norm(x_e, dim=-1, keepdim=True)
            a_e = a_e/torch.norm(a_e, dim=-1, keepdim=True)
            s = torch.sum(x_e*a_e, dim=-1)  # (B,Nk)
            cos_list.append(s)
        return cos_list, x_emb
    
    def encode_ts(self, x_enc, soft_mask=None, keys_null=None, values_null=None):
        """
        Encode time series with optional Soft-Boundary Injection.
        
        Args:
            x_enc: Input time series [B, L, n_var]
            soft_mask: Soft boundary mask [B, L] or [B, L, 1]
            keys_null: Key tensor for null/background condition
            values_null: Value tensor for null/background condition
        
        Returns:
            Encoded time series features [B, n_attrs, d_model]
        """
        if self.instance_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.patch_embedding(x_enc)

        enc_out, attns = self.encoder(
            enc_out,
            soft_mask=soft_mask,
            keys_null=keys_null,
            values_null=values_null
        )

        output = self.flatten(enc_out)
        output = self.dropout(output)

        output = self.projection(output)
        output = torch.reshape(output, (-1, self.n_attrs, self.d_model))
        output = F.gelu(output)
        output_list = []
        for i in range(self.n_attrs):
            o = self.project1[i](output[:,i,:])
            output_list.append(o)
        output = torch.stack(output_list, dim=1)
        return output    