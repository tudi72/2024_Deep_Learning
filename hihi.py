from copy import deepcopy
from typing import Tuple
import math
import torch
from torch import nn, Tensor
from torch.nn import MultiheadAttention
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator
import torch.nn.functional as F
from einops import rearrange, pack

class Model(BaseModel):
    """Base class for all models."""

    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.embedding_dim,
                                                  num_layers=self.num_layers)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,dropout: float):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,tgt_mask: Tensor,tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
     
        output, _ = self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,attn_mask=tgt_mask,key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns

class Decoder(nn.Module):
    def __init__(self,layer,vocab_size: int,d_model: int,num_layers: int, max_len =5000, dropout = 0.2):
        super().__init__()

        self.pad_id = 0
        self.cptn_emb = nn.Embedding(vocab_size, d_model,padding_idx=self.pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)


        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,src_img: Tensor) -> Tuple[Tensor, Tensor]:
    
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])

        tgt_mask = tgt_mask.to(tgt_cptn.device)

        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        # CAPTION/TGT_CPTN = [256,31]
        # SRC IMG = [256,128]
        # TGT MASK = [31,31]
        # stopped here 
        for layer in self.layers:
            # [ERROR]: boolean value of Tensor with more than one value is ambiguous
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)

        attns_all = torch.stack(attns_all)
        print("[TGT_CPTN ]",tgt_cptn.shape)
        print("[ATTNS_ALL ]",attns_all.shape)

        return tgt_cptn, attns_all

class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.embedding_dim = self.dino.embed_dim

        self.freeze()
        
        self.fc = nn.Sequential(
            nn.Linear(self.dino.embed_dim, embedding_dim),
            nn.ReLU()
        )


    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, image):
        scale: int = 1

        resized_image = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)
        
        intermediate_layers = self.dino.get_intermediate_layers(resized_image, n=1, reshape=True, return_class_token=True)[0]

        cls_token, patch_tokens = intermediate_layers[1], intermediate_layers[0]

        encoding = self.fc(cls_token)

        return encoding


class CaptionGenerator(BaseCaptionGenerator):

    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # self.decoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.embedding_dim,
        #     nhead=4,
        #     dim_feedforward=hidden_dim,
        #     dropout=0.1)
        
        self.decoder_layer = DecoderLayer(d_model=self.embedding_dim,
                                     num_heads=4,
                                     feedforward_dim=self.hidden_dim,
                                     dropout=0.1)
        self.decoder = Decoder(layer=self.decoder_layer,vocab_size=vocabulary_size,d_model=self.embedding_dim,num_layers=num_layers)

        self.to_logits = nn.Linear(self.embedding_dim, vocabulary_size, bias=False)

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        pass 

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        
        print("[CAPTION ]",caption_indices.shape)
        print("[ENCODED IMG ]",encoded_image.shape)
        # captions = [batch_size, max_len-1=51]
        #[encode_size^2, batch_size, image_embed_dim]
        outputs, attns = self.decoder(caption_indices, encoded_image)

        predictions = self.to_logits(outputs).permute(1, 0, 2)

        logits = predictions.contiguous()
    
        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    
def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = []

        output = self.forward(encoded_image, caption_indices=None, hidden_state=None)
        for _ in range(max_length):
            predicted_index = output['indices']

            caption_indices.append(predicted_index.item())
            if predicted_index.item() == eos_token_index:
                break

            output = self.forward(encoded_image=None,caption_indices=predicted_index,hidden_state=output['hidden_state'])

        return caption_indices


