import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from einops import rearrange, pack
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator

from copy import deepcopy
from typing import Tuple


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.embedding_dim = self.dino.embed_dim

        self.enc_self_attn = EncSelfAttension(img_embed_dim=self.embedding_dim,
                                              num_heads=4,
                                              dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(self.dino.embed_dim, embedding_dim),
            nn.ReLU()
        )

        self.freeze()

    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, image):
        """
        enc_inputs:   [encode_size^2, batch_size, embed_dim]
        enc_outputs:  [encode_size^2, batch_size, embed_dim]
        """
        scale: int = 1

        resized_image = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)

        # some self-attention
        resized_image = self.enc_self_attn(resized_image)

        intermediate_layers = self.dino.get_intermediate_layers(resized_image, n=1, reshape=True, return_class_token=True)[0]

        cls_token, patch_tokens = intermediate_layers[1], intermediate_layers[0]

        encoding = self.fc(cls_token)


        return encoding
    
class EncSelfAttension(nn.Module):

    def __init__(self, img_embed_dim: int, num_heads: int, dropout: float):
        super(EncSelfAttension, self).__init__()
        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout)
        self.layer_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        """
        enc_inputs:   [encode_size^2, batch_size, embed_dim]
        enc_outputs:  [encode_size^2, batch_size, embed_dim]
        """

        enc_outputs, _ = self.multi_head_attn(enc_inputs, enc_inputs,
                                              enc_inputs)
        enc_outputs = enc_outputs + enc_inputs
        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
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

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        param:
        dec_inputs:    [max_len, batch_size, embed_dim]
        enc_outputs:   [encode_size^2=196, batch_size, embed_dim]
        tgt_mask:      [max_len , max_len]
        tgt_pad_mask:  [batch_size , max_len]
        output:        [max_len, batch_size, embed_dim]
        attn:          [layer_num, batch_size, head_num, max_len, encode_size^2]
        """
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns



from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, Tensor

from .encoder_layers import EncoderLayer
from .decoder_layers import DecoderLayer
from .pe import PositionalEncoding


class Encoder(nn.Module):
    """
    param:

    layer:      an instance of the EecoderLayer() class

    num_layers: the number of decoder-layers
                int
    """

    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        # Make copies of the encoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        """
        param:
        x:  encoder input
            Tensor
            [encode_size^2, batch_size, image_embed_dim]

        outputs:
        x:  encoder output
            Tensor
            [encode_size^2, batch_size, model_embed_dim]
        """

        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    """
    layer:          an instance of the EecoderLayer() class
    vocab_size:     the number of vocabulary
    d_model:        size of features in the transformer inputs
    num_layers:     the number of decoder-layers
    max_len:        maximum len pf target captions
    dropout:        dropout value
    pad_id:         padding token id
    """
    def __init__(self,
                 layer: DecoderLayer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.pad_id = pad_id

        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,
                src_img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        tgt_cptn:   [batch_size, max_len-1]
        src_img:    [encode_size^2, batch_size, image_embed_dim]
        output:     [max_len, batch_size, model_embed_dim]
        attn_all:   [layer_num, batch_size, head_num, max_len-1,encode_size^2]
        """
        # create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all


class Transformer(nn.Module):
    """
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 img_encode_size: int,
                 enc_ff_dim: int,
                 dec_ff_dim: int,
                 enc_n_layers: int,
                 dec_n_layers: int,
                 enc_n_heads: int,
                 dec_n_heads: int,
                 max_len: int,
                 dropout: float = 0.1,
                 pad_id: int = 0):
        super(Transformer, self).__init__()
        encoder_layer = EncoderLayer(img_encode_size=img_encode_size,
                                     img_embed_dim=d_model,
                                     feedforward_dim=enc_ff_dim,
                                     num_heads=enc_n_heads,
                                     dropout=dropout)
        decoder_layer = DecoderLayer(d_model=d_model,
                                     num_heads=dec_n_heads,
                                     feedforward_dim=dec_ff_dim,
                                     dropout=dropout)
        
        self.decoder = Decoder(layer=decoder_layer,
                               vocab_size=vocab_size,
                               d_model=d_model,
                               num_layers=dec_n_layers,
                               max_len=max_len,
                               dropout=dropout,
                               pad_id=pad_id)

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images_encoded: Tensor,captions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        images [batch_size, encode_size^2=196, image_feature_size=512]

        captions:   target captions
                    [batch_size, max_len-1=51]

        outputs:
        predictions:    Decoder output
                        Tensor
                        [batch_size, max_len, vocab_size]

        attn_all:       Attension weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
                        See comments in decoder_layers.DecoderLayer
        """
        # encode, decode, predict
        tgt_cptn, attns = self.decoder(captions, images_encoded)
        predictions = self.predictor(tgt_cptn).permute(1, 0, 2) 

        return predictions.contiguous(), attns.contiguous()
