# TODO train this for at least 20 epochs, 
import math 
from typing import Tuple 
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator

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

class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.embedding_dim = self.dino.embed_dim

        self.fc = nn.Sequential(
            nn.Linear(self.dino.embed_dim, embedding_dim),
            nn.ReLU()
        )

        self.freeze()

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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.vocabulary_size,embedding_dim=self.embedding_dim),
            torch.nn.Dropout(0.5))
        

        self.to_logits = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

        self.transformer_decoder = TransformerModel(
            ntoken=vocabulary_size,
            d_model=self.embedding_dim,
            nhead =2,
            d_hid = self.hidden_dim,
            nlayers= self.num_layers,
            )
        
        self.image_projection = nn.Linear(self.embedding_dim,self.embedding_dim)

    def freeze(self):
        pass

    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        if caption_indices is None:
            embeddings = rearrange(encoded_image, 'batch embedding_dim -> batch 1 embedding_dim')
        else:
            embeddings = self.embedding(caption_indices)
            if encoded_image is not None:
                embeddings, _ = pack([encoded_image, embeddings], 'batch * embedding_dim')

        return embeddings

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        """Forward method.
        :param encoded_image: torch.tensor of the shape [batch_size, *] or None
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length] or None

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

################################################################################################################################################

        #1. pre-fed vectors encoded-image
        #2. embeddings using tokenization like BERT
        #3. positional embeddings instead of RNN
        #4. global self-attention /causal         
        batch_size,seq_len = caption_indices.size()

        encoded_image = self.image_projection(encoded_image).unsqueeze(0)

        caption_indices = caption_indices.transpose(0, 1)
        # self-attention layer
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(caption_indices.device)

    
        logits = self.transformer_decoder(encoded_image, src_mask=tgt_mask)
        
################################################################################################################################################

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
       
        caption_indices = []

        output = self.forward(encoded_image, caption_indices=None, hidden_state=None)
        for _ in range(max_length):
            predicted_index = output['indices']

            caption_indices.append(predicted_index.item())
            if predicted_index.item() == eos_token_index:
                break

            output = self.forward(encoded_image=None,
                                  caption_indices=predicted_index,
                                  hidden_state=output['hidden_state'])

        return caption_indices