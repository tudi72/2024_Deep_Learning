from copy import deepcopy
from typing import Tuple
import math
import torch
from torch import nn, Tensor
from torch.nn import MultiheadAttention
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


class CNNFeedForward(nn.Module):
    def __init__(self, encode_size: int, embed_dim: int, feedforward_dim: int,dropout: float):
        super(CNNFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=encode_size,out_channels=feedforward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim,out_channels=encode_size,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv2(self.relu(self.conv1(inputs.permute(1, 0, 2))))
        output = self.dropout(output)  # type: Tensor
        return self.layer_norm(output.permute(1, 0, 2) + inputs)

class EncSelfAttension(nn.Module):

    def __init__(self, img_embed_dim: int, num_heads: int, dropout: float):
        super(EncSelfAttension, self).__init__()
        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim,num_heads=num_heads,dropout=dropout)
        self.layer_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, enc_inputs: Tensor) -> Tensor:

        enc_outputs, _ = self.multi_head_attn(enc_inputs, enc_inputs,enc_inputs)
        enc_outputs = enc_outputs + enc_inputs
        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,dropout: float):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiheadAttention(d_model,num_heads,dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,num_heads,dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),nn.ReLU(inplace=True), nn.Dropout(p=dropout),nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,tgt_mask: Tensor,tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        output, _ = self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,attn_mask=tgt_mask,key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)

        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output) 
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns

class EncoderLayer(nn.Module):

    def __init__(self, img_encode_size: int, img_embed_dim: int,feedforward_dim: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = EncSelfAttension(img_embed_dim=img_embed_dim,num_heads=num_heads,dropout=dropout)
        self.cnn_ff = CNNFeedForward(encode_size=img_encode_size,embed_dim=img_embed_dim,feedforward_dim=feedforward_dim,dropout=dropout)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs = self.enc_self_attn(enc_inputs)
        enc_outputs = self.cnn_ff(enc_outputs)
        return enc_outputs

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

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)

        return x

class Decoder(nn.Module):
    def __init__(self,layer: DecoderLayer,vocab_size: int,d_model: int,num_layers: int, max_len =5000, dropout = 0.2):
        super().__init__()

        self.cptn_emb = nn.Embedding(vocab_size, d_model)
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
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all

class CaptionGenerator(BaseCaptionGenerator):

    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        encoder_layer = EncoderLayer(img_encode_size=embedding_dim,img_embed_dim=hidden_dim,feedforward_dim=hidden_dim,num_heads=4,dropout=0.2)
        self.encoder = Encoder(layer=encoder_layer, num_layers=num_layers)

        decoder_layer = DecoderLayer(d_model=self.embedding_dim,num_heads=4,feedforward_dim=hidden_dim,dropout=0.2)
        self.decoder = Decoder(layer=decoder_layer,vocab_size=vocabulary_size,d_model=self.embedding_dim,num_layers=num_layers)

        self.to_logits = nn.Linear(self.embedding_dim, vocabulary_size, bias=False)

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        raise NotImplementedError

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        
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


