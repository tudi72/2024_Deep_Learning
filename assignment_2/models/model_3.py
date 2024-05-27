import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

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

        # loading pre-trained DINOv2 model
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # initializing the embedding_dim
        self.embedding_dim = self.dino.embed_dim

        self.fc = nn.Sequential(
            # linear layer that maps DINOv2 output to embedding dimension
            nn.Linear(self.dino.embed_dim, embedding_dim),
            # activation function f(x) = max(0, x)
            nn.ReLU()
        )

        # freeze the DINOv2 backbone
        self.freeze()

    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, image):
        scale: int = 1

        # extracting image features from DINOv2
        resized_image = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)
        outputs = self.dino.get_intermediate_layers(resized_image, n=1, reshape=True, return_class_token=True)[0]

        # extracting the <CLS> token representation
        # output shape is 256 x 384
        cls_token, patch_tokens = outputs[1], outputs[0]

        # encoding the extracted <CLS> token to the dimension of the embedding
        encoding = self.fc(cls_token)

        return encoding


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(torch.nn.Embedding(num_embeddings=self.vocabulary_size,
                                                                embedding_dim=self.embedding_dim),
                                             torch.nn.Dropout(0.5))
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True)

        self.to_logits = torch.nn.Linear(in_features=self.hidden_dim * 2, out_features=self.vocabulary_size)
  
    
    def init_hidden_state(self, encoded_image):
        # [4,256,128] invalid for 65_536
        # print(f"[ENCODED_IMAGE]: {encoded_image.shape}")
        batch_size = encoded_image.shape[0]

        # print(f"[BATCH SIZE]:{batch_size.shape}")

        hidden_gate = torch.randn(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        # print(f"[HIDDEN STATE]:{hidden_gate.shape}")

        cell_gate = torch.randn(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
   
        return (hidden_gate,cell_gate)

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
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
        :param args: e.g., hidden state

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

        if hidden_state is None:
            hidden_state = self.init_hidden_state(encoded_image)

        # print(f"[hidden_state]:{hidden_state.shape}")
        # [256,24,256]
        output, hidden_state = self.rnn(input=embeddings, hx=hidden_state)

        # LOGITS [256, 24, 2994]      
        logits = self.to_logits(output)

        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')
        # LOGITS [256, 2994, 24]        

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}


    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        """Generates caption indices like torch.tensor([1, 23, 5, 8, 2]).

        :param encoded_image: torch.tensor of the shape [1, *]
        :param sos_token_index: index of the "start of sequence" token (int)
        :param eos_token_index: index of the "end of sequence" token (int)
        :param max_length: maximum caption length (int)

        :return: caption indices (list of the length <= max_length)
        """
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