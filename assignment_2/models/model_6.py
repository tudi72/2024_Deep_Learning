from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):

    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        print("number of layers ", num_layers)

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
        
        # TODO replace Cls_token with spat
        intermediate_layers = self.dino.get_intermediate_layers(resized_image, n=1, reshape=True, return_class_token=True)[0]

        # [256,384,16,16]
        cls_token, patch_tokens = intermediate_layers[1], intermediate_layers[0]

        # encoding = self.fc(patch_tokens)

        return patch_tokens

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, num_features, feature_dim]

        query = self.query_proj(hidden)  # [batch_size, hidden_dim]
        key = self.key_proj(encoder_outputs)  # [batch_size, num_features, hidden_dim]
        value = self.value_proj(encoder_outputs)  # [batch_size, num_features, hidden_dim]

        # Calculate attention scores
        scores = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))  # [batch_size, 1, num_features]
        attention_weights = self.softmax(scores.squeeze(1))  # [batch_size, num_features]

        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), value).squeeze(1)  # [batch_size, hidden_dim]

        return context, attention_weights


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
            batch_first=True)

        self.cross_attention = CrossAttention(feature_dim=self.hidden_dim,hidden_dim=self.hidden_dim)

        self.to_logits = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size)

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

        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

#############################################################################################################################################
        # EMBEDDINGS        [256,384, 16,16]
        print("[EMBEDDING].1")
        if hidden_state is None:
            print("[EMBEDDING].2")
            hidden_state = (torch.zeros(self.num_layers, 256, self.hidden_dim).to(encoded_image.device),
                            torch.zeros(self.num_layers, 256, self.hidden_dim).to(encoded_image.device))

        outputs = []
        for t in range(embeddings.size(1)):
            print("[EMBEDDING].3")
            context, _ = self.cross_attention(hidden_state[0][-1], encoded_image)
            print("[EMBEDDING].4")
            rnn_input = torch.cat((embeddings[:, t, :], context), dim=1).unsqueeze(1)
            print("[EMBEDDING].5")
            output, hidden_state = self.rnn(rnn_input, hidden_state)
            print("[EMBEDDING].6")
            outputs.append(output)

        print("[EMBEDDING].7")
        outputs = torch.cat(outputs, dim=1)

        #RuntimeError: Expected size 16 but got size 128 for tensor number 1.
 ############################################################################################################################################

        logits = self.to_logits(outputs)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

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