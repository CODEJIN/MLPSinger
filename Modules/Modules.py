from argparse import Namespace
import torch
import math
from typing import Optional, List, Dict, Tuple

from .Layer import Linear, Lambda

class MLPSinger(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        encoding_size = self.hp.Encoder.Token_Size + self.hp.Encoder.Note_Size + self.hp.Encoder.Genre_Size
        
        if self.hp.Feature_Type == 'Spectrogram':
            feature_size = self.hp.Sound.N_FFT // 2 + 1
        elif self.hp.Feature_Type == 'Mel':
            feature_size = self.hp.Sound.Mel_Dim
        else:
            raise ValueError('Unknown feature type: {}'.format(self.hp.Feature_Type))

        self.encoder = Encoder(self.hp)
        
        self.mixer_blocks = torch.nn.Sequential()
        for index in range(self.hp.Mixer.Stack):
            self.mixer_blocks.add_module('Block_{}'.format(index), MixerBlock(
                in_features= encoding_size,
                calc_features= encoding_size * 4,
                in_steps= self.hp.Mixer.Pattern_Length,
                calc_steps= self.hp.Mixer.Pattern_Length * 4,
                dropout_rate= self.hp.Mixer.Dropout_Rate
                ))

        self.projection = torch.nn.Sequential(
            Linear(
                in_features= encoding_size,
                out_features= feature_size,
                w_init_gain= 'linear'
                ),
            Lambda(lambda x: x.permute(0, 2, 1))
            )

    def forward(
        self,
        tokens: torch.LongTensor,
        notes: torch.LongTensor,
        genres: torch.LongTensor,
        ):
        x = self.encoder(tokens, notes, genres)
        x = self.mixer_blocks(x)
        x = self.projection(x)

        return x

class Encoder(torch.nn.Module): 
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        encoding_size = self.hp.Encoder.Token_Size + self.hp.Encoder.Note_Size + self.hp.Encoder.Genre_Size

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Token_Size,
            )
        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Notes,
            embedding_dim= self.hp.Encoder.Note_Size,
            )
        self.gener_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Genres,
            embedding_dim= self.hp.Encoder.Genre_Size,
            )
        self.linear = Linear(
            in_features= encoding_size,
            out_features= encoding_size,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        tokens: torch.Tensor,
        notes: torch.Tensor,
        genres: torch.Tensor,
        ):
        '''
        tokens: [Batch, Time]
        notes: [Batch, Time]
        genres: [Batch]
        '''
        x = torch.cat([
            self.token_embedding(tokens),
            self.note_embedding(notes),
            self.gener_embedding(genres).unsqueeze(1).expand(-1, tokens.size(1), -1),
            ], dim= 2)

        x = self.linear(x)

        return x    # [Batch, Time, Dim]

class MixerBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        calc_features: int,
        in_steps: int,
        calc_steps: int,
        dropout_rate: float= 0.1,
        layer_norm_eps=1e-5,
        ) -> None:
        super().__init__()
        
        self.channel_mixer = Mixer(
            transpose= False,
            in_features= in_features,
            calc_features= calc_features,
            dropout_rate= dropout_rate,
            layer_norm_eps= layer_norm_eps
            )
        self.token_mixer = Mixer(
            transpose= True,
            in_features= in_steps,
            calc_features= calc_steps,
            dropout_rate= dropout_rate,
            layer_norm_eps= layer_norm_eps,
            layer_norm_features= in_features
            )

    def forward(
        self,
        x: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Time, Dim]
        '''
        x = self.channel_mixer(x) + x
        x = self.token_mixer(x) + x

        return x

class Mixer(torch.nn.Module):
    def __init__(
        self,
        transpose: bool,
        in_features: int,
        calc_features: int,
        dropout_rate: float= 0.5,
        layer_norm_eps: float=1e-5,
        layer_norm_features: int= None
        ) -> None:
        super().__init__()
        self.transpose = transpose

        self.norm = torch.nn.LayerNorm(layer_norm_features or in_features, eps=layer_norm_eps)
        self.feedforward = torch.nn.Sequential(
            Linear(
                in_features= in_features,
                out_features= calc_features,
                w_init_gain= 'relu'
                ),
            torch.nn.GELU(),
            torch.nn.Dropout(p= dropout_rate),
            Linear(
                in_features= calc_features,
                out_features= in_features,
                w_init_gain= 'linear'
                ),
            torch.nn.Dropout(p= dropout_rate)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: [Batch, Time, Dim]
        '''
        x = self.norm(x)
        if self.transpose:
            x = x.transpose(2, 1)   # [Batch, Dim, Time]
        x = self.feedforward(x)
        if self.transpose:
            x = x.transpose(2, 1)   # [Batch, Time, Dim]

        return x
