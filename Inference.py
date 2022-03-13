import torch
import numpy as np
import logging, yaml, os, sys, argparse, math
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
from librosa import griffinlim
from scipy.io import wavfile

from Modules.Modules import MLPSinger
from Datasets import Convert_Feature_Based_Music, Lyric_to_Token, Token_Stack, Note_Stack
from meldataset import spectral_de_normalize_torch
from Arg_Parser import Recursive_Parse

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],        
        singer_info_dict: Dict[str, int],
        genre_info_dict: Dict[str, int],
        message_times_list: List[float],
        lyrics: List[List[str]],
        notes: List[List[int]],
        singers: List[List[str]],
        genres: List[str],
        sample_rate: int,
        frame_shift: int,
        equality_duration: bool= False,
        consonant_duration: int= 3
        ):
        super().__init__()
        self.token_dict = token_dict
        self.singer_info_dict = singer_info_dict
        self.genre_info_dict = genre_info_dict
        self.equality_duration = equality_duration
        self.consonant_duration = consonant_duration

        self.patterns = []
        for message_time, lyric, note, singer, genre in zip(message_times_list, lyrics, notes, singers, genres):
            text = lyric
            lyric, note = Convert_Feature_Based_Music(
                music= list(zip(message_time, lyric, note)),
                sample_rate= sample_rate,
                frame_shift= frame_shift,
                consonant_duration= consonant_duration,
                equality_duration= equality_duration
                )
            singer = self.singer_info_dict[singer]
            genre = self.genre_info_dict[genre]

            self.patterns.append((lyric, note, singer, genre, text))

    def __getitem__(self, idx):
        lyric, note, singer, genre, text = self.patterns[idx]

        return Lyric_to_Token(lyric, self.token_dict), note, singer, genre, text

    def __len__(self):        
        return len(self.patterns)

class Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        pattern_length: int
        ):
        self.token_dict = token_dict
        self.pattern_length = pattern_length
         
    def __call__(self, batch):
        tokens, notes, singers, genres, lyrics = zip(*batch)
        
        lengths = np.array([len(token) for token in tokens])
        max_length = lengths.max() + self.pattern_length - (lengths.max() % self.pattern_length)

        tokens = Token_Stack(tokens, self.token_dict, max_length)
        notes = Note_Stack(notes, max_length)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        lengths = torch.LongTensor(lengths)   # [Batch]
        notes = torch.LongTensor(notes)   # [Batch, Time]
        singers = torch.LongTensor(singers)  # [Batch]
        genres = torch.LongTensor(genres)  # [Batch]
        
        lyrics = [''.join([(x if x != '<X>' else ' ') for x in lyric]) for lyric in lyrics]

        return tokens, notes, singers, genres, lengths, lyrics

class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,
        out_path: str,
        batch_size= 1
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = MLPSinger(self.hp).to(self.device)
        if self.hp.Feature_Type == 'Mel':
            self.vocoder = torch.jit.load('vocgan_sing_mzf_22k_403.pts', map_location='cpu').to(self.device)

        self.Load_Checkpoint(checkpoint_path)
        self.out_path = out_path
        self.batch_size = batch_size

    def Dataset_Generate(self, message_times_list, lyrics, notes, singers, genres):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        singer_info_dict = yaml.load(open(self.hp.Singer_Info_Path), Loader=yaml.Loader)
        genre_info_dict = yaml.load(open(self.hp.Genre_Info_Path), Loader=yaml.Loader)

        return torch.utils.data.DataLoader(
            dataset= Dataset(
                token_dict= token_dict,
                singer_info_dict= singer_info_dict,
                genre_info_dict= genre_info_dict,
                message_times_list= message_times_list,
                lyrics= lyrics,
                notes= notes,
                singers= singers,
                genres= genres,
                sample_rate= self.hp.Sound.Sample_Rate,
                frame_shift= self.hp.Sound.Frame_Shift,
                equality_duration= self.hp.Duration.Equality,
                consonant_duration= self.hp.Duration.Consonant_Duration
                ),
            shuffle= False,
            collate_fn= Collater(
                token_dict= token_dict,
                pattern_length= self.hp.Train.Pattern_Length
                ),
            batch_size= self.batch_size,
            num_workers= 0,
            pin_memory= True
            )

    def Load_Checkpoint(self, path):
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])        
        self.steps = state_dict['Steps']

        self.model.eval()

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    @torch.no_grad()
    def Inference_Step(self, tokens, notes, singers, genres, lengths, lyrics):
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        
        predictions = torch.cat([
            self.model(
                tokens=  tokens[:, index:index + self.hp.Train.Pattern_Length],
                notes= notes[:, index:index + self.hp.Train.Pattern_Length],
                )
            for index in range(0, lengths.max(), self.hp.Train.Pattern_Length)
            ], dim= 2)
        predictions = (predictions + 1.0) / 2.0 * (2.0957 + 11.5129) - 11.5129
        
        if self.hp.Feature_Type == 'Mel':
            audios = self.vocoder(predictions)
            if audios.ndim == 1:
                audios = audios.unsqueeze(0)
            audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(audios, lengths)
                ]
        elif self.hp.Feature_Type == 'Spectrogram':
            audios = []
            for feature, length in zip(
                predictions,
                lengths
                ):                
                audio = griffinlim(
                    spectral_de_normalize_torch(feature).cpu().numpy(),
                    hop_length= 256,
                    win_length= 1024,
                    center= False
                    )
                audio = audio[:min(length * self.hp.Sound.Frame_Shift, audio.shape[0])]
                audio = (audio / np.abs(audio).max() * 32767.5).astype(np.int16)
                audios.append(audio)

        return audios

    def Inference_Epoch(self, message_times_list, lyrics, notes, singers, genres, use_tqdm= True):
        dataloader = self.Dataset_Generate(
            message_times_list= message_times_list,
            lyrics= lyrics,
            notes= notes,
            singers= singers,
            genres= genres
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )
        
        export_audios = []
        for tokens, notes, singers, genres, lengths, lyrics in dataloader:
            audios = self.Inference_Step(tokens, notes, singers, genres, lengths, lyrics)
            export_audios.append(audios)

        export_audios = [audio for audios in export_audios for audio in audios]
        
        return export_audios

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-checkpoint', '--checkpoint', type= str, required= True)
    parser.add_argument('-outdir', '--outdir', type= str, required= True)
    parser.add_argument('-batch', '--batch', default= 1, type= int)
    args = parser.parse_args()
    
    inferencer = Inferencer(
        hp_path= args.hyper_parameters,
        checkpoint_path= args.checkpoint,
        out_path= args.outdir,
        batch_size= args.batch
        )

    patterns = []
    for path in [
        # './Inference_for_Training/Example1.txt',
        # './Inference_for_Training/Example2.txt',
        # './Inference_for_Training/Example3.txt',
        './Inference_for_Training/Example4.txt',
        './Inference_for_Training/Example5.txt',
        ]:        
        pattern = []
        for line in open(path, 'r', encoding= 'utf-8').readlines()[1:]:
            duration, text, note = line.strip().split('\t')
            pattern.append((int(duration), text, int(note)))
        patterns.append(pattern)
    audios = inferencer.Inference_Epoch(patterns, True)

# python Inference.py -hp Hyper_Parameters.yaml -checkpoint /data/results/MLPSinger/MLPSinger.Spect/Checkpoint/S_100000.pt -outdir ./results/