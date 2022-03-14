from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging
from typing import Dict, List, Optional
import hgtk

from Pattern_Generator import Convert_Feature_Based_Music

def Decompose(syllable: str):    
    onset, nucleus, coda = hgtk.letter.decompose(syllable)
    coda += '_'

    return onset, nucleus, coda

def Lyric_to_Token(lyric: List[str], token_dict: Dict[str, int]):
    return [
        token_dict[letter]
        for letter in list(lyric)
        ]

def Token_Stack(tokens: List[List[int]], token_dict: Dict[str, int], max_length: Optional[int]= None):
    max_token_length = max_length or max([len(token) for token in tokens])
    tokens = np.stack(
        [np.pad(token[:max_token_length], [0, max_token_length - len(token[:max_token_length])], constant_values= token_dict['<X>']) for token in tokens],
        axis= 0
        )
    return tokens

def Note_Stack(notes: List[List[int]], max_length: Optional[int]= None):
    max_note_length = max_length or max([len(note) for note in notes])
    notes = np.stack(
        [np.pad(note[:max_note_length], [0, max_note_length - len(note[:max_note_length])], constant_values= 0) for note in notes],
        axis= 0
        )
    return notes

def Feature_Stack(features: List[np.array], padding_value: float, max_length: Optional[int]= None):
    max_feature_length = max_length or max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= padding_value) for feature in features],
        axis= 0
        )
    return features

def Log_F0_Stack(log_f0s, max_length: Optional[int]= None):
    max_log_f0_length = max_length or max([log_f0.shape[0] for log_f0 in log_f0s])
    log_f0s = np.stack(
        [np.pad(log_f0, [0, max_log_f0_length - log_f0.shape[0]], constant_values= -10.0) for log_f0 in log_f0s],
        axis= 0
        )
    return log_f0s

def Energy_Stack(energies, max_length: Optional[int]= None):
    max_energy_length = max_length or max([energy.shape[0] for energy in energies])
    energies = np.stack(
        [np.pad(energy, [0, max_energy_length - energy.shape[0]], constant_values= 0.0) for energy in energies],
        axis= 0
        )
    return energies

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],        
        log_f0_info_dict: Dict[str, Dict[str, float]],
        energy_info_dict: Dict[str, Dict[str, float]],
        singer_info_dict: Dict[str, int],
        genre_info_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        feature_type: str,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super().__init__()
        self.token_dict = token_dict
        self.log_f0_info_dict = log_f0_info_dict
        self.energy_info_dict = energy_info_dict
        self.singer_info_dict = singer_info_dict
        self.genre_info_dict = genre_info_dict
        self.pattern_path = pattern_path
        self.feature_type = feature_type

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))

        self.patterns = []
        max_pattern_by_singer = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Singer_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Singer_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_singer)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns *= accumulated_dataset_epoch

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))

        log_f0 = np.clip(pattern_dict['Log_F0'], -10.0, np.inf)
        indices = np.where(log_f0 != -10.0)
        log_f0[indices] = (log_f0[indices] - self.log_f0_info_dict[pattern_dict['Singer']]['Mean']) / self.log_f0_info_dict[pattern_dict['Singer']]['Std']
        energy = (pattern_dict['Energy'] - self.energy_info_dict[pattern_dict['Singer']]['Mean']) / self.energy_info_dict[pattern_dict['Singer']]['Std']

        singer = self.singer_info_dict[pattern_dict['Singer']]
        genre = self.genre_info_dict[pattern_dict['Genre']]

        return Lyric_to_Token(pattern_dict['Lyric'], self.token_dict), pattern_dict['Note'], log_f0, energy, singer, genre, pattern_dict[self.feature_type]

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],        
        singer_info_dict: Dict[str, int],
        genre_info_dict: Dict[str, int],
        pattern_paths: List[str],
        singers: List[str],
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
        for path, singer, genre in zip(pattern_paths, singers, genres):
            if not singer in self.singer_info_dict.keys():
                logging.warn('The singer \'{}\' is incorrect. The \'{}\' is ignoired.'.format(singer, path))
                continue
            if not genre in self.genre_info_dict.keys():
                logging.warn('The genre \'{}\' is incorrect. The \'{}\' is ignoired.'.format(genre, path))
                continue
            
            music = []
            text = []
            for line in open(path, 'r', encoding= 'utf-8').readlines()[1:]:
                message_time, lyric, note = line.strip().split('\t')
                music.append((float(message_time), lyric, int(note)))
                text.append(lyric)
            lyric, note = Convert_Feature_Based_Music(
                music= music,
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
    def __init__(
        self,
        token_dict: Dict[str, int],
        feature_min: float,
        feature_max: float,
        pattern_length: int
        ):
        self.token_dict = token_dict
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.pattern_length = pattern_length

    def __call__(self, batch):
        tokens, notes, log_f0s, energies, singers, genres, features = zip(*batch)        
        
        offsets = []
        for token in tokens:
            while True:
                offset = np.random.randint(0, len(token) - self.pattern_length)
                if len([x for x in token[offset:offset+self.pattern_length] if x == self.token_dict['<X>']]) < len(token[offset:offset+self.pattern_length]) // 2:
                    break
            offsets.append(offset)
        
        tokens = Token_Stack([
            token[offset:offset+self.pattern_length]
            for token, offset in zip(tokens, offsets)
            ], self.token_dict)
        notes = Note_Stack([
            note[offset:offset+self.pattern_length]
            for note, offset in zip(notes, offsets)
            ])
        log_f0s = Log_F0_Stack([
            log_f0[offset:offset+self.pattern_length]
            for log_f0, offset in zip(log_f0s, offsets)
            ])
        energies = Energy_Stack([
            energie[offset:offset+self.pattern_length]
            for energie, offset in zip(energies, offsets)
            ])
        features = Feature_Stack([
            feature[offset:offset+self.pattern_length]
            for feature, offset in zip(features, offsets)
            ], self.feature_min)
        
        features = (features - self.feature_min) / (self.feature_max - self.feature_min) * 2.0 - 1.0

        tokens = torch.LongTensor(tokens)   # [Batch, Featpure_t]
        notes = torch.LongTensor(notes) # [Batch, Featpure_t]
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_d, Featpure_t]
        log_f0s = torch.FloatTensor(log_f0s)    # [Batch, Featpure_t]
        energies = torch.FloatTensor(energies)  # [Batch, Featpure_t]
        singers = torch.LongTensor(singers)  # [Batch]
        genres = torch.LongTensor(genres)  # [Batch]

        return tokens, notes, features, log_f0s, energies, singers, genres

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        pattern_length: int
        ):
        self.token_dict = token_dict
        self.pattern_length = pattern_length
         
    def __call__(self, batch):
        tokens, notes, singers, genres, lyrics = zip(*batch)
        
        lengths = np.array([len(token) for token in tokens])        
        tokens = Token_Stack(tokens, self.token_dict, self.pattern_length)
        notes = Note_Stack(notes, self.pattern_length)

        # This is temporal because MLP Singer's compatible pattern length is fixed.
        tokens = tokens[:, :self.pattern_length]
        notes = notes[:, :self.pattern_length]
        lengths = np.minimum(lengths, self.pattern_length)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        lengths = torch.LongTensor(lengths)   # [Batch]
        notes = torch.LongTensor(notes)   # [Batch, Time]
        singers = torch.LongTensor(singers)  # [Batch]
        genres = torch.LongTensor(genres)  # [Batch]
        
        lyrics = [''.join([(x if x != '<X>' else ' ') for x in lyric]) for lyric in lyrics]

        return tokens, notes, singers, genres, lengths, lyrics