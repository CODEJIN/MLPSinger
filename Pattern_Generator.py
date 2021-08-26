from unicodedata import decimal
import numpy as np
import mido, os, pickle, yaml, argparse, math, librosa, hgtk
from tqdm import tqdm
from typing import List, Tuple
from argparse import Namespace  # for type
import torch

from meldataset import mel_spectrogram, spectrogram
from Arg_Parser import Recursive_Parse

def CSD(
    hyper_paramters: Namespace,
    dataset_path: str,
    note_step: int= 10
    ):
    min_duration, max_duration = math.inf, -math.inf
    min_note, max_note = math.inf, -math.inf

    paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'wav').replace('\\', '/')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            score_path = wav_path.replace('wav', 'csv')
            lyric_path = wav_path.replace('/wav/', '/lyric/').replace('.wav', '.txt')
            paths.append((wav_path, score_path, lyric_path))

    for index, (wav_path, score_path, lyric_path) in enumerate(paths):
        scores = open(score_path, encoding='utf-8-sig').readlines()[1:]
        lyrics = open(lyric_path, encoding='utf-8-sig').read().strip().replace(' ', '').replace('\n', '')
        assert len(scores) == len(lyrics), 'Different length \'{}\''.format(score_path)

        music = []
        previous_end_time = 0.0
        for score, lyric in zip(scores, lyrics):
            start_time, end_time, note, _, = score.strip().split(',')
            start_time, end_time, note = float(start_time), float(end_time), int(note)

            if start_time != previous_end_time:
                music.append((start_time - previous_end_time, '<X>', 0))
            music.append((end_time - start_time, lyric, note))
            previous_end_time = end_time

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)
        if music[0][1] == '<X>':    # remove initial silence
            audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
            music = music[1:]
        audio = audio[:int(previous_end_time * hyper_paramters.Sound.Sample_Rate)]  # remove last silence
        audio = librosa.util.normalize(audio) * 0.95

        music = Convert_Mel_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            frame_shfit= hyper_paramters.Sound.Frame_Shift
            )

        pattern_min_duration, pattern_max_duration = Pattern_File_Generate(
            music= music,
            audio= audio,
            note_step= note_step,
            music_index= index,
            singer= 'CSD',
            dataset= 'CSD',
            is_eval_music= index == (len(paths) - 1),
            description= os.path.basename(wav_path),
            hyper_paramters= hyper_paramters
            )
        
        min_duration, max_duration = min(pattern_min_duration, min_duration), max(pattern_max_duration, max_duration)
        min_note, max_note = min(list(zip(*music))[3] + (min_note,)), max(list(zip(*music))[3] + (max_note,))


def Convert_Mel_Based_Music(
    music: List[Tuple[float, str, int]],
    sample_rate: int,
    frame_shfit: int
    ):
    previous_used = 0
    absolute_position = 0
    mel_based = []
    for x in music:
        duration = int(x[0] * sample_rate) + previous_used
        previous_used = duration % frame_shfit
        duration = duration // frame_shfit
        mel_based.append((absolute_position, duration, x[1], x[2])) # [start_point, end_point, lyric, note]
        absolute_position += duration

    return mel_based

def Pattern_File_Generate(
    music: List[Tuple[int, int, str, int]], # [start_point, end_point, lyric, note]
    audio: np.array,
    note_step: int,
    singer: str,
    dataset: str,
    music_index: int,
    is_eval_music: bool,
    description: str,
    hyper_paramters: Namespace,
    ):
    min_duration, max_duration = math.inf, -math.inf

    spect = spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= hyper_paramters.Sound.N_FFT,
        hop_size= hyper_paramters.Sound.Frame_Shift,
        win_size= hyper_paramters.Sound.Frame_Length,
        center= False
        ).squeeze(0).T.numpy()
    mel = mel_spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= hyper_paramters.Sound.N_FFT,
        num_mels= hyper_paramters.Sound.Mel_Dim,
        sampling_rate= hyper_paramters.Sound.Sample_Rate,
        hop_size= hyper_paramters.Sound.Frame_Shift,
        win_size= hyper_paramters.Sound.Frame_Length,
        fmin= hyper_paramters.Sound.Mel_F_Min,
        fmax= hyper_paramters.Sound.Mel_F_Max,
        center= False
        ).squeeze(0).T.numpy()

    pattern_index = 0
    for start_index in tqdm(range(0, len(music)), desc= description):
        for end_index in range(start_index + 1, len(music), note_step):
            music_sample = music[start_index:end_index]
            sample_length = music_sample[-1][0] + music_sample[-1][1] - music_sample[0][0]
            if sample_length < hyper_paramters.Duration.Min:
                continue
            elif sample_length > hyper_paramters.Duration.Max:
                break

            audio_sample = audio[music_sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_sample[-1][0] + music_sample[-1][1]) * hyper_paramters.Sound.Frame_Shift]
            spect_sample = spect[music_sample[0][0]:music_sample[-1][0] + music_sample[-1][1]]
            mel_sample = mel[music_sample[0][0]:music_sample[-1][0] + music_sample[-1][1]]
            
            _, duration_sample, text_sample, note_sample = zip(*music_sample)

            pattern = {
                'Audio': audio_sample.astype(np.float32),
                'Spectrogram': spect_sample.astype(np.float32),
                'Mel': mel_sample.astype(np.float32),
                'Duration': duration_sample,
                'Text': text_sample,
                'Note': note_sample,
                'Singer': singer,
                'Dataset': dataset,
                }

            pattern_path = os.path.join(
                hyper_paramters.Train.Train_Pattern.Path if not is_eval_music else hyper_paramters.Train.Eval_Pattern.Path,
                dataset,
                '{:03d}'.format(music_index),
                '{}.S_{:03d}.P_{:05d}.pickle'.format(dataset, music_index, pattern_index)
                ).replace('\\', '/')
            os.makedirs(os.path.dirname(pattern_path), exist_ok= True)
            pickle.dump(
                pattern,
                open(pattern_path, 'wb'),
                protocol= 4
                )
            pattern_index += 1

            min_duration, max_duration = min(sample_length, min_duration), max(sample_length, max_duration)

    return min_duration, max_duration


def Token_Dict_Generate(hyper_parameters: Namespace):
    tokens = \
        list(hgtk.letter.CHO) + \
        list(hgtk.letter.JOONG) + \
        ['{}_'.format(x) for x in hgtk.letter.JONG]
    
    os.makedirs(os.path.dirname(hyper_parameters.Token_Path), exist_ok= True)
    yaml.dump(
        {token: index for index, token in enumerate(['<S>', '<E>', '<X>'] + sorted(tokens))},
        open(hyper_parameters.Token_Path, 'w')
        )

def Metadata_Generate(
    hyper_parameters: Namespace,
    eval: bool= False
    ):
    pattern_path = hyper_parameters.Train.Eval_Pattern.Path if eval else hyper_parameters.Train.Train_Pattern.Path
    metadata_file = hyper_parameters.Train.Eval_Pattern.Metadata_File if eval else hyper_parameters.Train.Train_Pattern.Metadata_File

    new_metadata_dict = {
        'N_FFT': hyper_parameters.Sound.N_FFT,
        'Mel_Dim': hyper_parameters.Sound.Mel_Dim,
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Spect_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Music_Length_Dict': {},
        'Singer_Dict': {},
        'File_List_by_Singer_Dict': {},
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)
            file = os.path.join(root, file).replace("\\", "/").replace(pattern_path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Audio', 'Spectrogram', 'Mel', 'Duration', 'Text', 'Note', 'Singer', 'Dataset')
                    ]):
                    continue
                new_metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_metadata_dict['Spect_Length_Dict'][file] = pattern_dict['Spectrogram'].shape[0]
                new_metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[0]
                new_metadata_dict['Music_Length_Dict'][file] = len(pattern_dict['Duration'])
                new_metadata_dict['Singer_Dict'][file] = pattern_dict['Singer']
                new_metadata_dict['File_List'].append(file)
                if not pattern_dict['Singer'] in new_metadata_dict['File_List_by_Singer_Dict'].keys():
                    new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']] = []
                new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']].append(file)
            except Exception as e:
                print('File \'{}\' is not correct pattern file. This file is ignored. Error: {}'.format(file, e))
            files_tqdm.update(1)

    with open(os.path.join(pattern_path, metadata_file.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            {singer: index for index, singer in enumerate(sorted(set(new_metadata_dict['Singer_Dict'].values())))},
            open(hyper_parameters.Singer_Info_Path, 'w')
            )

    print('Metadata generate done.')


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-csd", "--csd_path", required= True)
    argParser.add_argument("-step", "--note_step", default= 10, type= int)
    argParser.add_argument("-hp", "--hyper_paramters", required= True)
    args = argParser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_paramters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    Token_Dict_Generate(hyper_parameters= hp)
    CSD(
        hyper_paramters= hp,
        dataset_path= args.csd_path,
        note_step= args.note_step
        )
    Metadata_Generate(hp, False)
    Metadata_Generate(hp, True)

# python Pattern_Generator.py -hp Hyper_Parameters.yaml -csd "E:/Pattern/Sing/CSD/korean"