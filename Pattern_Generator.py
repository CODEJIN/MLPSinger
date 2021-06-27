import numpy as np
import mido, os, pickle, yaml, argparse, math, librosa, hgtk
from tqdm import tqdm
from argparse import Namespace  # for type
import torch

from meldataset import spectrogram
from Arg_Parser import Recursive_Parse

def Pattern_Generate(
    hyper_paramters: Namespace,
    dataset_path: str,
    eval_index: int
    ):
    min_duration, max_duration = math.inf, -math.inf
    min_note, max_note = math.inf, -math.inf

    paths = []
    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            midi_path = wav_path.replace('vox.wav', 'midi.mid')
            paths.append((wav_path, midi_path))

    if eval_index >= len(paths):
        raise ValueError('eval_index is bigger than maximum path index: {} >= {}'.format(eval_index, len(paths)))
    for index, (wav_path, midi_path) in enumerate(paths):
        mid = mido.MidiFile(midi_path, charset='CP949')
        music = []
        current_lyric = ''
        current_note = None
        current_time = 0.0

        # Note on 쉼표
        # From Lyric to message before note on: real note
        for message in list(mid):
            if message.type == 'note_on':
                if message.time < 0.1:
                    current_time += message.time
                    if current_lyric in ['J', 'H', None]:
                        music.append((current_time, '<X>', 0))
                    else:
                        music.append((current_time, current_lyric, current_note))
                else:
                    if not current_lyric in ['J', 'H', None]:
                        music.append((current_time, current_lyric, current_note))
                    else:
                        message.time += current_time
                    music.append((message.time, '<X>', 0))
                current_time = 0.0
                current_lyric = ''
                current_note = None
            elif message.type == 'lyrics':
                current_lyric = message.text.strip()
                current_time += message.time
            elif message.type == 'note_off':
                current_note = message.note
                current_time += message.time
                if current_lyric == 'H':    # it is temp.
                    break
            else:
                current_time += message.time

        if current_lyric in ['J', 'H']:
            if music[-1][1] == '<X>':
                music[-1] = (music[-1][0] + current_time, music[-1][1], music[-1][2])
            else:
                music.append((current_time, '<X>', 0))
        else:
            music.append((current_time, current_lyric, current_note))
        music = music[1:]

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)
        audio = librosa.util.normalize(audio) * 0.95

        if music[0][1] == '<X>':
            audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
            music = music[1:]
        if music[-1][1] == '<X>':
            audio = audio[:-int(music[-1][0] * hyper_paramters.Sound.Sample_Rate)]
            music = music[:-1]

        previous_used = 0
        absolute_position = 0
        mel_based = []
        for x in music:
            duration = int(x[0] * hyper_paramters.Sound.Sample_Rate) + previous_used
            previous_used = duration % hyper_paramters.Sound.Frame_Shift
            duration = duration // hyper_paramters.Sound.Frame_Shift
            mel_based.append((absolute_position, duration, x[1], x[2]))
            absolute_position += duration
        music = mel_based

        spect = spectrogram(
            y= torch.from_numpy(audio).float().unsqueeze(0),
            n_fft= hyper_paramters.Sound.N_FFT,
            hop_size= hyper_paramters.Sound.Frame_Shift,
            win_size= hyper_paramters.Sound.Frame_Length,
            center= False
            ).squeeze(0).T.numpy()

        pattern_index = 0
        for start_index in tqdm(range(0, len(music)), desc= os.path.basename(wav_path)):
            for end_index in range(start_index + 1, len(music), 10):
                music_sample = music[start_index:end_index]
                sample_length = music_sample[-1][0] + music_sample[-1][1] - music_sample[0][0]
                if sample_length < hyper_paramters.Duration.Min:
                    continue
                elif sample_length > hyper_paramters.Duration.Max:
                    break

                audio_sample = audio[music_sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_sample[-1][0] + music_sample[-1][1]) * hyper_paramters.Sound.Frame_Shift]
                spect_sample = spect[music_sample[0][0]:music_sample[-1][0] + music_sample[-1][1]]
                
                _, duration_sample, text_sample, note_sample = zip(*music_sample)

                pattern = {
                    'Audio': audio_sample.astype(np.float32),
                    'Spectrogram': spect_sample.astype(np.float32),
                    'Duration': duration_sample,
                    'Text': text_sample,
                    'Note': note_sample,
                    'Singer': 'Female_0',
                    'Dataset': 'NAMS',
                    }

                pattern_path = os.path.join(
                    hyper_paramters.Train.Train_Pattern.Path if index != eval_index else hyper_paramters.Train.Eval_Pattern.Path,
                    'NAMS',
                    '{:03d}'.format(index),
                    'NAMS.S_{:03d}.P_{:05d}.pickle'.format(index, pattern_index)
                    ).replace('\\', '/')
                os.makedirs(os.path.dirname(pattern_path), exist_ok= True)
                pickle.dump(
                    pattern,
                    open(pattern_path, 'wb'),
                    protocol=4
                    )
                pattern_index += 1

                min_duration, max_duration = min(sample_length, min_duration), max(sample_length, max_duration)
        min_note, max_note = min(list(zip(*music))[3] + (min_note,)), max(list(zip(*music))[3] + (max_note,))
    
    print('Duration range: {} - {}'.format(min_duration, max_duration))
    print('Note range: {} - {}'.format(min_note, max_note))

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
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Spectrogram_Length_Dict': {},
        'Music_Length_Dict': {},
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
                    for key in ('Audio', 'Spectrogram', 'Duration', 'Text', 'Note', 'Singer', 'Dataset')
                    ]):
                    continue
                new_metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_metadata_dict['Spectrogram_Length_Dict'][file] = pattern_dict['Spectrogram'].shape[0]
                new_metadata_dict['Music_Length_Dict'][file] = len(pattern_dict['Duration'])
                new_metadata_dict['File_List'].append(file)
            except Exception as e:
                print('File \'{}\' is not correct pattern file. This file is ignored. Error: {}'.format(file, e))
            files_tqdm.update(1)

    with open(os.path.join(pattern_path, metadata_file.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    print('Metadata generate done.')



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset_path", required= True)
    argParser.add_argument("-hp", "--hyper_paramters", required= True)
    argParser.add_argument("-eval", "--eval_index", required= True, type= int)
    args = argParser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_paramters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    Token_Dict_Generate(hyper_parameters= hp)
    Pattern_Generate(hyper_paramters= hp, dataset_path= args.dataset_path, eval_index= args.eval_index)
    Metadata_Generate(hp, False)
    Metadata_Generate(hp, True)

    # python Pattern_Generator.py -hp Hyper_Parameters.yaml -d "E:/Pattern/Sing/108곡 시작 끝 Note 수정 작업 파일" -eval 107