import numpy as np
import mido, os, pickle, yaml, argparse, math, librosa, hgtk
from tqdm import tqdm
from pysptk.sptk import rapt
from typing import List, Tuple
from argparse import Namespace  # for type
import torch

from meldataset import mel_spectrogram, spectrogram, spec_energy
from Arg_Parser import Recursive_Parse


def Mediazen(
    hyper_paramters: Namespace,
    dataset_path: str,
    singer: str,
    dataset: str
    ):
    genre_dict = {
        line.strip().split('\t')[0]: line.strip().split('\t')[2]
        for line in open(os.path.join(dataset_path, 'genre.txt').replace('\\', '/'), 'r', encoding='utf-8-sig').readlines()[1:]
        }

    paths = []
    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            midi_path = wav_path.replace('vox', 'midi').replace('.wav', '.mid')

            if not os.path.exists(midi_path):
                raise FileExistsError(midi_path)

            paths.append((wav_path, midi_path))

    for index, (wav_path, midi_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= f'{dataset}|{singer}'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            dataset,
            singer,
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path):
            continue

        genre = genre_dict[os.path.splitext(os.path.basename(wav_path))[0]]

        mid = mido.MidiFile(midi_path, charset='CP949')
        music = []
        current_lyric = ''
        current_note = None
        current_time = 0.0

        # Note on 쉼표
        # From Lyric to message before note on: real note
        for message in list(mid):
            if message.type == 'note_on' and message.velocity != 0:
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
                if message.text == '\r':    # mzm 02678.mid
                    continue
                current_lyric = message.text.strip()
                current_time += message.time
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
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
        if music[0][1] == '<X>':
            audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
            music = music[1:]
        if music[-1][1] == '<X>':
            audio = audio[:-int(music[-1][0] * hyper_paramters.Sound.Sample_Rate)]
            music = music[:-1]
        audio = librosa.util.normalize(audio) * 0.95

        lyrics, notes = Convert_Feature_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            frame_shift= hyper_paramters.Sound.Frame_Shift
            )

        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= genre,
            dataset= dataset,
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters
            )

def CSD(
    hyper_paramters: Namespace,
    dataset_path: str
    ):
    paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'wav').replace('\\', '/')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            score_path = wav_path.replace('wav', 'csv')
            lyric_path = wav_path.replace('/wav/', '/lyric/').replace('.wav', '.txt')
            
            if not os.path.exists(score_path):
                raise FileExistsError(score_path)
            elif not os.path.exists(lyric_path):
                raise FileExistsError(lyric_path)

            paths.append((wav_path, score_path, lyric_path))

    for index, (wav_path, score_path, lyric_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'CSD'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            'CSD',
            'CSD',
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path):
            continue
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

        lyrics, notes = Convert_Feature_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            frame_shift= hyper_paramters.Sound.Frame_Shift
            )

        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            audio= audio,
            music_label= os.path.splitext(os.path.basename(wav_path))[0],
            singer= 'CSD',
            genre= 'Children',
            dataset= 'CSD',
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters
            )

def Convert_Feature_Based_Music(
    music: List[Tuple[float, str, int]],
    sample_rate: int,
    frame_shift: int,
    consonant_duration: int= 3,
    equality_duration: bool= False
    ):
    previous_used = 0
    lyrics = []
    notes = []
    durations = []
    for message_time, lyric, note in music:
        duration = round(message_time * sample_rate) + previous_used
        previous_used = duration % frame_shift
        duration = duration // frame_shift

        if lyric == '<X>':
            lyrics.append(lyric)
            notes.append(note)
            durations.append(duration)
        else:
            lyrics.extend(Decompose(lyric))
            notes.extend([note] * 3)
            if equality_duration or duration < consonant_duration * 3:
                split_duration = [duration // 3] * 3
                split_duration[1] += duration % 3
                durations.extend(split_duration)
            else:
                durations.extend([
                    consonant_duration,    # onset
                    duration - consonant_duration * 2, # nucleus
                    consonant_duration # coda
                    ])

    lyrics = sum([[lyric] * duration for lyric, duration in zip(lyrics, durations)], [])
    notes = sum([*[[note] * duration for note, duration in zip(notes, durations)]], [])

    return lyrics, notes

def Decompose(syllable: str):
    onset, nucleus, coda = hgtk.letter.decompose(syllable)
    coda += '_'

    return onset, nucleus, coda

def Pattern_File_Generate(
    lyric: List[str],
    note: List[int],
    audio: np.array,
    singer: str,
    genre: str,
    dataset: str,
    music_label: str,
    is_eval_music: bool,
    hyper_paramters: Namespace,
    ):
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

    log_f0 = rapt(
        x= audio * 32768,
        fs= hyper_paramters.Sound.Sample_Rate,
        hopsize= hyper_paramters.Sound.Frame_Shift,
        min= hyper_paramters.Sound.F0_Min,
        max= hyper_paramters.Sound.F0_Max,
        otype= 2,   # log
        )[:mel.shape[0]]

    energy = spec_energy(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= hyper_paramters.Sound.N_FFT,
        hop_size= hyper_paramters.Sound.Frame_Shift,
        win_size=hyper_paramters.Sound.Frame_Length,
        center= False
        ).squeeze(0).numpy()


    if mel.shape[0] > len(lyric):
        spect = spect[math.floor((spect.shape[0] - len(lyric)) / 2.0):-math.ceil((spect.shape[0] - len(lyric)) / 2.0)]
        mel = mel[math.floor((mel.shape[0] - len(lyric)) / 2.0):-math.ceil((mel.shape[0] - len(lyric)) / 2.0)]
        log_f0 = log_f0[math.floor((log_f0.shape[0] - len(lyric)) / 2.0):-math.ceil((log_f0.shape[0] - len(lyric)) / 2.0)]
        energy = energy[math.floor((energy.shape[0] - len(lyric)) / 2.0):-math.ceil((energy.shape[0] - len(lyric)) / 2.0)]
    elif len(lyric) > mel.shape[0]:
        lyric = lyric[math.floor((len(lyric) - mel.shape[0]) / 2.0):-math.ceil((len(lyric) - mel.shape[0]) / 2.0)]
        note = note[math.floor((len(note) - mel.shape[0]) / 2.0):-math.ceil((len(note) - mel.shape[0]) / 2.0)]
        
    pattern = {
        'Audio': audio.astype(np.float32),
        'Spectrogram': spect.astype(np.float32),
        'Mel': mel.astype(np.float32),
        'Log_F0': log_f0.astype(np.float32),
        'Energy': energy.astype(np.float32),
        'Lyric': lyric,
        'Note': note,
        'Singer': singer,
        'Genre': genre,
        'Dataset': dataset,
        }

    pattern_path = os.path.join(
        hyper_paramters.Train.Train_Pattern.Path if not is_eval_music else hyper_paramters.Train.Eval_Pattern.Path,
        dataset,
        singer,
        f'{music_label}.pickle'
        ).replace('\\', '/')

    os.makedirs(os.path.dirname(pattern_path), exist_ok= True)
    pickle.dump(
        pattern,
        open(pattern_path, 'wb'),
        protocol= 4
        )


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

    spectrogram_range_dict = {}
    mel_range_dict = {}
    log_f0_dict = {}
    energy_dict = {}
    singers = []
    genres = []
    min_note, max_note = math.inf, -math.inf

    new_metadata_dict = {
        'N_FFT': hyper_parameters.Sound.N_FFT,
        'Mel_Dim': hyper_parameters.Sound.Mel_Dim,
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Feature_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Log_F0_Length_Dict': {},
        'Energy_Length_Dict': {},
        'Lyric_Length_Dict': {},
        'Note_Length_Dict': {},
        'Singer_Dict': {},
        'Genre_Dict': {},
        'File_List_by_Singer_Dict': {},
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path, followlinks= True)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path, followlinks= True):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)
            file = os.path.join(root, file).replace("\\", "/").replace(pattern_path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Audio', 'Spectrogram', 'Mel', 'Log_F0', 'Energy', 'Lyric', 'Note', 'Singer', 'Genre', 'Dataset')
                    ]):
                    continue
                new_metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_metadata_dict['Feature_Length_Dict'][file] = pattern_dict['Spectrogram'].shape[0]
                new_metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[0]
                new_metadata_dict['Log_F0_Length_Dict'][file] = pattern_dict['Log_F0'].shape[0]
                new_metadata_dict['Energy_Length_Dict'][file] = pattern_dict['Energy'].shape[0]
                new_metadata_dict['Lyric_Length_Dict'][file] = len(pattern_dict['Lyric'])
                new_metadata_dict['Note_Length_Dict'][file] = len(pattern_dict['Note'])
                new_metadata_dict['Singer_Dict'][file] = pattern_dict['Singer']
                new_metadata_dict['File_List'].append(file)
                if not pattern_dict['Singer'] in new_metadata_dict['File_List_by_Singer_Dict'].keys():
                    new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']] = []
                new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']].append(file)

                if not pattern_dict['Singer'] in spectrogram_range_dict.keys():
                    spectrogram_range_dict[pattern_dict['Singer']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Singer'] in mel_range_dict.keys():
                    mel_range_dict[pattern_dict['Singer']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Singer'] in log_f0_dict.keys():
                    log_f0_dict[pattern_dict['Singer']] = []
                if not pattern_dict['Singer'] in energy_dict.keys():
                    energy_dict[pattern_dict['Singer']] = []
                
                spectrogram_range_dict[pattern_dict['Singer']]['Min'] = min(spectrogram_range_dict[pattern_dict['Singer']]['Min'], pattern_dict['Spectrogram'].min().item())
                spectrogram_range_dict[pattern_dict['Singer']]['Max'] = max(spectrogram_range_dict[pattern_dict['Singer']]['Max'], pattern_dict['Spectrogram'].max().item())
                mel_range_dict[pattern_dict['Singer']]['Min'] = min(mel_range_dict[pattern_dict['Singer']]['Min'], pattern_dict['Spectrogram'].min().item())
                mel_range_dict[pattern_dict['Singer']]['Max'] = max(mel_range_dict[pattern_dict['Singer']]['Max'], pattern_dict['Spectrogram'].max().item())
                
                log_f0_dict[pattern_dict['Singer']].append(pattern_dict['Log_F0'])
                energy_dict[pattern_dict['Singer']].append(pattern_dict['Energy'])
                singers.append(pattern_dict['Singer'])
                genres.append(pattern_dict['Genre'])

                min_note = min(min_note, *[x for x in pattern_dict['Note'] if x > 0])
                max_note = max(max_note, *[x for x in pattern_dict['Note'] if x > 0])
            except Exception as e:
                print('File \'{}\' is not correct pattern file. This file is ignored. Error: {}'.format(file, e))
            files_tqdm.update(1)

    new_metadata_dict['Min_Note'] = min_note
    new_metadata_dict['Max_Note'] = max_note

    with open(os.path.join(pattern_path, metadata_file.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            spectrogram_range_dict,
            open(hp.Spectrogram_Range_Info_Path, 'w')
            )
        yaml.dump(
            mel_range_dict,
            open(hp.Mel_Range_Info_Path, 'w')
            )
        
        log_f0_info_dict = {}
        for singer, log_f0_list in log_f0_dict.items():
            log_f0 = np.hstack(log_f0_list)
            log_f0 = np.clip(log_f0, -10.0, np.inf)
            log_f0 = log_f0[log_f0 != -10.0]

            log_f0_info_dict[singer] = {
                'Mean': log_f0.mean().item(),
                'Std': log_f0.std().item(),
                }
        yaml.dump(
            log_f0_info_dict,
            open(hp.Log_F0_Info_Path, 'w')
            )

        energy_info_dict = {}
        for singer, energy_list in energy_dict.items():
            energy = np.hstack(energy_list)            
            energy_info_dict[singer] = {
                'Mean': energy.mean().item(),
                'Std': energy.std().item(),
                }
        yaml.dump(
            energy_info_dict,
            open(hp.Energy_Info_Path, 'w')
            )

        singer_index_dict = {
            singer: index
            for index, singer in enumerate(sorted(set(singers)))
            }
        yaml.dump(
            singer_index_dict,
            open(hyper_parameters.Singer_Info_Path, 'w')
            )

        genre_index_dict = {
            genre: index
            for index, genre in enumerate(sorted(set(genres)))
            }
        yaml.dump(
            genre_index_dict,
            open(hyper_parameters.Genre_Info_Path, 'w')
            )

    print('Metadata generate done.')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-nams", "--nams_path", required= False)
    argparser.add_argument("-mzm", "--mediazen_male_path", required= False)
    argparser.add_argument("-mzf", "--mediazen_female_path", required= False)
    argparser.add_argument("-kje", "--kje_path", required= False)
    argparser.add_argument("-csd", "--csd_path", required= False)
    argparser.add_argument("-hp", "--hyper_paramters", required= True)
    args = argparser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_paramters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    Token_Dict_Generate(hyper_parameters= hp)
    if args.nams_path:
        Mediazen(
            hyper_paramters= hp,
            dataset_path= args.nams_path,
            singer= 'NAMS',
            dataset= 'NAMS'
            )
    if args.mediazen_male_path:
        Mediazen(
            hyper_paramters= hp,
            dataset_path= args.mediazen_male_path,
            singer= 'Mediazen_Male',
            dataset= 'Mediazen'
            )
    if args.mediazen_female_path:
        Mediazen(
            hyper_paramters= hp,
            dataset_path= args.mediazen_female_path,
            singer= 'Mediazen_Female',
            dataset= 'Mediazen'
            )
    if args.kje_path:
        Mediazen(
            hyper_paramters= hp,
            dataset_path= args.kje_path,
            singer= 'KJE',
            dataset= 'Mediazen'
            )
    if args.csd_path:
        CSD(
            hyper_paramters= hp,
            dataset_path= args.csd_path
            )
    Metadata_Generate(hp, False)
    Metadata_Generate(hp, True)

# python Pattern_Generator.py -hp Hyper_Parameters.yaml -mzf E:/Pattern/Sing/Mediazen/mzf -mzm E:/Pattern/Sing/Mediazen/mzm -kje E:/Pattern/Sing/Mediazen/KJE -nams E:/Pattern/Sing/Nams -csd E:/Pattern/Sing/CSD/korean