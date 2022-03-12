import torch
import numpy as np
import logging, yaml, os, sys, argparse, math
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
from librosa import griffinlim
from scipy.io import wavfile

from Modules.Modules import MLPSinger
from Datasets import Inference_Dataset, Inference_Collater
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler
from Logger import Logger

from meldataset import spectral_de_normalize_torch
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
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

class Dataset(Inference_Dataset):
    def __init__(
        self,
        patterns: List[List[Tuple[int, str, int]]],
        token_dict: dict,
        max_duration: int,
        equality_duration: bool= False,
        consonant_duration: int= 3,
        ):
        self.token_dict = token_dict
        self.equality_duration = equality_duration
        self.consonant_duration = consonant_duration

        self.patterns = []
        for index, pattern in enumerate(patterns):
            durations, texts, notes = zip(*pattern)
            if sum(durations) > max_duration:
                logging.warn('The pattern index {} is too long. This pattern will be ignoired.'.format(index))
                continue
            self.patterns.append((durations, texts, notes, ''))

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
            self.vocoder = torch.jit.load('hifigan_jit_sing_0273.pts', map_location='cpu').to(self.device)

        self.Load_Checkpoint(checkpoint_path)
        self.out_path = out_path
        self.batch_size = batch_size

    def Dataset_Generate(self, patterns):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)

        return torch.utils.data.DataLoader(
            dataset= Dataset(
                patterns= patterns,
                token_dict= token_dict,
                max_duration= self.hp.Duration.Max,
                equality_duration= self.hp.Duration.Equality,
                consonant_duration= self.hp.Duration.Consonant_Duration
                ),
            shuffle= False,
            collate_fn= Inference_Collater(
                token_dict= token_dict,
                max_duration= self.hp.Duration.Max
                ),
            batch_size= self.batch_size,
            num_workers= 0,
            pin_memory= True
            )

    @torch.no_grad()
    def Inference_Step(self, tokens, notes, durations, texts, decomposed_texts, export_files= False, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)

        predictions = self.model(
            tokens= tokens,
            notes= notes,
            durations= durations
            )

        audios = []
        for prediction in predictions.transpose(2, 1):
            if self.hp.Feature_Type == 'Mel':
                audio = self.vocoder(prediction.unsqueeze(0)).cpu().numpy()
            elif self.hp.Feature_Type == 'Spectrogram':
                prediction = spectral_de_normalize_torch(prediction).cpu().numpy()
                audio = griffinlim(prediction)
            audios.append(audio)
        audios = [(audio / np.abs(audio).max() * 32767.5).astype(np.int16) for audio in audios]

        if export_files:
            files = []
            for index in range(predictions.size(0)):
                tags = []
                if tag_step: tags.append('Step-{}'.format(self.steps))
                tags.append('IDX_{}'.format(index + start_index))
                files.append('.'.join(tags))

            os.makedirs(os.path.join(self.out_path, 'PNG').replace('\\', '/'), exist_ok= True)
            os.makedirs(os.path.join(self.out_path, 'WAV').replace('\\', '/'), exist_ok= True)
            for index, (prediction, text, decomposed_text, audio, file) in enumerate(zip(
                predictions.cpu().numpy(),
                texts,
                decomposed_texts,
                audios,
                files
                )):
                title = 'Text: {}'.format(text if len(text) < 90 else text[:90] + '…')
                new_Figure = plt.figure(figsize=(20, 5 * 1), dpi=100)
                plt.subplot2grid((1, 1), (0, 0))
                plt.imshow(prediction.T, aspect='auto', origin='lower')
                plt.title('Feature    {}'.format(title))
                plt.colorbar()            
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_path, 'PNG', '{}.png'.format(file)).replace('\\', '/'))
                plt.close(new_Figure)
                
                wavfile.write(
                    os.path.join(self.out_path, 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                    self.hp.Sound.Sample_Rate,
                    audio
                    )

        audios = [audio[:sum(duration[:-1])] for audio, duration in zip(audios, durations.cpu().detach().numpy())]

        return audios
            
    def Inference_Epoch(self, patterns, export_files= False, use_tqdm= True):
        dataloader = self.Dataset_Generate(patterns)
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )

        export_audios = []
        for step, (tokens, notes, durations, texts, decomposed_texts) in enumerate(dataloader):
            audios = self.Inference_Step(tokens, notes, durations, texts, decomposed_texts, export_files= export_files, start_index= step * self.batch_size)
            export_audios.append(audios)

        export_audios = [audio for audios in export_audios for audio in audios]

        return export_audios

    def Load_Checkpoint(self, path):
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])        
        self.steps = state_dict['Steps']

        self.model.eval()

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

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