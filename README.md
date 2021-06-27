# TacoSinger

This code is an implementation for Singing TTS. The algorithm is based on the following papers:

```
Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., ... & Wu, Y. (2018, April). Natural tts synthesis by conditioning wavenet on mel spectrogram predictions. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 4779-4783). IEEE.
```

# Requirements
Please see the 'requirements.txt'.


# Structure
Structure is based on the Tacotron2.

# Used dataset
* Code verification was conducted through a private Korean dataset.
    * Thus, current [Pattern_Generator.py](Pattern_Generator.py) and [Datasets.py](Datasets.py) are based on the Korean.
* Please report the information about any available open source dataset.
    * The set of midi files with syncronized lyric and high resolution vocal wave files


# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in 'Hyper_Parameters.yaml' according to your environment.

* Sound
    * Setting basic sound parameters.

* Tokens
    * The number of Lyric token.

* Max_Note
    * The highest note value for embedding.

* Min/Max duration
    * Min/Min duration is used at pattern generating only.

* Encoder
    * Setting the encoder.

* Decoder
    * Setting for decoder.

* Postnet
    * Setting for postnet.

* Vocoder_Path
    * Setting the traced vocoder path.
    * To generate this, please check [Here](https://github.com/CODEJIN/PWGAN_for_HiFiSinger)

* Train
    * Setting the parameters of training.

* Use_Mixed_Precision
    * Setting mix precision usage.
    * Need a [Nvidia-Apex](https://github.com/NVIDIA/apex).

* Inference_Batch_Size
    * Setting the batch size when inference

* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Use_Mixed_Precision
    * Setting using mixed precision
* Use_Multi_GPU
    * Setting using multi gpu
    * By the nvcc problem, Only linux supports this option.
    * If this is `true`, device parameter is also multiple like '0,1,2,3'.
    * And training command is also changed: please check  'multi_gpu.sh'
* Device
    * Setting which GPU device is used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'. (But, I don't recommend while training.)

# Generate pattern

* There is no available open source dataset.
    
# Inference file path while training for verification.

* Inference_for_Training
    * There are two examples for inference.
    * It is midi file based script.

# Run

## Command
```
python Train.py -s <int>
```

* `-hp <path>`
    * The hyper paramter file path
    * This is required.

* `-s <int>`
    * The resume step parameter.
    * Default is `0`.
    * If value is `0`, model try to search the latest checkpoint.