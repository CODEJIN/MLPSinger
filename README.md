# MLP Singer

This code is an implementation for Singing TTS. The algorithm is based on the following papers:

```
Tae, J., Kim, H., & Lee, Y. (2021). MLP Singer: Towards Rapid Parallel Korean Singing Voice Synthesis. arXiv preprint arXiv:2106.07886.
Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., ... & Dosovitskiy, A. (2021). Mlp-mixer: An all-mlp architecture for vision. arXiv preprint arXiv:2105.01601.
```

# Structure
* Structure is based on the MLP Singer.
* I changed several hyper parameters and data type
    * Feature type is changed from mel to spectrogram.
    * Token type is changed from phoneme to grapheme.


# Used dataset
* Code verification was conducted through a private Korean dataset.
    * Thus, current [Pattern_Generator.py](Pattern_Generator.py) and [Datasets.py](Datasets.py) are based on the Korean.
* TODO: Scripting for the offical dataset.
    * [CSD Dataset](https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD)


# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in [Hyper_Parameters.yaml](Hyper_Parameters.yaml) according to your environment.

* Sound
    * Setting basic sound parameters.

* Tokens
    * The number of Lyric token.

* Max_Note
    * The highest note value for embedding.

* Duration
    * Min duration is used at pattern generating only.
    * Max duration is decided the maximum time step of model.
        MLP mixer always use the maximum time step.
    * Equality set the strategy about syllable to grapheme.
        * When `True`, onset, nucleus, and coda have same length or ±1 difference.
        * When `False`, onset and coda have Consonant_Duration length, and nucleus has duration - 2 * Consonant_Duration.
    
* Encoder
    * Setting the encoder(embedding).

* Mixer
    * Setting the MLP mixer.

* Train
    * Setting the parameters of training.

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
    * If this is `True`, device parameter is also multiple like '0,1,2,3'.
    * And training command is also changed: please check  'multi_gpu.sh'

* Device
    * Setting which GPU devices are used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'. (But, I don't recommend while training.)

# Generate pattern

* Current version does not support any open source dataset.
    
# Inference file path while training for verification.

* Inference_for_Training
    * There are three examples for inference.
    * It is midi file based script.

# Run

## Command

### Single GPU
```
python Train.py -hp <path> -s <int>
```

* `-hp <path>`
    * The hyper paramter file path
    * This is required.

* `-s <int>`
    * The resume step parameter.
    * Default is `0`.
    * If value is `0`, model try to search the latest checkpoint.

### Multi GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=32 python -m torch.distributed.launch --nproc_per_node=8 Train.py --hyper_parameters Hyper_Parameters.yaml --port 54322
```

* I recommend to check the `multi_gpu.sh`.