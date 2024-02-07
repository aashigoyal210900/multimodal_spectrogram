# A multimodal architecture with shared encoder that uses spectrograms for audio
## Course Project, CSCI 535, Spring 2024
Contributors - Sai Anuroop Kesanapalli, Riya Ranjan, Aashi Goyal, Wilson Tan

#### Usage
> :warning: To be continuously modified as we make progress!
* Install ```ffmpeg```<br>
  ```$ brew install ffmpeg``` (macOS)<br>
  ```$ sudo apt-get install ffmpeg``` (Linux)<br>
  ```$ python3 -m pip install ffmpeg``` (conda)

* Run ```flv_to_wav.py``` to convert FLV video files of CREMA-D to WAV audio files<br>
  ```$ python3 flv_to_wav.py input_folder output_folder```

* Install ```librosa```<br>
  ```$ python3 -m pip install librosa```
  
* Run ```wav_to_melspec.py``` to convert WAV audio files to Mel spectrograms<br>
  ```$ python3 wav_to_melspec.py input_folder output_folder```

* Install ```torch```, ```torchvision```, ```pillow```<br>
  ```$ python3 -m pip install torch torchvision pillow```
  > TODO: Bundle all requirements for the pipeline into a single ```requirements.txt```
  
* Run ```melspec_to_features_cnn.py``` to extract features out of Mel spectrograms using pre-trained ResNet-18<br>
  ```$ python3 -m melspec_to_features_cnn.py input_folder```
  > TODO: Explore the features extracted using pre-trained ResNet-18, think about training ResNet-18 on the Mel spectrograms / corresponding video files / both

#### Resources
Audio feature extraction via spectrograms - https://github.com/DeepSpectrum/DeepSpectrum
