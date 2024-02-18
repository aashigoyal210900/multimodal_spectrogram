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
  
* Run ```melspec_to_features_cnn.py``` to extract features out of Mel spectrograms using ResNet-18 (fine-tuned on Mel spectrograms) <br>
  ```$ python3 -m melspec_to_features_cnn.py input_folder```
  <!-- > TODO: Explore the features extracted using pre-trained ResNet-18, think about training ResNet-18 on the Mel spectrograms / corresponding video files / both -->

#### Findings
* Experiment: Fine-tune ResNet18 on Mel spectrograms generated from a subset of CREMA-D
    * Model: models.resnet18(weights='DEFAULT')
    * Number of classes: ```3``` (ANG, SAD, HAP)
    * Model fine-tuned on: Mel spectrograms
    * Total number of samples: ```273```
    * Number of train samples: ```191```
    * Number of test samples: ```82```
    * Batch size: ```32```
    * lr: ```0.001```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```0.5624``` 
        * Train Accuracy: ```0.9895```
        * Test Loss: ```0.6576```
        * Test Accuracy: ```0.8902```
    * Reproduce:
        * Notebook: [```notebooks/melspec_to_features_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/melspec_to_features_cnn.ipynb)
        * Script: [```scripts/melspec_to_features_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/melspec_to_features_cnn.py)
        * Saved Model: [ResNet18_melspec_50_32_0.001](https://drive.google.com/file/d/1HXjd7Ej0L4NJLfzxH0L8taDTXRGoGBML/view?usp=drive_link)

#### Resources
<!-- Audio feature extraction via spectrograms - https://github.com/DeepSpectrum/DeepSpectrum <br> -->
[GDrive](https://drive.google.com/drive/folders/1BhpgUDgbYwoTaTO6Yo8M3uR0Clw0bkiC?usp=drive_link) <br>
[GDoc](https://docs.google.com/document/d/1jN6ZpCUjqboJQLSFR-Osqlm5kRHGYX2a47GRADVYUPU/edit?usp=sharing)
