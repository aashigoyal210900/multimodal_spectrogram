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

* Install ```torch```, ```torchvision```, ```pillow```, ```numpy```, ```scikit-learn```, ```opencv-python```, ```transformers```<br>
  ```$ python3 -m pip install torch torchvision pillow```
  > TODO: Bundle all requirements for the pipeline into a single ```requirements.txt```

* Run ```melspec_to_features_cnn.py``` to extract features out of Mel spectrograms using ResNet-18 (fine-tuned on Mel spectrograms) <br>
  ```$ python3 -m melspec_to_features_cnn.py input_folder```
  <!-- > TODO: Explore the features extracted using pre-trained ResNet-18, think about training ResNet-18 on the Mel spectrograms / corresponding video files / both -->

#### For 3D experiments

* Start with creating mel spectrograms by running:

  ```python3 wav_to_melspec_3d.py input_folder output_folder```

* Then create 3D Data:

  ```python3 create_3d_data.py video_folder spectrogram_folder output_folder```

* Now train your model with any of the following where ```modality``` can be ```[audio, vision, multi]```. ```pretrain_checkpoint``` is optional. ImageNet pretrained model is provided in the ```models``` folder named ```rgb_imagenet.pt```. If ```pretrain_checkpoint``` is missing, the untrained I3D model will be used:

  ```python3 simple3d_train_test.py modality 3d_data_path output_path```

  ```python3 i3d_train_test.py modality 3d_data_path output_path pretrain_checkpoint```

  ```python3 videoMAE_train_test.py modality 3d_data_path output_path```

* For ablated tests. ```Checkpoint_path``` is optional but recommended, otherwise an untrained model is used:

  ```python3 simple3d_ablated_test.py modality 3d_data_path checkpoint_path```

  ```python3 i3d_ablated_test.py modality 3d_data_path checkpoint_path```

#### Findings

* Experiment: Check performance of pre-trained ResNet18 on Mel spectrograms generated from a subset of CREMA-D spectrograms
    * Model: models.resnet18(weights='DEFAULT')
    * Number of classes: ```3``` (ANG, SAD, HAP)
    <!-- * Model fine-tuned on: averaged one-second granular frames -->
    * Total number of samples: ```273```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```82```
    * Batch size: ```32```
    <!-- * lr: ```0.001``` -->
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```0.5809```  -->
        <!-- * Train Accuracy: ```0.9686``` -->
        * Test Loss: ```1.0953```
        * Test Accuracy: ```0.3415```
    * Reproduce:
        * Notebook: [```notebooks/pretrained/melspec_to_features_pretrained_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/pretrained/melspec_to_features_pretrained_cnn.ipynb)
        * Script: [```scripts/pretrained/melspec_to_features_pretrained_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/pretrained/melspec_to_features_pretrained_cnn.py)
        <!-- * Saved Model: [ResNet18_video_50_32_0.001](https://drive.google.com/file/d/1aZ4IMVIlKW8Qq-EvaVwd-7YKSm8obUXa/view?usp=drive_link) -->

* Experiment: Check performance of pre-trained ResNet18 on faces cropped out of middle frames extracted out of videos from a subset of CREMA-D
    * Model: models.resnet18(weights='DEFAULT')
    * Number of classes: ```3``` (ANG, SAD, HAP)
    <!-- * Model fine-tuned on: averaged one-second granular frames -->
    * Total number of samples: ```273```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```82```
    * Batch size: ```32```
    <!-- * lr: ```0.001``` -->
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```0.5809```  -->
        <!-- * Train Accuracy: ```0.9686``` -->
        * Test Loss: ```1.0967```
        * Test Accuracy: ```0.3659```
    * Reproduce:
        * Notebook: [```notebooks/pretrained/video_to_features_pretrained_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/pretrained/video_to_features_pretrained_cnn.ipynb)
        * Script: [```scripts/pretrained/video_to_features_pretrained_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/pretrained/video_to_features_pretrained_cnn.py)
        <!-- * Saved Model: [ResNet18_video_50_32_0.001](https://drive.google.com/file/d/1aZ4IMVIlKW8Qq-EvaVwd-7YKSm8obUXa/view?usp=drive_link) -->

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
        * Notebook: [```notebooks/finetuned-individual/melspec_to_features_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/finetuned-individual/melspec_to_features_cnn.ipynb)
        * Script: [```scripts/finetuned-individual/melspec_to_features_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/finetuned-individual/melspec_to_features_cnn.py)
        * Saved Model: [ResNet18_melspec_50_32_0.001](https://drive.google.com/file/d/1HXjd7Ej0L4NJLfzxH0L8taDTXRGoGBML/view?usp=drive_link)

* Experiment: Fine-tune ResNet18 on faces cropped out of middle frames extracted out of videos from a subset of CREMA-D
    * Model: models.resnet18(weights='DEFAULT')
    * Number of classes: ```3``` (ANG, SAD, HAP)
    * Model fine-tuned on: faces cropped out of middle frames
    * Total number of samples: ```273```
    * Number of train samples: ```191```
    * Number of test samples: ```82```
    * Batch size: ```32```
    * lr: ```0.001```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```0.5809```
        * Train Accuracy: ```0.9686```
        * Test Loss: ```0.7736```
        * Test Accuracy: ```0.7805```
    * Reproduce:
        * Notebook: [```notebooks/finetuned-individual/video_to_features_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/finetuned-individual/video_to_features_cnn.ipynb)
        * Script: [```scripts/finetuned-individual/video_to_features_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/finetuned-individual/video_to_features_cnn.py)
        * Saved Model: [ResNet18_video_50_32_0.001](https://drive.google.com/file/d/1aZ4IMVIlKW8Qq-EvaVwd-7YKSm8obUXa/view?usp=drive_link)

* Experiment: Check cross performance of ResNet18 finetuned with Mel spectrograms on videos
    * Model: model.load_state_dict(torch.load('/content/drive/MyDrive/csci535/models/ResNet18_melspec_50_32_0.001'))
    * Number of classes: ```3``` (ANG, SAD, HAP)
    <!-- * Model fine-tuned on: averaged one-second granular frames -->
    * Total number of samples: ```273```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```82```
    * Batch size: ```32```
    <!-- * lr: ```0.001``` -->
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```0.5809```  -->
        <!-- * Train Accuracy: ```0.9686``` -->
        * Test Loss: ```1.2611```
        * Test Accuracy: ```0.2561```
    * Reproduce:
        * Notebook: [```notebooks/finetuned-cross/video_to_features_audio_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/finetuned-cross/video_to_features_audio_cnn.ipynb)
        * Script: [```scripts/finetuned-cross/video_to_features_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/finetuned-cross/video_to_features_audio_cnn.py)
        <!-- * Saved Model: [ResNet18_video_50_32_0.001](https://drive.google.com/file/d/1aZ4IMVIlKW8Qq-EvaVwd-7YKSm8obUXa/view?usp=drive_link) -->

* Experiment: Check cross performance of ResNet18 finetuned with faces on Mel spectrograms
    * Model: model.load_state_dict(torch.load('/content/drive/MyDrive/csci535/models/ResNet18_video_50_32_0.001'))
    * Number of classes: ```3``` (ANG, SAD, HAP)
    <!-- * Model fine-tuned on: averaged one-second granular frames -->
    * Total number of samples: ```273```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```82```
    * Batch size: ```32```
    <!-- * lr: ```0.001``` -->
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```0.5809```  -->
        <!-- * Train Accuracy: ```0.9686``` -->
        * Test Loss: ```1.2287```
        * Test Accuracy: ```0.3171```
    * Reproduce:
        * Notebook: [```notebooks/finetuned-cross/melspec_to_features_video_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/finetuned-cross/melspec_to_features_video_cnn.ipynb)
        * Script: [```scripts/finetuned-cross/melspec_to_features_video_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/finetuned-cross/melspec_to_features_video_cnn.py)
        <!-- * Saved Model: [ResNet18_video_50_32_0.001](https://drive.google.com/file/d/1aZ4IMVIlKW8Qq-EvaVwd-7YKSm8obUXa/view?usp=drive_link) -->

* Experiment: Check performance of pre-trained ResNet18 on faces cropped from middle frames extracted out of videos and Mel spectrograms concatenated, from a subset of CREMA-D
    * Model: models.resnet18(weights='DEFAULT')
    * Number of classes: ```3``` (ANG, SAD, HAP)
    <!-- * Model fine-tuned on: faces cropped out of middle frames and Mel spectrograms concatenated -->
    * Total number of samples: ```273```
    * Number of train samples: ```191```
    * Number of test samples: ```82```
    * Batch size: ```32```
    <!-- * lr: ```0.001``` -->
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```0.5515```  -->
        <!-- * Train Accuracy: ```1.0000``` -->
        * Test Loss: ```1.0839```
        * Test Accuracy: ```0.3780```
    * Reproduce:
        * Notebook: [```notebooks/pretrained/audio_video_pretrained_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/pretrained/audio_video_pretrained_cnn.ipynb)
        * Script: [```scripts/pretrained/audio_video_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/pretrained/audio_video_pretrained_cnn.py)
        <!-- * Saved Model: [ResNet18_audio_video_50_32_0.001](https://drive.google.com/file/d/1kjqmT-UssMUMGVL8dcJymzE_iCCqKtQ2/view?usp=drive_link) -->

* Experiment: Fine-tune ResNet18 on faces cropped from middle frames extracted out of videos and Mel spectrograms concatenated, from a subset of CREMA-D
    * Model: models.resnet18(weights='DEFAULT')
    * Number of classes: ```3``` (ANG, SAD, HAP)
    * Model fine-tuned on: faces cropped out of middle frames and Mel spectrograms concatenated
    * Total number of samples: ```273```
    * Number of train samples: ```191```
    * Number of test samples: ```82```
    * Batch size: ```32```
    * lr: ```0.001```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```0.5515```
        * Train Accuracy: ```1.0000```
        * Test Loss: ```0.7124```
        * Test Accuracy: ```0.8415```
    * Reproduce:
        * Notebook: [```notebooks/finetuned-combined/audio_video_cnn.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/finetuned-combined/audio_video_cnn.ipynb)
        * Script: [```scripts/finetuned-combined/audio_video_cnn.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/finetuned-combined/audio_video_cnn.py)
        * Saved Model: [ResNet18_audio_video_50_32_0.001](https://drive.google.com/file/d/1kjqmT-UssMUMGVL8dcJymzE_iCCqKtQ2/view?usp=drive_link)

---

* Experiment: Train Vision Transformer on Mel spectrograms generated from a subset of CREMA-D
    * Model: ViT
    * Number of classes: ```3``` (ANG, SAD, HAP)
    * Model trained on: Mel spectrograms
    * Total number of samples: ```273```
    * Number of train samples: ```191```
    * Number of test samples: ```82```
    * Batch size: ```16```
    * lr: ```0.0001```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```1.0618```
        * Train Accuracy: ```0.4293```
        * Test Loss: ```1.1899```
        * Test Accuracy: ```0.3293```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit.py)
        * Saved Model: [ViT_audio_50_16_0.0001](https://drive.google.com/file/d/1-2JteyPMQvxtQk2YU1i99lMoIEAum07q/view?usp=drive_link)


* Experiment: Train Vision Transformer on faces cropped out of middle frames extracted out of videos from a subset of CREMA-D
    * Model: ViT
    * Number of classes: ```3``` (ANG, SAD, HAP)
    * Model trained on: faces cropped out of middle frames
    * Total number of samples: ```273```
    * Number of train samples: ```191```
    * Number of test samples: ```82```
    * Batch size: ```16```
    * lr: ```0.0001```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```0.1950```
        * Train Accuracy: ```0.9424```
        * Test Loss: ```1.9058```
        * Test Accuracy: ```0.6341```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit.py)
        * Saved Model: [ViT_video_50_16_0.0001](https://drive.google.com/file/d/1-2whSQKAXO_jJ4ROaeq8U4JcF5CRAvWT/view?usp=drive_link)


* Experiment: Train Vision Transformer on faces cropped from middle frames extracted out of videos and Mel spectrograms concatenated, from a subset of CREMA-D
    * Model: ViT
    * Number of classes: ```3``` (ANG, SAD, HAP)
    * Model trained on: faces cropped out of middle frames and Mel spectrograms concatenated
    * Total number of samples: ```273```
    * Number of train samples: ```191```
    * Number of test samples: ```82```
    * Batch size: ```16```
    * lr: ```0.0001```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```0.1222```
        * Train Accuracy: ```0.9581```
        * Test Loss: ```2.0300```
        * Test Accuracy: ```0.6098```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit.py)
        * Saved Model: [ViT_audio_video_50_16_0.0001](https://drive.google.com/file/d/1CCly-Bsybb2MaTlibX5SFXxhVak92jee/view?usp=drive_link)


* Experiment: Check cross-performance of audio-trained Vision Transformer on faces
    * Model: ViT
    * Number of classes: ```3``` (ANG, SAD, HAP)
    <!-- * Model trained on: Mel spectrograms -->
    * Total number of samples: ```273```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```82```
    * Batch size: ```16```
    * lr: ```0.0001```
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```1.0618```  -->
        <!-- * Train Accuracy: ```0.4293``` -->
        * Test Loss: ```3.7878```
        * Test Accuracy: ```0.3293```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT_crossed.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT_crossed.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT_crossed.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit_crossed.py)
        <!-- * Saved Model: [ViT_audio_50_16_0.0001](https://drive.google.com/file/d/1-2JteyPMQvxtQk2YU1i99lMoIEAum07q/view?usp=drive_link) -->

* Experiment: Check cross-performance of video-trained Vision Transformer on Mel spectrograms
    * Model: ViT
    * Number of classes: ```3``` (ANG, SAD, HAP)
    <!-- * Model trained on: Mel spectrograms -->
    * Total number of samples: ```273```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```82```
    * Batch size: ```16```
    * lr: ```0.0001```
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```1.0618```  -->
        <!-- * Train Accuracy: ```0.4293``` -->
        * Test Loss: ```3.7308```
        * Test Accuracy: ```0.3902```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT_crossed.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT_crossed.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT_crossed.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit_crossed.py)
        <!-- * Saved Model: [ViT_audio_50_16_0.0001](https://drive.google.com/file/d/1-2JteyPMQvxtQk2YU1i99lMoIEAum07q/view?usp=drive_link) -->

* Experiment: Train Vision Transformer on Mel spectrograms generated on fullscale CREMA-D
    * Model: ViT
    * Number of classes: ```6``` (ANG, SAD, HAP, DIS, FEA, NEU)
    * Model trained on: Mel spectrograms
    * Total number of samples: ```7442```
    * Number of train samples: ```5209```
    * Number of test samples: ```2233```
    * Batch size: ```16```
    * lr: ```0.0001```
    * dropout: ```0.4```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```1.7931```
        * Train Accuracy: ```0.1634```
        * Test Loss: ```1.7908```
        * Test Accuracy: ```0.1738```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT_fullscale.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT_fullscale.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT_fullscale.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit_fullscale.py)
        * Saved Model: [ViT_audio_fullscale_50_16_0.0001](https://drive.google.com/file/d/1PqjDAIMp6-y8b_TKGX3vHAeQKp-f1Q9Y/view?usp=drive_link)


* Experiment: Train Vision Transformer on faces cropped out of middle frames extracted out of videos on fullscale CREMA-D
    * Model: ViT
    * Number of classes: ```6``` (ANG, SAD, HAP, DIS, FEA, NEU)
    * Model trained on: faces cropped out of middle frames
    * Total number of samples: ```7442```
    * Number of train samples: ```5209```
    * Number of test samples: ```2233```
    * Batch size: ```16```
    * lr: ```0.0001```
    * dropout: ```0.4```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```0.6939```
        * Train Accuracy: ```0.7414```
        * Test Loss: ```1.4087```
        * Test Accuracy: ```0.5795```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT_fullscale.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT_fullscale.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT_fullscale.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit_fullscale.py)
        * Saved Model: [ViT_video_fullscale_50_16_0.0001](https://drive.google.com/file/d/11kJw0ksKJPdgP2OgZrfQO9gDHHvUVcM5/view?usp=drive_link)


* Experiment: Train Vision Transformer on faces cropped from middle frames extracted out of videos and Mel spectrograms concatenated, on fullscale CREMA-D
    * Model: ViT
    * Number of classes: ```6``` (ANG, SAD, HAP, DIS, FEA, NEU)
    * Model trained on: faces cropped out of middle frames and Mel spectrograms concatenated
    * Total number of samples: ```7442```
    * Number of train samples: ```5209```
    * Number of test samples: ```2233```
    * Batch size: ```16```
    * lr: ```0.0001```
    * dropout: ```0.4```
    * Loss: ```nn.CrossEntropyLoss()```
    * Train epochs: ```50```
    * Results:
        * Train Loss: ```0.8365```
        * Train Accuracy: ```0.6811```
        * Test Loss: ```1.3221```
        * Test Accuracy: ```0.5598```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT_fullscale.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT_fullscale.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT_fullscale.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit_fullscale.py)
        * Saved Model: [ViT_audio_video_fullscale_50_16_0.0001](https://drive.google.com/file/d/1xTV2Bbtp_pqlwVI_ik1W4-QJua-cn5lm/view?usp=drive_link)


* Experiment: Check cross-performance of audio-trained Vision Transformer on fullscale faces
    * Model: ViT
    * Number of classes: ```6``` (ANG, SAD, HAP, DIS, FEA, NEU)
    <!-- * Model trained on: Mel spectrograms -->
    * Total number of samples: ```7442```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```2233```
    * Batch size: ```16```
    * lr: ```0.0001```
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```1.0618```  -->
        <!-- * Train Accuracy: ```0.4293``` -->
        * Test Loss: ```1.7932```
        * Test Accuracy: ```0.1738```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT_fullscale_crossed.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT_fullscale_crossed.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT_fullscale_crossed.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit_fullscale_crossed.py)
        <!-- * Saved Model: [ViT_audio_50_16_0.0001](https://drive.google.com/file/d/1-2JteyPMQvxtQk2YU1i99lMoIEAum07q/view?usp=drive_link) -->

* Experiment: Check cross-performance of video-trained Vision Transformer on fullscale Mel spectrograms
    * Model: ViT
    * Number of classes: ```6``` (ANG, SAD, HAP, DIS, FEA, NEU)
    <!-- * Model trained on: Mel spectrograms -->
    * Total number of samples: ```7442```
    <!-- * Number of train samples: ```191``` -->
    * Number of test samples: ```2233```
    * Batch size: ```16```
    * lr: ```0.0001```
    * Loss: ```nn.CrossEntropyLoss()```
    <!-- * Train epochs: ```50``` -->
    * Results:
        <!-- * Train Loss: ```1.0618```  -->
        <!-- * Train Accuracy: ```0.4293``` -->
        * Test Loss: ```3.0131```
        * Test Accuracy: ```0.1720```
    * Reproduce:
        * Notebook: [```notebooks/ViT/audio_video_ViT_fullscale_crossed.ipynb```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/notebooks/ViT/audio_video_ViT_fullscale_crossed.ipynb)
        * Script: [```scripts/ViT/audio_video_ViT_fullscale_crossed.py```](https://github.com/ksanu1998/multimodal_course_project/blob/anuroop/scripts/ViT/audio_video_vit_fullscale_crossed.py)
        <!-- * Saved Model: [ViT_audio_50_16_0.0001](https://drive.google.com/file/d/1-2JteyPMQvxtQk2YU1i99lMoIEAum07q/view?usp=drive_link) -->


#### Resources
<!-- Audio feature extraction via spectrograms - https://github.com/DeepSpectrum/DeepSpectrum <br> -->
[GDrive](https://drive.google.com/drive/folders/1BhpgUDgbYwoTaTO6Yo8M3uR0Clw0bkiC?usp=drive_link) <br>
[GDoc](https://docs.google.com/document/d/1jN6ZpCUjqboJQLSFR-Osqlm5kRHGYX2a47GRADVYUPU/edit?usp=sharing)
