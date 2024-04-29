# A comparison of shared encoders for multimodal emotion recognition
## Course Project, CSCI 535, Spring 2024
Contributors - Sai Anuroop Kesanapalli, Riya Ranjan, Aashi Goyal, Wilson Tan

#### For 2D experiments

* Generate Mel spectrograms from WAV audio files by running ```wav_to_melspec.py```<br>
  ```python3 wav_to_melspec.py /path/to/WAVFiles /path/to/output```

The following describes running 2D experiments -- `ResNet18`, `VGG16`, `GoogLeNet`, and `ViT`, on audio, vision, and multimodal data. 


* Run ```audio_video_vit_fullscale.py``` to train ViT on audio, vision, and multimodal data. Pre-processed data (faces and spectrograms) is stored as ```.npy``` files [here](https://drive.google.com/drive/folders/1RbFeXB-B6r3BBEEsGDGudHrLR6Selkfq?usp=drive_link).

  ```python3 audio_video_vit_fullscale.py path/to/X.npy path/to/y.npy path/to/X_spec.npy, path/to/y_spec.npy```

* TODO (@aashi, @riya): Add similar one-liners for 2DCNNs (ResNet18, VGG16, GoogLeNet).


#### For 3D experiments

* Start with creating mel spectrograms by running:

  ```python3 wav_to_melspec_3d.py input_folder output_folder```

* Then create 3D Data:

  ```python3 create_3d_data.py video_folder spectrogram_folder output_folder```

* Now train your model with any of the following where ```modality``` can be ```[audio, vision, multi]```. ```pretrain_checkpoint``` is optional. The ImageNet pretrained model used here is provided in the ```models``` folder at [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) named ```rgb_imagenet.pt```. If ```pretrain_checkpoint``` is missing, the untrained I3D model will be used:

  ```python3 simple3d_train_test.py modality 3d_data_path output_path```

  ```python3 i3d_train_test.py modality 3d_data_path output_path pretrain_checkpoint```

  ```python3 videoMAE_train_test.py modality 3d_data_path output_path```

* For ablated tests. ```Checkpoint_path``` is optional but recommended, otherwise an untrained model is used:

  ```python3 simple3d_ablated_test.py modality 3d_data_path checkpoint_path```

  ```python3 i3d_ablated_test.py modality 3d_data_path checkpoint_path```


> **_NOTE:_**  We have retained notebooks/scripts/models for the pre-midterm (subset experiments) as well, in this repository.
