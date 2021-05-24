# BraTS3D

## 준비사항

반드시 ROOT 위치에 하기 3개 폴더를 넣어주시고, 해당 폴더에 BraTS2020에서 받은 폴더들을 분류해서 넣어주세요.

````
#### - MICCAI_BraTS20_TrainingData
#### - MICCAI_BraTS20_TrainingData_Part
#### - MICCAI_BraTS20_ValidationData
````
#### 예를 들면 아래와 같습니다.
````
#####  BraTS20_Training_001 폴더부터 BraTS20_Training_300 폴더까지는 MICCAI_BraTS20_TrainingData 내 위치
#####  BraTS20_Training_301 폴더부터 BraTS20_Training_362 폴더까지는 MICCAI_BraTS20_TrainingData_Part 내 위치
#####  BraTS20_Validation_001 폴더부터 BraTS20_Validation_125 폴더까지는 MICCAI_BraTS20_ValidationData 내 위치
````


## 결과값 저장

결과값은 Result 폴더가 새로 생성된 후에 BraTS20_Validation_NNN.nii.gz 같은 이름으로 저장됩니다.

## 추가 설명
Colab에서 돌릴거면 ColabPyRunner_BraTs20.ipynb를 열고 경로를 수정한 다음에 돌리시면 됩니다.

Local 환경에서 그냥 돌리실거면 PyCharm 또는 VSCode 등의 IDE를 설정한 후에 nnunet이나 unet을 실행하셔도 되고,
CMD에서 python unet.py 이런 식으로 돌려도 됩니다.
(CUDA로 실행하려면 최소 VRAM 8GB 이상의 그래픽 카드 필요)


### 참고로 Windows 10, Linux, MacOS 에서 전부 돌아갑니다.
