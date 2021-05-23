# BraTS3D

## 준의사항

#### - MICCAI_BraTS20_TrainingData
#### - MICCAI_BraTS20_TrainingData_Part
#### - MICCAI_BraTS20_ValidationData

반드시 ROOT 위치에 3개 폴더를 넣어주시고, 해당 폴더에 BraTS2020에서 받은 폴더들만 넣어주세요.

결과값은 Result 폴더가 새로 생성된 후에 BraTS20_Validation_NNN.nii.gz 같은 이름으로 저장됩니다.

Local 환경에서 그냥 돌리실거면 nnunet이나 unet을 실행하면 됩니다.
(CUDA로 실행하려면 최소 VRAM 8GB 이상의 그래픽 카드 필요)
