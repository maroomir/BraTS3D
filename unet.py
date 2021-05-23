import math
import os.path
import pathlib
import random

import SimpleITK
import numpy
import torch
import torch.nn
import torch.nn.functional
import nibabel
from collections import defaultdict
from numpy import ndarray
from torch import tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


class BraTsTransform(object):
    def __init__(self,
                 bTrain=True):
        self.is_train = bTrain

    def __call__(self,
                 strFileName: str,
                 pImageT1: ndarray,
                 pImageT1Ce: ndarray,
                 pImageFlair: ndarray,
                 pImageT2: ndarray,
                 pTarget: ndarray = None):

        def to_tensor(pArray: ndarray):
            if numpy.iscomplexobj(pArray):
                pArray = numpy.stack((pArray.real, pArray.imag), axis=-1)
            return torch.from_numpy(pArray)

        def z_normalize(pTensor: tensor):
            pFuncMask = pTensor > 0
            pMean = pTensor[pFuncMask].mean()
            pStd = pTensor[pFuncMask].std()
            return torch.mul((pTensor - pMean) / pStd, pFuncMask.float())

        pImageT1 = z_normalize(to_tensor(pImageT1))
        pImageT1Ce = z_normalize(to_tensor(pImageT1Ce))
        pImageFlair = z_normalize(to_tensor(pImageFlair))
        pImageT2 = z_normalize(to_tensor(pImageT2))
        pTensorInput = torch.cat([pImageT1.unsqueeze(0), pImageT1Ce.unsqueeze(0),
                                  pImageFlair.unsqueeze(0), pImageT2.unsqueeze(0)], dim=0)
        if self.is_train:
            pTarget = to_tensor(pTarget)
            pLabelT1 = (pTarget == 0).float()
            pLabelT1Ce = (pTarget == 1).float()
            pLabelFlair = (pTarget == 2).float()
            pLabelT2 = (pTarget == 4).float()
            pTensorTarget = torch.cat([pLabelT1.unsqueeze(0), pLabelT1Ce.unsqueeze(0),
                                       pLabelFlair.unsqueeze(0), pLabelT2.unsqueeze(0)], dim=0)
            pTarget[pTarget == 4] = 3
            pTarget = pTarget.unsqueeze(0)
            return pTensorInput, pTensorTarget, pTarget, strFileName
        else:
            return pTensorInput, strFileName


class BraTsDataset(Dataset):
    def __init__(self,
                 strRoot: str,
                 pTransform=None,
                 dSamplingRate=1.0,
                 bTrain=True):
        self.transform = pTransform
        self.is_train = bTrain
        self.collect_dirs = []
        pListFileRoot = list(pathlib.Path(strRoot).iterdir())
        if dSamplingRate < 1.0:
            random.seed(42)
            random.shuffle(pListFileRoot)
            nCountFiles = round(len(pListFileRoot) * dSamplingRate)
            pListFileRoot = pListFileRoot[:nCountFiles]
        for iPath in sorted(pListFileRoot):
            if iPath.is_dir():
                self.collect_dirs.append(iPath)

    def __len__(self):
        return len(self.collect_dirs)

    def __getitem__(self, item):
        pPath = self.collect_dirs[item]
        strFileT1 = os.path.split(str(pPath))[-1] + '_t1.nii.gz'
        strFileT1Ce = os.path.split(str(pPath))[-1] + '_t1ce.nii.gz'
        strFileFlair = os.path.split(str(pPath))[-1] + '_flair.nii.gz'
        strFileT2 = os.path.split(str(pPath))[-1] + '_t2.nii.gz'
        strFileSeg = os.path.split(str(pPath))[-1] + '_seg.nii.gz'
        pImageT1 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(os.path.join(str(pPath), strFileT1))) \
            .astype(numpy.float32)
        pImageT1Ce = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(os.path.join(str(pPath), strFileT1Ce))) \
            .astype(numpy.float32)
        pImageFlair = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(os.path.join(str(pPath), strFileFlair))) \
            .astype(numpy.float32)
        pImageT2 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(os.path.join(str(pPath), strFileT2))) \
            .astype(numpy.float32)
        if self.is_train:
            pImageTarget = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(os.path.join(str(pPath), strFileSeg))) \
                .astype(numpy.float32)
            return self.transform(pPath.name, pImageT1, pImageT1Ce, pImageFlair, pImageT2, pImageTarget)
        else:
            return self.transform(pPath.name, pImageT1, pImageT1Ce, pImageFlair, pImageT2)


def get_dice_loss(pTensorPredict: tensor,  # Batch, 4, ??, ??, ??
                  pTensorTarget: tensor,  # Batch, 4, ??, ??, ??
                  dSmooth=1e-4):
    pTensorDiceBG = get_dice_coefficient(pTensorPredict[:, 0, :, :, :],
                                         pTensorTarget[:, 0, :, :, :],
                                         dSmooth)
    pTensorDiceNCR = get_dice_coefficient(pTensorPredict[:, 1, :, :, :],
                                          pTensorTarget[:, 1, :, :, :],
                                          dSmooth)
    pTensorDiceED = get_dice_coefficient(pTensorPredict[:, 2, :, :, :],
                                         pTensorTarget[:, 2, :, :, :],
                                         dSmooth)
    pTensorDiceSET = get_dice_coefficient(pTensorPredict[:, 3, :, :, :],
                                          pTensorTarget[:, 3, :, :, :],
                                          dSmooth)
    return 1 - (pTensorDiceBG + pTensorDiceNCR + pTensorDiceED + pTensorDiceSET) / 4


def get_dice_coefficient(pTensorPredict: tensor,
                         pTensorTarget: tensor,
                         dSmooth=1e-4):
    pTensorPredict = pTensorPredict.contiguous().view(-1)
    pTensorTarget = pTensorTarget.contiguous().view(-1)
    pTensorIntersection = (pTensorPredict * pTensorTarget).sum()
    pTensorCoefficient = (2.0 * pTensorIntersection + dSmooth) / (pTensorPredict.sum() + pTensorTarget.sum() + dSmooth)
    return pTensorCoefficient


class Convolution(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 dRateDropout: float = 0.3):
        super(Convolution, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv3d(nDimInput, nDimOutput, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm3d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout3d(dRateDropout),
            torch.nn.Conv3d(nDimOutput, nDimOutput, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm3d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout3d(dRateDropout)
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class UpSampler(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int):
        super(UpSampler, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(nDimInput, nDimOutput, kernel_size=2, stride=2, bias=True),
            torch.nn.InstanceNorm3d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class UNet3D(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nChannel: int,
                 nCountDepth: int,
                 dRateDropout: float = 0.3):
        super(UNet3D, self).__init__()
        # Init Encoders and Decoders
        self.encoders = torch.nn.ModuleList([Convolution(nDimInput, nChannel, dRateDropout)])
        for i in range(nCountDepth - 1):
            self.encoders += [Convolution(nChannel, nChannel * 2, dRateDropout)]
            nChannel *= 2
        self.down_sampler = torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        self.worker = Convolution(nChannel, nChannel * 2, dRateDropout)
        self.decoders = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        for i in range(nCountDepth - 1):
            self.up_samplers += [UpSampler(nChannel * 2, nChannel)]
            self.decoders += [Convolution(nChannel * 2, nChannel, dRateDropout)]
            nChannel //= 2
        self.up_samplers += [UpSampler(nChannel * 2, nChannel)]
        self.decoders += [
            torch.nn.Sequential(
                Convolution(nChannel * 2, nChannel, dRateDropout),
                torch.nn.Conv3d(nChannel, nDimOutput, kernel_size=1, stride=1),
                torch.nn.Softmax(dim=1)
            )
        ]

    def __padding(self, pTensorX: tensor):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        nBatch, nFlag, nDensity, nHeight, nWidth = pTensorX.shape
        nWidthBitMargin = ((nWidth - 1) | 15) + 1  # 15 = (1111)
        nHeightBitMargin = ((nHeight - 1) | 15) + 1
        nDensityBitMargin = ((nDensity - 1) | 15) + 1
        pPadWidth = floor_ceil((nWidthBitMargin - nWidth) / 2)
        pPadHeight = floor_ceil((nHeightBitMargin - nHeight) / 2)
        pPadDensity = floor_ceil((nDensityBitMargin - nDensity) / 2)
        x = torch.nn.functional.pad(pTensorX, pPadWidth + pPadHeight + pPadDensity)
        return x, (pPadDensity, pPadHeight, pPadWidth, nDensityBitMargin, nHeightBitMargin, nWidthBitMargin)

    def __unpadding(self, x, pPadDensity, pPadHeight, pPadWidth, nDensityMargin, nHeightMargin, nWidthMargin):
        return x[..., pPadDensity[0]:nDensityMargin - pPadDensity[1], pPadHeight[0]:nHeightMargin - pPadHeight[1],
               pPadWidth[0]:nWidthMargin - pPadWidth[1]]

    def forward(self, pTensorX: tensor):
        pTensorX, pPadOption = self.__padding(pTensorX)
        pListStack = []
        pTensorResult = pTensorX
        # Apply down sampling layers
        for i, pEncoder in enumerate(self.encoders):
            pTensorResult = pEncoder(pTensorResult)
            pListStack.append(pTensorResult)
            pTensorResult = self.down_sampler(pTensorResult)
        pTensorResult = self.worker(pTensorResult)
        # Apply up sampling layers
        for pSampler, pDecoder in zip(self.up_samplers, self.decoders):
            pTensorAttached = pListStack.pop()
            pTensorResult = pSampler(pTensorResult)
            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            pPadding = [0, 0, 0, 0]  # left, right, top, bottom
            if pTensorResult.shape[-1] != pTensorAttached.shape[-1]:
                pPadding[1] = 1  # Padding right
            if pTensorResult.shape[-2] != pTensorAttached.shape[-2]:
                pPadding[3] = 1  # Padding bottom
            if sum(pPadding) != 0:
                pTensorResult = torch.nn.functional.pad(pTensorResult, pPadding, "reflect")
            pTensorResult = torch.cat([pTensorResult, pTensorAttached], dim=1)
            pTensorResult = pDecoder(pTensorResult)
        pListStack.clear()  # To Memory Optimizing
        pTensorResult = self.__unpadding(pTensorResult, *pPadOption)
        return pTensorResult


def __process_train(nEpoch: int, pModel: UNet3D, pDataLoader: DataLoader, pOptimizer: Adam):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Perform a training using the defined network
    pModel.train()
    # Warp the iterable Data Loader with TQDM
    pBar = tqdm(enumerate(pDataLoader))
    nLengthSample = 0
    nTotalLoss = 0
    nTotalAcc = 0
    for i, (pTensorInput, pTensorTarget, pTensorLabel, strFileName) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.to(pDevice)
        pTensorTarget = pTensorTarget.to(pDevice)
        pTensorLabel = pTensorLabel.to(pDevice)
        # Pass the input data through the defined network architecture
        pTensorOutput = pModel(pTensorInput)  # Shape : (batch, 4, 155, 240, 240)
        pTensorPredict = torch.argmax(pTensorOutput, dim=1)  # shape : (batch, 155, 240, 240)
        # Compute a loss function
        pTensorLoss = get_dice_loss(pTensorOutput, pTensorTarget)  # shape : (batch, 155, 240, 240)
        # Compute network accuracy
        pPredictBG = (pTensorPredict == 0)
        pTargetBG = (pTensorLabel == 0).squeeze(1)  # shape : (batch, 155, 240, 240)
        pDiceBG = get_dice_coefficient(pPredictBG, pTargetBG)
        pPredictNCR = (pTensorPredict == 1)
        pTargetNCR = (pTensorLabel == 1).squeeze(1)
        pDiceNCR = get_dice_coefficient(pPredictNCR, pTargetNCR)
        pPredictED = (pTensorPredict == 2)
        pTargetED = (pTensorLabel == 2).squeeze(1)
        pDiceED = get_dice_coefficient(pPredictED, pTargetED)
        pPredictSET = (pTensorPredict == 3)
        pTargetSET = (pTensorLabel == 3).squeeze(1)
        pDiceSET = get_dice_coefficient(pPredictSET, pTargetSET)
        # Perform backpropagation to update network parameters
        pOptimizer.zero_grad()
        pTensorLoss.backward()
        pOptimizer.step()
        pBar.set_description('Epoch:{:3d} [{}/{} {:.2f}%], Loss={:.4f}, BG={:.4f}, NCR={:.4f}, ED={:4f}, SET={:.4f}'.
                             format(nEpoch, i, len(pDataLoader), 100.0 * (i / len(pDataLoader)),
                                    pTensorLoss.item(), pDiceBG, pDiceNCR, pDiceED, pDiceSET))
        # Fix the CUDA Out of Memory problem
        del pTensorOutput
        del pTensorPredict
        del pTensorLoss
        torch.cuda.empty_cache()


def __process_evaluate(pModel: UNet3D, pDataLoader: DataLoader):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Perform an evaluation using the defined network
    pModel.eval()
    # Warp the iterable Data Loader with TQDM
    pBar = tqdm(enumerate(pDataLoader))
    nLengthSample = 0
    nTotalLoss = 0
    with torch.no_grad():
        for i, (pTensorInput, pTensorTarget, pTensorLabel, strFileName) in pBar:
            # Move data and label to device
            pTensorInput = pTensorInput.to(pDevice)
            pTensorTarget = pTensorTarget.to(pDevice)
            # Pass the input data through the defined network architecture
            pTensorOutput = pModel(pTensorInput)  # Module
            # Compute a loss function
            pTensorLoss = get_dice_loss(pTensorOutput, pTensorTarget)
            nTotalLoss += pTensorLoss.item() * len(pTensorTarget)
            nLengthSample += len(pTensorTarget)
            pBar.set_description('{}/{} {:.2f}%, Loss={:.4f}'.
                                 format(i, len(pDataLoader), 100.0 * (i / len(pDataLoader)),
                                        nTotalLoss / nLengthSample))
    # Fix the CUDA Out of Memory problem
    del pTensorOutput
    del pTensorLoss
    torch.cuda.empty_cache()
    return nTotalLoss / nLengthSample


def __save_result_to_nii(pDicSegmentation: dict, strDirValidation: str, pPathResult=pathlib.Path('Result/')):
    pPathResult.mkdir(exist_ok=True)
    for strPath, pData in pDicSegmentation.items():
        strFileDefault = strPath + '_t1.nii.gz'
        pImageDefault = nibabel.load(str(os.path.join(strDirValidation, strPath, strFileDefault)))
        pData = numpy.transpose(numpy.squeeze(pData), (2, 1, 0))
        pImageResult = nibabel.Nifti1Image(pData, pImageDefault.affine, pImageDefault.header)
        nibabel.save(pImageResult, str(os.path.join(str(pPathResult), strPath + '.nii.gz')))


def train(nEpoch: int,
          strRoot: str,
          strModelPath: str = None,
          nChannel=8,
          nCountDepth=4,
          nBatchSize=1,
          nCountWorker=2,  # 0: CPU / 2 : GPU
          dRateDropout=0.3,
          dLearningRate=0.0001,
          bInitEpoch=False,
          ):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define the training and testing data-set
    pTrainSet = BraTsDataset(strRoot=strRoot + 'MICCAI_BraTS20_TrainingData/', pTransform=BraTsTransform(),
                             dSamplingRate=1.0, bTrain=True)
    pTrainLoader = DataLoader(dataset=pTrainSet, batch_size=nBatchSize, shuffle=True,
                              num_workers=nCountWorker, pin_memory=True)
    pValidationSet = BraTsDataset(strRoot=strRoot + 'MICCAI_BraTS20_TrainingData_Part/', pTransform=BraTsTransform(),
                                  dSamplingRate=1.0, bTrain=True)
    pValidationLoader = DataLoader(dataset=pValidationSet, batch_size=1, shuffle=False,
                                   num_workers=nCountWorker, pin_memory=True)
    # Define a network model
    pModel = UNet3D(nDimInput=4, nDimOutput=4, nChannel=nChannel, nCountDepth=nCountDepth,
                    dRateDropout=dRateDropout).to(pDevice)
    # Set the optimizer with adam
    pOptimizer = torch.optim.Adam(pModel.parameters(), lr=dLearningRate)
    # Set the scheduler
    pScheduler = torch.optim.lr_scheduler.StepLR(pOptimizer, step_size=1)
    # Load pre-trained model
    nStart = 0
    print("Directory of the pre-trained model: {}".format(strModelPath))
    if strModelPath is not None and os.path.exists(strModelPath) and bInitEpoch is False:
        pModelData = torch.load(strModelPath)
        nStart = pModelData['epoch']
        pModel.load_state_dict(pModelData['model'])
        pOptimizer.load_state_dict(pModelData['optimizer'])
        print("## Successfully load the model at {} epochs!".format(nStart))
    # Train and Test Repeat
    dMinLoss = 10000.0
    nCountDecrease = 0
    for iEpoch in range(nStart, nEpoch + 1):
        # Train the network
        __process_train(iEpoch, pModel=pModel, pDataLoader=pTrainLoader, pOptimizer=pOptimizer)
        # Test the network
        dLoss = __process_evaluate(pModel=pModel, pDataLoader=pValidationLoader)
        pScheduler.step()
        # Rollback the model when loss is NaN
        if math.isnan(dLoss):
            if strModelPath is not None and os.path.exists(strModelPath):
                # Reload the best model and decrease the learning rate
                pModelData = torch.load(strModelPath)
                pModel.load_state_dict(pModelData['model'])
                nStart = pModelData['epoch']
                pOptimizerData = pModelData['optimizer']
                pOptimizerData['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                pOptimizer.load_state_dict(pOptimizerData)
                print("## Rollback the model at {} epochs!".format(nStart))
                nCountDecrease = 0
        # Save the optimal model
        elif dLoss < dMinLoss:
            dMinLoss = dLoss
            torch.save({'epoch': iEpoch, 'model': pModel.state_dict(), 'optimizer': pOptimizer.state_dict()},
                       strModelPath)
            nCountDecrease = 0
        else:
            nCountDecrease += 1
            # Decrease the learning rate by 2 when the test loss decrease 3 times in a row
            if nCountDecrease == 3:
                pDicOptimizerState = pOptimizer.state_dict()
                pDicOptimizerState['param_groups'][0]['lr'] /= 2
                pOptimizer.load_state_dict(pDicOptimizerState)
                print('learning rate is divided by 2')
                nCountDecrease = 0


def test(strRoot: str,
         strModelPath: str,
         nChannel=8,
         nCountDepth=4,
         nCountWorker=2,  # 0: CPU / 2 : GPU
         dRateDropout=0.3):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define a network model
    pModel = UNet3D(nDimInput=4, nDimOutput=4, nChannel=nChannel, nCountDepth=nCountDepth,
                    dRateDropout=dRateDropout).to(pDevice)
    pModelData = torch.load(strModelPath)
    pModel.load_state_dict(pModelData['model'])
    pModel.eval()
    print("Successfully load the Model in path")
    # Define the validation data-set
    pTestSet = BraTsDataset(strRoot=strRoot + 'MICCAI_BraTS20_ValidationData/', pTransform=BraTsTransform(bTrain=False),
                            dSamplingRate=1.0, bTrain=False)
    pTestLoader = DataLoader(dataset=pTestSet, batch_size=1, shuffle=False,
                             num_workers=nCountWorker, pin_memory=True)
    pBar = tqdm(pTestLoader)
    pDicOutput = defaultdict(list)
    pDicResult = defaultdict(list)
    with torch.no_grad():
        for pTensorInput, strFileName in pBar:
            pTensorInput = pTensorInput.to(pDevice)
            pTensorResult = pModel(pTensorInput).to('cpu')
            pTensorResult = torch.argmax(pTensorResult, dim=1)
            pTensorResult[pTensorResult == 3] = 4
            for i in range(pTensorResult.shape[0]):
                pDicOutput[strFileName[i]].append(pTensorResult[i].numpy())
    # Collect and sort the result dictionary
    for strName, pArrayData in pDicOutput.items():
        pDicResult[strName].append(numpy.stack([pData for pData in sorted(pArrayData)]))
    # Save the result to nii
    __save_result_to_nii(pDicResult, strRoot + 'MICCAI_BraTS20_ValidationData/')


if __name__ == '__main__':
    train(nEpoch=100,
          strRoot='',
          strModelPath='model_unet.pth',
          nChannel=8,  # 8 >= VRAM 9GB / 4 >= VRAM 6.5GB
          nCountDepth=4,
          nBatchSize=1,
          nCountWorker=2,  # 0= CPU / 2 >= GPU
          dRateDropout=0.3,
          dLearningRate=0.01,
          bInitEpoch=False)
    test(strRoot='',
         strModelPath='model_unet.pth',
         nChannel=8,  # 8 : colab / 4 : RTX2070
         nCountDepth=4,
         nCountWorker=2,  # 0: CPU / 2 : GPU
         dRateDropout=0.3)
