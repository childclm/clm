# -*- coding: utf-8 -*-
# @Time : 2019/3/16 10:11
# @Author : longxuandong
import fractions
import time

import numpy as np


# 条件特征下的数据归一化，使其归于{0,1}之间
def NormalizedData(DataMatrix, LabelLength):
    ConditionalLength = DataMatrix.shape[1] - LabelLength
    for i in range(ConditionalLength):
        MaxValue = max(DataMatrix[:, i])
        MinValue = min(DataMatrix[:, i])
        for j in range(DataMatrix.shape[0]):
            DataMatrix[j, i] = (DataMatrix[j, i] - MinValue) / (MaxValue - MinValue)
    return DataMatrix


# 读取数据文件并将数据进行归一化
def LoadFile(Path, Labellength):
    Marix = np.loadtxt(Path, dtype=np.float64, delimiter=",")
    Result = NormalizedData(Marix, Labellength)
    return Result


# 返回整个实例集合下的邻域阈值
# @jit(nopython=True, parallel=True)
def GetThreshold(DataMatrix, ParameterOmega, ConditionalFeatureLength):
    Temp = 0
    for i in range(ConditionalFeatureLength):
        T = float(np.std(DataMatrix[:, i]) / ParameterOmega)
        Temp += T
    Result = float((Temp / ConditionalFeatureLength) / ParameterOmega)
    return Result


# 获取所有实例下的邻域集合
# @jit(nopython=True, parallel=True)
def GetAllInstanceNeigborhoodList(DataMatrix, ParameterOmega, ConditionalFeatureLength):
    Result = []
    Data_Length = DataMatrix.shape[0]
    Threshold = GetThreshold(DataMatrix, ParameterOmega, ConditionalFeatureLength)
    for i in range(Data_Length):
        T = []
        Vector_1 = DataMatrix[i, 0:ConditionalFeatureLength]
        for j in range(Data_Length):
            Vector_2 = DataMatrix[j, 0:ConditionalFeatureLength]
            Distance = np.linalg.norm(Vector_1 - Vector_2)
            if Distance <= Threshold:
                T.append(j)
        Result.append(T)
    return Result, Threshold


# 获取实例集合下所有的标记集合
# @jit(nopython=True, parallel=True)
def GetAllLabelList(NeigborhoodList, LabelMatrix):
    Result = []
    for V in NeigborhoodList:
        Temp = []
        for i in V:
            Temp.append(list(LabelMatrix[i, :]))
        Result.append(Temp)
    return Result


# 将标记分布化整(处理加起来的和不满足1的情况)
def HandleLabelDistribution(ith_LabelDistribution):
    Result = 0
    Temp = []
    for i in ith_LabelDistribution:
        Result += fractions.Fraction(i)
    if Result != 0:
        for j in ith_LabelDistribution:
            Temp.append(str(fractions.Fraction(j) / Result))
    else:
        for j in ith_LabelDistribution:
            Temp.append(str(0))
    return Temp


# 将传统的标记转换为标记分布的形式
# @jit(nopython=True, parallel=True)
def ChangeTraditionalToLabelDistribution(LabelMatrix, LabelList, labelLength):
    Total_Length = LabelMatrix.shape[0]
    T = []
    Result = []
    for i in range(Total_Length):
        Temp = []
        Denominator = 0
        for V in LabelList[i]:
            Denominator += V.count(1)
        for j in range(labelLength):
            if int(LabelMatrix[i, j]) == 0:
                Temp.append(str(0))
            else:
                InstanceArray = np.array(LabelList[i])
                Numerator = int(sum(InstanceArray[:, j]))
                Temp.append(str(fractions.Fraction(Numerator, Denominator)))
        T.append(Temp)
    for Vec in T:
        Result.append(HandleLabelDistribution(Vec))
    return Result


def ReturnLabelDistribution(DataMatrix, ParameterOmega, LabelLength):
    ConditionalFeatureLength = DataMatrix.shape[1] - LabelLength
    Neigborhood_List, Data_Threshold = GetAllInstanceNeigborhoodList(DataMatrix, ParameterOmega,
                                                                     ConditionalFeatureLength)
    LabelMatrix = DataMatrix[:, ConditionalFeatureLength:]
    LabelMatrix = LabelMatrix.astype(np.int64)
    LabelList = GetAllLabelList(Neigborhood_List, LabelMatrix)
    LabelDistribution_Result = ChangeTraditionalToLabelDistribution(LabelMatrix, LabelList, LabelLength)
    return LabelDistribution_Result, Data_Threshold


if __name__ == '__main__':
    DataPath = "flags.csv"
    Omega = 0.5
    LabelNum = 7
    S = time.time()
    Data = LoadFile(DataPath, LabelNum)
    AAAA = ReturnLabelDistribution(Data, Omega, LabelNum)
    E = time.time()
    print(AAAA)
    print(E - S)
