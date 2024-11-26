import math
import os
import shutil
import pandas as pd
import numpy as np
from fractions import Fraction

from matplotlib import pyplot as plt
import seaborn as sns
import Label_Enhanced_UsingNumba
from sklearn.preprocessing import MinMaxScaler


class FeatureSelectionAlgorithm:
    def __init__(self, file_path=None):
        self.file_path = file_path  # 读取的文件路径
        self.output_dir = None  # 增强结果可视化路径
        self.data = None
        self.result = None  # 归一结果
        self.labellist = None  # 增强结果
        self.target_column = None  # 目标列
        self.label_length = None  # 目标列长度

    def get_columns(self):
        return list(self.data.columns)

    def exchange(self, response):
        columns = self.get_columns()
        # 重新组合列名，将选择的列放到最后
        new_columns = [col for col in columns if col not in response] + response

        # 根据新的列顺序创建新的 DataFrame
        self.data = self.data[new_columns].copy()

    # 加载文件
    def loadfile(self):
        if self.file_path:
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.file_path)
            else:
                raise FileNotFoundError('File type not supported')
        else:
            raise FileNotFoundError('File path not specified')

    # 归一化 将每个特征下的数据归一化{0,1}
    def normalization(self):
        minmax = MinMaxScaler(feature_range=(0, 1))
        data = np.array(self.data, float)
        self.result = minmax.fit_transform(data)

    # 获取数据集中的邻域阈值
    def GetNeigborhoodThreshold(self, ConditionalFeatues, Omega):
        """
        获取数据集的领域阈值
        :param ConditionalFeatues:
        :param Omega:
        :return:
        """
        Temp = 0.0
        Length = ConditionalFeatues.__len__()
        for f in ConditionalFeatues:
            Column_Vector = self.result[:, f]
            T = np.std(Column_Vector) / Omega
            Temp += T
        Result = float((Temp / Length) / Omega)
        return Result

    # 构建标记集合的0~1对应实例集合
    def ConstructLiInstanceList(self, LabelList, weight_factor=0.1):
        """
        基于标记增强，生成 [0,1] 小数范围的标记集合对应实例集合。
        :param LabelList: 要处理的标记集合
        :param weight_factor: 控制标记增强的小数范围权重
        :return: 包含增强标记的小数结果列表
        1. 软标记（Soft Labeling）的理论支持
        传统的二分类标记（0 和 1）被称为硬标记（Hard Labeling），但在许多机器学习任务中，特别是在深度学习领域，**软标记（Soft Labeling）**被证明可以提高模型的鲁棒性和泛化能力。

        软标记的公式：𝑦′=y+ϵ
        其中：𝑦是原始的离散标记（0 或 1）。
        𝜖是一个随机噪声或加权偏移量，用于引入连续性。
        通过将标记值从离散值转化为带权重的小数（如 0.1 或 0.9），可以更准确地表示数据点的实际类别分布。这种方式在半监督学习和对抗训练中有较广泛的应用。
        文献支持：
        Hinton, G., et al. "Distilling the Knowledge in a Neural Network" (2015). 提出使用**温度软化（Softmax with Temperature）**的方式将分类标记转化为连续概率分布。
        """
        ResultList = []
        for Index in LabelList:
            T = []
            for Criterion in [0, 1]:
                Temp = [
                    (i, Criterion + weight_factor * (2 * (i % 2) - 1))  # 增加小数权重
                    for i, x in enumerate(self.result[:, Index])
                    if x == Criterion
                ]
                T.append(Temp)
            ResultList.append(T)
        # 创建 img 文件夹
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
                                       "static\\img\\%s"%self.file_path.split('.')[0].split('\\')[-1])
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        label_length = len(LabelList)
        for i in range(label_length):
            self.save_plot_to_local(ResultList, col_index=i, columns=self.get_columns(), output_dir=self.output_dir,
                                    label_length=len(LabelList))
        return ResultList

    def save_plot_to_local(self, ResultList, col_index, columns, output_dir, label_length):
        """
           从 ResultList 中提取目标列的增强值，并绘制直方图。

           :param ResultList: 标记增强结果列表
           :param col_index: 要绘制的目标列索引
           """
        # 提取目标列的增强值
        target_data = []
        column_data = ResultList[col_index]  # 获取指定目标列的 Criterion 列表
        for criterion_data in column_data:  # 遍历 Criterion 0 和 1
            target_data.extend([val[1] for val in criterion_data])  # 提取增强值

        # 绘制直方图
        plt.figure(figsize=(8, 5))
        sns.histplot(target_data, bins=30, kde=True, color="skyblue")
        plt.title(f"Histogram of Enhanced Values {columns[len(columns)-label_length+col_index]}")
        plt.xlabel("Enhanced Values")
        plt.ylabel("Frequency")
        # 保存到 img 文件夹
        output_file = os.path.join(output_dir, f"histogram_column_{columns[len(columns)-label_length+col_index]}.png")
        plt.savefig(output_file)
        plt.close()
        plt.show()

    # 构建每个特征下的每个实例的邻域集合的矩阵
    def ConstructNeighborhoodMatrix_Feature(self, ConditionalFeatures, Threshold):
        Features_Length = ConditionalFeatures.__len__()
        Data_Length = self.result.shape[0]
        Result_Matrix = np.zeros((Features_Length, Data_Length), dtype=list)
        # TODO 这里需要改参数
        for f in ConditionalFeatures[0:1000]:
            for Xi in range(Data_Length):
                T = []
                for Xj in range(Data_Length):
                    Vector1 = self.result[Xi, f]
                    Vector2 = self.result[Xj, f]
                    Distance = np.linalg.norm(Vector1 - Vector2)
                    if Distance <= Threshold:
                        T.append(Xj)
                Result_Matrix[f][Xi] = T  # T为Xi在f下的领域矩阵
        return Result_Matrix  # 返回的是所有样例在所有特征下的领域样例所构成的矩阵


    # 获取标记分布下的标记权重
    def Get_Label_Weight(self, LabelLength, Omega):
        """
        获取标记分布：调用 Label_Enhanced_UsingNumba.ReturnLabelDistribution 方法，传入结果集 self.result、参数 Omega 和标记长度 LabelLength。该方法返回标记分布结果 Label_Distribution_Result。

        转换为 NumPy 数组：将 Label_Distribution_Result 转换为 NumPy 数组 Maritix，以便于后续的数组操作。

        初始化权重结果列表：创建一个空的列表 Weight_Result 用于存储每个标记的权重。

        循环计算每个标记的权重：

        对于每个标记索引 index（范围从 0 到 LabelLength-1），提取对应的分布值 T。
        初始化 Total_Sum 为 0，用于累加分数。
        遍历 T 中的每个值 i，将其转换为分数类型 Fraction(i)，并累加到 Total_Sum 中。
        计算权重：通过 Fraction(Total_Sum, self.result.shape[0]) 计算该标记的权重 Temp，即该标记分布总和与结果集大小的比值。

        保存权重：将计算得到的权重 Temp 添加到 Weight_Result 列表中。

        返回结果：最终返回 Weight_Result 列表，包含每个标记的权重。
        """
        Label_Distribution_Result, _ = Label_Enhanced_UsingNumba.ReturnLabelDistribution(self.result, Omega,
                                                                                         LabelLength)
        Maritix = np.array(Label_Distribution_Result)
        Weight_Result = []
        for index in range(LabelLength):
            T = Maritix[:, index]
            Total_Sum = 0
            for i in T:
                Total_Sum += Fraction(i)
            Temp = Fraction(Total_Sum, self.result.shape[0])
            Weight_Result.append(Temp)
        return Weight_Result


    # 计算候选特征与已选特征集合的交互性
    def ComputeInteractiveBetween_fi_and_S(self, fi, Selected_S, LabelIndexList, RelevanceMatrix,
                                           RedundancyMatrix, NeighborhoodMatrix_Feature, Threshold):
        Mid = 0.0
        Label_Length = LabelIndexList.__len__()
        DataLength = self.result.shape[0]
        # TODO 这里需要改参数
        if fi > 1000:
            for Xi in range(DataLength):
                Neigborhood_fi_List = []
                Vector_1 = self.result[Xi][fi]
                for Xj in range(DataLength):
                    Vector_2 = self.result[Xj][fi]
                    Distance = np.linalg.norm(Vector_1 - Vector_2)
                    if Distance <= Threshold:
                        Neigborhood_fi_List.append(Xj)
                T = Neigborhood_fi_List.__len__()
                Temp = float(T / DataLength)
                Mid += math.log2(Temp)
            Entropy_fi = -1 / DataLength * Mid
        else:
            for Xi in range(DataLength):
                Neigborhood_fi_List = NeighborhoodMatrix_Feature[fi][Xi]
                T = Neigborhood_fi_List.__len__()
                Temp = float(T / DataLength)
                Mid += math.log2(Temp)
            Entropy_fi = -1 / DataLength * Mid
        Result = 0.0
        for s in Selected_S:
            Value_1 = 0.0
            if RedundancyMatrix[fi][s] == 0:
                for Lj in range(Label_Length):
                    Value_1 += (1 - (RelevanceMatrix[fi][Lj] / Entropy_fi)) * RedundancyMatrix[s][fi]
            else:
                for Lj in range(Label_Length):
                    Value_1 += (1 - (RelevanceMatrix[fi][Lj] / Entropy_fi)) * RedundancyMatrix[fi][s]
            Result += Value_1
        # return ResultComputeInteractiveBetween_fi_and_candidate
        return Result


    # 构建特征依赖度矩阵
    def ConstructRelevanceMatrix(self, CondintionalFeatures, LabelList, Weight_List,
                                 NeighborhoodMatrix_Feature, LiisLabelList, Threshold):
        """
        :param CondintionalFeatures:
        :param LabelList:
        :param Weight_List:
        :param NeighborhoodMatrix_Feature:
        :param LiisLabelList: ?
        :param Threshold:
        :return:
        """
        LabelLength_LabelList = LabelList.__len__()
        Relevance_Matrix = np.zeros((CondintionalFeatures.__len__(), LabelLength_LabelList), dtype=float)
        StarIndex = LabelList[0]  # 开始为标签列表索引
        Total_Length = self.result.shape[0]
        # TODO 这里需要改参数
        for f in CondintionalFeatures:
            # print(f)
            if f > 1000:
                Column = -1
                for l in LabelList:
                    Column += 1
                    Weight_Index = l - StarIndex
                    Mid = 0.0
                    for Xi in range(Total_Length):
                        Vector_1 = self.result[Xi][f]
                        Neigborhood_f_List = []
                        for Xj in range(Total_Length):
                            Vector_2 = self.result[Xj][f]
                            Distance = np.linalg.norm(Vector_1 - Vector_2)
                            if Distance <= Threshold:
                                Neigborhood_f_List.append(Xj)
                        Neigborhood_li_List = LiisLabelList[Weight_Index][int(self.result[Xi][l])]
                        T = list(set(Neigborhood_f_List) & set(Neigborhood_li_List))
                        if T.__len__() == 0:
                            Temp = 1e-10
                        else:
                            Temp = float(
                                Neigborhood_f_List.__len__() * Neigborhood_li_List.__len__() / (
                                        Total_Length * T.__len__()))
                        Mid += math.log2(Temp)
                    Result = float((-1 / Total_Length) * Weight_List[Weight_Index]) * Mid
                    Relevance_Matrix[f, Column] = Result
            else:
                Column = -1
                for l in LabelList:
                    Column += 1
                    Weight_Index = l - StarIndex
                    Mid = 0.0
                    for Xi in range(Total_Length):
                        Neigborhood_f_List = NeighborhoodMatrix_Feature[f][Xi]
                        Neigborhood_li_List = LiisLabelList[Weight_Index][int(self.result[Xi][l])]
                        T = list(set(Neigborhood_f_List) & set(Neigborhood_li_List))
                        intersection_len = T.__len__()
                        if intersection_len == 0:
                            Temp = 1e-10  # 选择一个很小的正数来代替
                        else:
                            Temp = float(Neigborhood_f_List.__len__() * Neigborhood_li_List.__len__() / (
                                    Total_Length * intersection_len))
                        Mid += math.log2(Temp)
                    Result = float((-1 / Total_Length) * Weight_List[Weight_Index]) * Mid
                    Relevance_Matrix[f, Column] = Result
        return Relevance_Matrix
        # """
        # 为什么当特征小于1000时，在该特征下的样例的领域矩阵，可能跟存储有关
        # """


    # 构建特征冗余度矩阵
    def ConstructRedundancyMatrix(self, ConditionalFeatures, NeighborhoodMatrix_Feature, Threshold):
        """
        :param ConditionalFeatures: 条件特征
        :param NeighborhoodMatrix_Feature: 在该特征下的领域特征矩阵
        :param Threshold:
        :return:
        """
        Length_ConditionalFeatures = ConditionalFeatures.__len__()
        Redundancuy_Matrix = np.zeros((Length_ConditionalFeatures, Length_ConditionalFeatures), dtype=float)
        DataLength = self.result.shape[0]
        for fi in ConditionalFeatures:
            for fj in range(fi + 1, Length_ConditionalFeatures):
                # TODO 这里需要改参数
                if (fi > 1000) and (fj <= 1000):
                    Mid = 0.0
                    for Xi in range(DataLength):
                        Neigborhood_fi_List = []
                        Vector_1 = self.result[Xi][fi]
                        for Xj in range(DataLength):
                            Vector_2 = self.result[Xj][fi]
                            Distance = np.linalg.norm(Vector_1 - Vector_2)
                            if Distance <= Threshold:
                                Neigborhood_fi_List.append(Xj)
                        Neigborhood_fj_List = NeighborhoodMatrix_Feature[fj][Xi]
                        T = list(set(Neigborhood_fi_List) & set(Neigborhood_fj_List))
                        Temp = float(
                            Neigborhood_fi_List.__len__() * Neigborhood_fj_List.__len__() / (DataLength * T.__len__()))
                        Mid += math.log2(Temp)
                    Result = float((-1 / DataLength) * Mid)
                    Redundancuy_Matrix[fi, fj] = Result
                elif (fi <= 1000) and (fj > 1000):
                    Mid = 0.0
                    for Xi in range(DataLength):
                        Neigborhood_fj_List = []
                        Vector_1 = self.result[Xi][fj]
                        for Xj in range(DataLength):
                            Vector_2 = self.result[Xj][fj]
                            Distance = np.linalg.norm(Vector_1 - Vector_2)
                            if Distance <= Threshold:
                                Neigborhood_fj_List.append(Xj)
                        Neigborhood_fi_List = NeighborhoodMatrix_Feature[fi][Xi]
                        T = list(set(Neigborhood_fi_List) & set(Neigborhood_fj_List))
                        Temp = float(
                            Neigborhood_fi_List.__len__() * Neigborhood_fj_List.__len__() / (DataLength * T.__len__()))
                        Mid += math.log2(Temp)
                    Result = float((-1 / DataLength) * Mid)
                    Redundancuy_Matrix[fi, fj] = Result
                elif (fi <= 1000) and (fj <= 1000):
                    Mid = 0.0
                    for Xi in range(DataLength):
                        Neigborhood_fi_List = NeighborhoodMatrix_Feature[fi][Xi]
                        Neigborhood_fj_List = NeighborhoodMatrix_Feature[fj][Xi]
                        T = list(set(Neigborhood_fi_List) & set(Neigborhood_fj_List))
                        Temp = float(
                            Neigborhood_fi_List.__len__() * Neigborhood_fj_List.__len__() / (DataLength * T.__len__()))
                        Mid += math.log2(Temp)
                    Result = float((-1 / DataLength) * Mid)
                    Redundancuy_Matrix[fi, fj] = Result
                else:
                    Mid = 0.0
                    for Xi in range(DataLength):
                        Neigborhood_fi_List = []
                        Neigborhood_fj_List = []
                        Vector_fi_1 = self.result[Xi][fi]
                        Vector_fj_1 = self.result[Xi][fj]
                        for Xj in range(DataLength):
                            Vector_fi_2 = self.result[Xj][fi]
                            Vector_fj_2 = self.result[Xj][fj]
                            Distance1 = np.linalg.norm(Vector_fi_1 - Vector_fi_2)
                            Distance2 = np.linalg.norm(Vector_fj_1 - Vector_fj_2)
                            if Distance1 <= Threshold:
                                Neigborhood_fi_List.append(Xj)
                            if Distance2 <= Threshold:
                                Neigborhood_fj_List.append(Xj)
                        T = list(set(Neigborhood_fi_List) & set(Neigborhood_fj_List))
                        Temp = float(
                            Neigborhood_fi_List.__len__() * Neigborhood_fj_List.__len__() / (DataLength * T.__len__()))
                        Mid += math.log2(Temp)
                    Result = float((-1 / DataLength) * Mid)
                    Redundancuy_Matrix[fi, fj] = Result
        return Redundancuy_Matrix

    # 选择最优特征
    def ChooseTheBestFeature(self, RemainingFeatures, selecedFeatures, LabelList, RelevanceMatrix,
                             RedundancyMatrix, NeighborhoodMatrix_Feature, Threshold):
        Length_SelectedFeatures = len(selecedFeatures)
        if Length_SelectedFeatures == 0:
            # 初始阶段，直接按相关性最大选择
            Temp_List = []
            for row in RemainingFeatures:
                T = sum(RelevanceMatrix[row, :])
                Temp_List.append(T)
            Result = RemainingFeatures[Temp_List.index(max(Temp_List))]
            Score = max(Temp_List)  # 记录该特征的得分
            return Result, Score
        else:
            # 有已选特征时，综合考虑相关性、冗余性、交互性
            Result_List = []
            for f in RemainingFeatures:
                Relevance_Value = sum(RelevanceMatrix[f, :])
                T2 = 0
                for s in selecedFeatures:
                    if RedundancyMatrix[f, s] == 0:
                        T2 += RedundancyMatrix[s, f]
                    else:
                        T2 += RedundancyMatrix[f, s]
                Redundancy_Value = T2
                Interactive_Value = self.ComputeInteractiveBetween_fi_and_S(f, selecedFeatures, LabelList,
                                                                            RelevanceMatrix, RedundancyMatrix,
                                                                            NeighborhoodMatrix_Feature, Threshold)
                # 计算当前特征的评分
                Score = Relevance_Value - Redundancy_Value / len(selecedFeatures) + Interactive_Value
                # 每个特征的重要度由以下三部分决定：
                # 相关性 (Relevance_Value)：特征与目标列的关系，越高越重要。
                # 冗余性 (Redundancy_Value)：特征与已选特征的重复程度，越低越重要。
                # 交互性 (Interactive_Value)：特征与已选特征的协同作用，越高越重要。
                Result_List.append(Score)
            # 找到得分最高的特征
            Best_Index = Result_List.index(max(Result_List))
            Result = RemainingFeatures[Best_Index]
            Score = Result_List[Best_Index]
            return Result, Score



    def FeatureSelectionMarkerEnhancement(self, label, weight_factor):
        self.target_column = label
        self.label_length = len(label)
        all_length = len(self.get_columns())
        # 目标列
        LabelIndexList = [i for i in range(all_length - self.label_length, all_length)]
        # 标记增强 0~1
        self.labellist = self.ConstructLiInstanceList(LabelIndexList, weight_factor)

    # 原始特征集合按重要度排序
    def RankFeaturesBasedOnLabelDistributionUsingNeigborhoodMutualInformation(self, omega):
        all_length = len(self.get_columns())
        # 特征列
        ConditionalAttributes = [i for i in range(all_length - self.label_length)]
        # 目标列
        LabelIndexList = [i for i in range(all_length - self.label_length, all_length)]
        # 获取标记权重
        Label_Weigth_List = self.Get_Label_Weight(self.label_length, omega)
        # 获取领域阈值
        Threshold = self.GetNeigborhoodThreshold(ConditionalAttributes, omega)
        # 每个实例下的领域集合的矩阵
        NeighborhoodMatrix_Attri = self.ConstructNeighborhoodMatrix_Feature(ConditionalAttributes, Threshold)
        # 构建依赖矩阵
        RelevanceMatrix = self.ConstructRelevanceMatrix(ConditionalAttributes, LabelIndexList, Label_Weigth_List,
                                                        NeighborhoodMatrix_Attri, self.labellist, Threshold)
        # 构建冗余矩阵
        RedundancyMatrix = self.ConstructRedundancyMatrix(ConditionalAttributes, NeighborhoodMatrix_Attri,
                                                          Threshold)
        Selected_Features = []
        Feature_Importance = {}  # 用于存储特征的重要度
        Judge_Criterion = len(ConditionalAttributes)

        while len(Selected_Features) < Judge_Criterion:
            # 挑选出最优解和对应的重要度
            The_Best_Feature, Score = self.ChooseTheBestFeature(ConditionalAttributes, Selected_Features,
                                                                LabelIndexList,
                                                                RelevanceMatrix, RedundancyMatrix,
                                                                NeighborhoodMatrix_Attri,
                                                                Threshold)
            Selected_Features.append(The_Best_Feature)
            Feature_Importance[The_Best_Feature] = float(Score)  # 记录特征及其重要度
            ConditionalAttributes.remove(The_Best_Feature)

        # 返回选择的特征以及对应的重要度
        return Feature_Importance


if __name__ == '__main__':
    file_path = '1.csv'
    feature_selection = FeatureSelectionAlgorithm(file_path=file_path)
    feature_selection.loadfile()
    response = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    all_columns = feature_selection.get_columns()
    feature_selection.exchange(response)
    feature_selection.normalization()

    parameter_omega = 0.66
    label_length = len(response)
    feature_selection.FeatureSelectionMarkerEnhancement(response)
    list2 = feature_selection.RankFeaturesBasedOnLabelDistributionUsingNeigborhoodMutualInformation(parameter_omega)
    print(list2)


# 这个代码实现了一个特征选择算法的框架，利用邻域计算和标记权重来评估特征的重要性，并基于冗余和相关性来进行特征筛选。特征选择算法中的大部分功能主要集中在以下几个方面：
