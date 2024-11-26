import os
from copy import deepcopy
import numpy as np
import json
import pandas as pd
from typing import Optional, Dict, Any
from flask import jsonify, make_response
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from pandas import DataFrame


class DataCleaner:
    def __init__(self, file_path=None, *, file=None):
        self.file_path = file_path
        self.file = file
        self.data: Optional[DataFrame] = None
        self.cleaned_data: Optional[DataFrame] = None


    def load_data(self):
        # 加载数据的方法，支持CSV或XLSX格式 文件读取或则是路径
        if self.file_path:
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
                self.cleaned_data = deepcopy(self.data)
            elif self.file_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.file_path)
                self.cleaned_data = deepcopy(self.data)
            else:
                raise FileNotFoundError('File type not supported')
        elif self.file:
            if self.file.filename.endswith('.csv'):
                self.data = pd.read_csv(self.file)
                self.cleaned_data = deepcopy(self.data)
            elif self.file.filename.endswith('.xlsx'):
                self.data = pd.read_excel(self.file)
                self.cleaned_data = deepcopy(self.data)
            else:
                raise TypeError('File type not supported')
        else:
            raise FileNotFoundError('File type not supported')

    def pre_view(self):
        if self.file.filename.endswith('.csv'):
            data = pd.read_csv(self.file, nrows=10)
        elif self.file.filename.endswith('.xlsx'):
            data = pd.read_excel(self.file, nrows=10)
        else:
            raise TypeError('File type not supported')
        columns = data.columns
        dict_ = {'columns': list(columns)}
        # 替换掉 DataFrame 中的 NaN 值为 None
        data = data.replace({np.nan: None})
        for column in columns:
            dict_[column] = list(data[column])
            # 使用 make_response 和 json.dumps 来自定义返回
        response = make_response(json.dumps(dict_, ensure_ascii=False))  # 确保中文字符不会被转义
        response.mimetype = 'application/json'  # 指定响应类型为 JSON
        return response

    def get_missing_info(self):
        missing_info = self.cleaned_data.isnull().sum()
        missing_columns = missing_info[missing_info > 0].to_dict()

        # 获取缺失值的列以及数量
        return missing_columns

    def get_columns(self):
        return list(self.cleaned_data.columns)

    # 去重
    def clean_duplicates(self, columns=None):
        original_count = self.cleaned_data.shape[0]
        self.cleaned_data = self.cleaned_data.drop_duplicates(subset=columns)
        new_count = self.cleaned_data.shape[0]
        # print(original_count, new_count)
        return original_count, new_count

    def delete_duplicates(self, columns=None):
        self.cleaned_data = self.cleaned_data

    # 异常值处理  Z-score为标准分数，测量数据点和平均值的距离，若A与平均值相差2个标准差，Z-score为2。当把Z-score=3作为阈值去剔除异常点时，便相当于3sigma。
    def handle_outliers(self, action='delete'):
        original_count = self.cleaned_data.shape[0]
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns

        # 计算每列的 Z-score
        z_scores = np.abs((self.cleaned_data[numeric_cols] - self.cleaned_data[numeric_cols].mean()) /
                          self.cleaned_data[numeric_cols].std())

        # 标记 Z-score 大于 3 的为异常值
        outliers = (z_scores > 3)

        # 根据不同的处理方式处理异常值
        if action == 'delete':
            # 删除包含异常值的行
            self.cleaned_data = self.cleaned_data[~outliers.any(axis=1)]

        elif action == 'replace_zero':
            # 将异常值替换为 0
            for col in numeric_cols:
                self.cleaned_data.loc[outliers[col], col] = 0

        elif action == 'replace_mean':
            # 将异常值替换为该列的均值
            for col in numeric_cols:
                self.cleaned_data[col] = self.cleaned_data[col].astype(float)
                mean_value = self.cleaned_data[col].mean()
                self.cleaned_data.loc[outliers[col], col] = mean_value

        elif action == 'replace_mode':
            # 将异常值替换为该列的众数
            for col in numeric_cols:
                self.cleaned_data[col] = self.cleaned_data[col].astype(float)
                if self.cleaned_data[col].isnull().all():  # 检查该列是否全为空
                    print(f"列 {col} 全为空，无法计算众数")
                    continue  # 跳过此列的处理
                mode_value = self.cleaned_data[col].mode()[0]
                # self.cleaned_data.loc[outliers[col], col] = mode_value
                # 检查 outliers[col] 是否存在且形状匹配
                if col in outliers and outliers[col].index.equals(self.cleaned_data.index):
                    self.cleaned_data.loc[outliers[col], col] = mode_value
                else:
                    print(f"列 {col} 的异常值索引不匹配或不存在")
        # 返回处理前后的行数
        new_count = self.cleaned_data.shape[0]
        return original_count, new_count

    def delete_missing_values(self, columns):
        self.cleaned_data = self.cleaned_data.dropna(subset=columns)

    # 缺失值处理
    # 数值型
    def handle_numeric_missing_values(self, numeric_columns=None):
        if numeric_columns is None:
            numeric_columns = self.cleaned_data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if self.cleaned_data[column].isnull().all():  # 检查该列是否全为空
                raise ValueError(f'列 {column} 全为空，无法进行Iterative Imputer')

            imputer = IterativeImputer(max_iter=10, random_state=0)
            # 将 Series 转换为 DataFrame
            self.cleaned_data[[column]] = imputer.fit_transform(self.cleaned_data[[column]])

    # 分类型
    def handle_categorical_missing_values(self, categorical_columns=None):
        if categorical_columns is None:
            categorical_columns = self.cleaned_data.select_dtypes(exclude=[np.number]).columns

        for column in categorical_columns:
            if self.cleaned_data[column].isnull().all():  # 检查该列是否全为空
                raise ValueError(f'列 {column} 全为空，无法进行使用Simple Imputer')
            imputer = SimpleImputer(strategy='most_frequent')
            # 将 Series 转换为 DataFrame
            self.cleaned_data[[column]] = imputer.fit_transform(self.cleaned_data[[column]])

    # 对数值型数据，使用了Iterative Imputer，这是一个基于多重插补的迭代算法，通常比简单插补（如均值插补）更准确。
    # 对分类型数据，使用了Simple Imputer，用最频繁值填补缺失项，这是处理类别数据的一种常见方法。

    # 保存清洗后的数据
    def save_cleaned_data(self, output_path):
        full_path = os.path.join(output_path, 'cleaned_data')
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            os.makedirs(full_path)
        output_path = os.path.join(full_path, self.file_path.split('\\')[-1])
        if self.file_path.endswith('.csv'):
            self.cleaned_data.to_csv(output_path, index=False, encoding='utf-8')
        else:
            self.cleaned_data.to_excel(output_path, index=False, engine='openpyxl')


