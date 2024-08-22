import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
data = pd.read_csv(r'D:\WeChat\WeChat Files\wxid_m898ol87nwxa22\FileStorage\File\2024-07\实验三：近地卫星危险预测.csv')

# 检查数据集的列名
print(data.columns)

# 2. 数据预处理
# 检查缺失值
print(data.isnull().sum())

# 填充缺失值或删除缺失值行（视数据情况而定）
# 例如：data = data.dropna()

# 转换分类变量为数值变量
# 例如：data['orbiting_body'] = data['orbiting_body'].astype('category').cat.codes

# 删除非数值特征
data = data.drop(['id', 'name'], axis=1)

# 转换分类变量为数值变量
for column in data.select_dtypes(include=['object']).columns:
    data[column] = data[column].astype('category').cat.codes

# 选择特征和目标变量
X = data.drop(['hazardous'], axis=1)
y = data['hazardous']

# 3. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构建和训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. 模型评估
y_pred = rf_model.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 特征重要性
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# 可视化特征重要性
plt.figure(figsize=(15, 5))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.show()
