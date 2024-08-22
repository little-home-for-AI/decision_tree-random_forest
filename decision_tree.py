import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = '实验三：近地卫星危险预测.csv'
data = pd.read_csv(file_path)

# 查看数据基本信息
print(data.head())
print(data.info())
print(data.describe())

# 检查缺失值
print(data.isnull().sum())

### 详细解释：

# 1. **`data.isnull()`**：
#    - 这一部分代码会返回一个与原数据集 `data` 形状相同的布尔型DataFrame，其中每个元素是布尔值（`True` 或 `False`）。如果某个位置的值是缺失值（即 `NaN`），则对应的位置为 `True`，否则为 `False`。
#
# 2. **`.sum()`**：
#    - 这一部分代码会对每一列的布尔值进行求和运算。因为布尔值 `True` 在计算时等同于1，而 `False` 等同于0，所以 `sum()` 会计算每列中缺失值的总数。
#
# 3. **`print()`**：
#    - 最后，`print()`函数会将每一列的缺失值数量打印出来，以便用户查看和分析。
#
# ### 举例说明：
#
# 假设数据集 `data` 如下：
#
# | A     | B     | C     |
# |-------|-------|-------|
# | 1     | NaN   | 3     |
# | NaN   | 2     | NaN   |
# | 4     | NaN   | 6     |
# | 7     | 8     | 9     |
#
# 调用 `data.isnull()` 后的结果是：
#
# | A     | B     | C     |
# |-------|-------|-------|
# | False | True  | False |
# | True  | False | True  |
# | False | True  | False |
# | False | False | False |
#
# 再调用 `.sum()` 后的结果是：
#
# | A | B | C |
# |---|---|---|
# | 1 | 2 | 1 |
#
# 这表示：
# - 列 A 有1个缺失值
# - 列 B 有2个缺失值
# - 列 C 有1个缺失值
#
# 最终通过 `print()` 输出结果：
#
# ```
# A    1
# B    2
# C    1
# dtype: int64
# ```


# 对分类特征进行编码（例如，布尔特征和行星名称）
data['hazardous'] = data['hazardous'].astype(int)  # 将布尔值转换为整数
data = pd.get_dummies(data, columns=['orbiting_body'])

# 分离特征和标签
X = data.drop(columns=['id', 'name', 'hazardous'])
y = data['hazardous']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 训练决策树模型
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 预测
y_pred_dt = dt_model.predict(X_test)

# 模型评估
print("决策树模型分类报告：")
print(classification_report(y_test, y_pred_dt))
print("决策树模型混淆矩阵：")
print(confusion_matrix(y_test, y_pred_dt))

# 决策树模型的特征重要性
feature_importances_dt = pd.Series(dt_model.feature_importances_, index=X.columns)
feature_importances_dt.sort_values().plot(kind='barh')
plt.title("The importance of each feature")
plt.show()
