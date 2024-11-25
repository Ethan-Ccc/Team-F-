import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_excel("/Users/zhangshichen/Desktop/项目/统计分析表数据源.xlsx")# 替换为实际文件路径

# 定义特征和目标变量
X = data[["Growth Temperature (°C)", "Field Precipitation (mm)", "Sunlight (hours/day)",
          "Soil pH", "Organic Matter (%)", "Plant Height (cm)",
          "Leaf Area Index", "Biomass (kg/ha)"]]  # 替换为实际列名
y = data["Yield (kg/ha)"]  # 替换为实际目标变量列名

# 数据预处理：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建神经网络
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # 输入层：根据特征数自动调整
    layers.Dense(64, activation='relu'),  # 隐藏层1
    layers.Dense(32, activation='relu'),  # 隐藏层2
    layers.Dense(1)  # 输出层：预测产量
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练模型
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32)

# 测试集评估
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")