# 運動彩券賽事預測系統

基於 Hugging Face Transformers 的智能運動賽事結果預測系統，專門用於運動彩券分析和預測。

## 🏆 系統特色

- **先進的 LLM 技術**：使用 BERT/RoBERTa 等預訓練語言模型
- **多運動支援**：支援足球、籃球、棒球等多種運動項目
- **智能特徵工程**：整合球隊狀態、歷史對戰、球員數據、環境因素等
- **高精度預測**：結合深度學習和運動領域知識
- **靈活配置**：支援多種預訓練模型和參數調整
- **完整工作流程**：從數據收集到模型訓練再到預測的一站式解決方案

## 📁 項目結構

```
sports_betting_llm/
├── sports_prediction_llm.py    # 核心預測模型
├── data_collector.py           # 數據收集器
├── config.py                   # 配置管理
├── run_sports_prediction.py    # 主運行腳本
├── requirements.txt            # 依賴需求
├── README.md                   # 說明文件
├── data/                       # 數據目錄
├── models/                     # 模型保存目錄
└── outputs/                    # 輸出結果目錄
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 運行演示

```bash
# 完整演示流程（推薦首次使用）
python run_sports_prediction.py --mode demo

# 指定運動類型和樣本數
python run_sports_prediction.py --mode demo --sport football --samples 1000
```

### 3. 訓練模型

```bash
# 訓練足球預測模型
python run_sports_prediction.py --mode train --sport football --samples 2000 --epochs 5

# 使用真實數據收集器
python run_sports_prediction.py --mode train --real-data --samples 1000
```

### 4. 進行預測

```bash
# 預測示例比賽
python run_sports_prediction.py --mode predict

# 預測自定義比賽
python run_sports_prediction.py --mode predict --match-text "曼城主場迎戰利物浦，雙方近期狀態良好..."

# 互動預測模式
python run_sports_prediction.py --mode interactive
```

## 🎯 使用方法

### 命令行參數

| 參數           | 選項                                           | 默認值          | 說明               |
| -------------- | ---------------------------------------------- | --------------- | ------------------ |
| `--mode`       | train/predict/interactive/demo                 | demo            | 運行模式           |
| `--model`      | chinese-roberta/chinese-bert/multilingual-bert | chinese-roberta | 預訓練模型         |
| `--sport`      | football/basketball/baseball                   | football        | 運動類型           |
| `--samples`    | 整數                                           | 2000            | 訓練樣本數量       |
| `--epochs`     | 整數                                           | 3               | 訓練週期           |
| `--batch-size` | 整數                                           | 16              | 批次大小           |
| `--real-data`  | -                                              | False           | 使用真實數據收集器 |
| `--model-path` | 路徑                                           | -               | 模型保存/載入路徑  |
| `--match-text` | 文字                                           | -               | 自定義比賽描述     |

### 程式化使用

```python
from sports_prediction_llm import SportsPredictor
from data_collector import SportsDataCollector

# 初始化預測器
predictor = SportsPredictor(model_name='hfl/chinese-roberta-wwm-ext')

# 收集數據
collector = SportsDataCollector()
training_data = collector.prepare_training_data(sport='football', num_samples=1000)

# 訓練模型
data_splits = predictor.prepare_data(training_data)
train_dataset, val_dataset, test_dataset = predictor.create_datasets(data_splits)
trainer = predictor.train(train_dataset, val_dataset, num_epochs=3)

# 進行預測
match_description = "曼城主場迎戰利物浦，雙方近期狀態良好..."
prediction = predictor.predict_match(match_description)
print(f"預測結果: {prediction['prediction']}")
print(f"信心度: {prediction['confidence']:.3f}")
```

## 🧠 模型架構

### 支援的預訓練模型

1. **chinese-roberta** (`hfl/chinese-roberta-wwm-ext`)

   - 專為中文優化的 RoBERTa 模型
   - 推薦用於中文運動數據

2. **chinese-bert** (`bert-base-chinese`)

   - Google 的中文 BERT 模型
   - 穩定可靠的選擇

3. **chinese-macbert** (`hfl/chinese-macbert-base`)

   - 改進的中文 BERT 模型
   - 更好的中文理解能力

4. **multilingual-bert** (`bert-base-multilingual-cased`)
   - 多語言 BERT 模型
   - 支援多語言混合數據

### 預測類別

- **客勝** (0)：客隊獲勝
- **平局** (1)：雙方平手
- **主勝** (2)：主隊獲勝

## 📊 數據特徵

### 球隊特徵

- 近期狀態評分
- 聯賽排名
- 主場優勢
- 歷史對戰記錄

### 球員特徵

- 主力球員可用性
- 傷病情況
- 球員狀態評級

### 環境特徵

- 天氣條件
- 比賽場地
- 氣溫影響

### 市場特徵

- 博彩賠率
- 市場預期
- 投注趨勢

## 🔧 配置選項

### 模型配置 (`config.py`)

```python
@dataclass
class ModelConfig:
    model_name: str = 'hfl/chinese-roberta-wwm-ext'
    num_labels: int = 3
    max_length: int = 512
    dropout_rate: float = 0.3
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
```

### 數據配置

```python
@dataclass
class DataConfig:
    num_samples: int = 2000
    sports_types: List[str] = ['football', 'basketball', 'baseball']
    data_dir: str = './data'
    model_dir: str = './models'
```

### 環境變量支援

```bash
export MODEL_NAME="hfl/chinese-roberta-wwm-ext"
export BATCH_SIZE=32
export LEARNING_RATE=1e-5
export NUM_EPOCHS=10
export NUM_SAMPLES=5000
```

## 📈 性能評估

系統會自動計算以下評估指標：

- **準確率 (Accuracy)**：正確預測的比例
- **F1 分數 (F1-Score)**：精確率和召回率的調和平均
- **分類報告**：各類別的詳細性能指標
- **混淆矩陣**：預測結果的詳細分布

## 🎮 使用示例

### 1. 演示模式

```bash
python run_sports_prediction.py --mode demo --sport football --samples 1000 --epochs 3
```

輸出：

```
🏆 運動彩券賽事預測系統
============================================================
🔄 收集 football 訓練數據 (1000 樣本)...
✅ 成功收集 1000 條訓練數據
📊 結果分布:
   主勝: 420 (42.0%)
   客勝: 350 (35.0%)
   平局: 230 (23.0%)

🚀 開始訓練運動預測模型...
📚 數據集大小:
   訓練集: 720
   驗證集: 80
   測試集: 200

📈 評估模型性能...
測試準確率: 0.7650
F1分數: 0.7580

🔮 進行示例預測...
預測第 1 場比賽:
   結果: 主勝 (信心度: 0.832)
```

### 2. 互動模式

```bash
python run_sports_prediction.py --mode interactive
```

```
🎮 進入互動預測模式
輸入 'quit' 或 'exit' 退出
============================================================

請輸入比賽描述: 曼城主場迎戰阿森納，曼城近期狀態極佳，阿森納有3名主力受傷

🔮 進行比賽預測...
============================================================
📋 比賽描述:
曼城主場迎戰阿森納，曼城近期狀態極佳，阿森納有3名主力受傷

🎯 預測結果:
   預測結果: 主勝
   信心度: 0.876

📊 各結果機率:
   客勝: 0.067 (6.7%)
   平局: 0.057 (5.7%)
   主勝: 0.876 (87.6%)
============================================================
```

## 🔬 進階功能

### 1. 自定義數據收集

```python
from data_collector import SportsDataCollector

collector = SportsDataCollector()

# 添加新的球隊數據
new_team_data = {
    'strength': 85,
    'form': [3, 1, 3, 3, 0],
    'home_advantage': 80
}
collector.real_teams_data['football']['premier_league']['新球隊'] = new_team_data

# 收集特定聯賽數據
data = collector.prepare_training_data(sport='football', league='premier_league')
```

### 2. 模型微調

```python
from transformers import TrainingArguments

# 自定義訓練參數
training_args = TrainingArguments(
    output_dir='./custom_model',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    learning_rate=1e-5,
    warmup_steps=1000,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True
)
```

### 3. 批量預測

```python
match_descriptions = [
    "曼城 vs 利物浦，主場作戰，近期狀態良好",
    "阿森納 vs 切爾西，客場比賽，有傷病困擾",
    "曼聯 vs 熱刺，中立場地，雙方實力相當"
]

predictions = pipeline.batch_predict(match_descriptions)
for pred in predictions:
    print(f"比賽 {pred['match_id']}: {pred['prediction']} (信心度: {pred['confidence']:.3f})")
```

## ⚠️ 注意事項

1. **負責任博彩**：本系統僅供學習和研究使用，請理性對待運動博彩
2. **數據準確性**：模擬數據僅供演示，實際使用需要真實可靠的數據源
3. **預測限制**：任何預測系統都無法保證 100% 準確，運動比賽存在很多不確定因素
4. **法律合規**：請確保在您所在地區的法律框架內使用本系統

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

1. Fork 本項目
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

本項目採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

- Hugging Face Transformers 團隊
- 中文預訓練模型提供者
- 開源機器學習社群

## 📞 聯絡方式

如有問題或建議，請通過以下方式聯絡：

- 提交 GitHub Issue
- 電子郵件：[您的郵箱]
- 項目主頁：[項目網址]

---

**免責聲明**：本系統僅用於教育和研究目的。運動賽事預測涉及多種不確定因素，任何預測結果都不應作為投注決策的唯一依據。使用者需要自行承擔使用本系統的所有風險和後果。
