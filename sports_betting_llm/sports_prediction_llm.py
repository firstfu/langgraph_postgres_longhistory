"""
=============================================================================
基於 Hugging Face Transformers 微調的運動彩券賽事預測系統
功能：使用預訓練語言模型進行運動賽事結果預測
技術：BERT/RoBERTa + 微調 + 分類預測
數據：球隊歷史數據、球員統計、比賽條件等
應用：足球、籃球、棒球等多種運動項目預測
=============================================================================
"""

import warnings

import matplotlib

matplotlib.use('Agg')
import json
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

try:
    from transformers import EarlyStoppingCallback
except ImportError:
    from transformers.trainer_callback import EarlyStoppingCallback

warnings.filterwarnings('ignore')

# 設定隨機種子確保結果可重現
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SportsDataGenerator:
    """運動賽事數據生成器"""

    def __init__(self):
        self.teams = {
            'football': ['曼城', '利物浦', '切爾西', '阿森納', '曼聯', '熱刺', '紐卡斯爾', '布萊頓'],
            'basketball': ['湖人', '勇士', '塞爾提克', '熱火', '公牛', '馬刺', '快艇', '太陽'],
            'baseball': ['洋基', '道奇', '紅襪', '巨人', '老虎', '天使', '遊騎兵', '太空人']
        }

        self.player_stats = {
            'excellent': {'form': 9, 'injury': 0, 'performance': 'excellent'},
            'good': {'form': 7, 'injury': 0, 'performance': 'good'},
            'average': {'form': 5, 'injury': 1, 'performance': 'average'},
            'poor': {'form': 3, 'injury': 2, 'performance': 'poor'}
        }

        self.weather_conditions = ['晴天', '陰天', '小雨', '大雨', '雪', '強風']
        self.venues = ['主場', '客場', '中立場地']

    def generate_match_context(self, sport='football', num_samples=1000):
        """生成比賽情境數據"""
        data = []

        for _ in range(num_samples):
            teams = random.sample(self.teams[sport], 2)
            home_team, away_team = teams[0], teams[1]

            # 隨機生成球隊狀態
            home_form = random.randint(1, 10)
            away_form = random.randint(1, 10)

            # 隨機生成其他因素
            weather = random.choice(self.weather_conditions)
            venue = random.choice(self.venues)

            # 球員狀態
            home_player_status = random.choice(list(self.player_stats.keys()))
            away_player_status = random.choice(list(self.player_stats.keys()))

            # 歷史對戰記錄
            head_to_head = random.randint(0, 10)  # 過去10場比賽中主隊勝場數

            # 生成文本描述
            match_description = self._create_match_description(
                home_team, away_team, home_form, away_form,
                weather, venue, home_player_status, away_player_status, head_to_head
            )

            # 決定比賽結果（基於各種因素的邏輯）
            result = self._determine_result(
                home_form, away_form, venue,
                self.player_stats[home_player_status]['form'],
                self.player_stats[away_player_status]['form'],
                head_to_head
            )

            data.append({
                'text': match_description,
                'home_team': home_team,
                'away_team': away_team,
                'result': result,  # 0: 客勝, 1: 平局, 2: 主勝
                'home_form': home_form,
                'away_form': away_form,
                'weather': weather,
                'venue': venue
            })

        return pd.DataFrame(data)

    def _create_match_description(self, home_team, away_team, home_form, away_form,
                                weather, venue, home_player_status, away_player_status, h2h):
        """創建比賽描述文本"""

        descriptions = [
            f"本場比賽由{home_team}主場迎戰{away_team}。",
            f"{home_team}近期狀態為{home_form}/10，{away_team}近期狀態為{away_form}/10。",
            f"比賽當天天氣條件為{weather}，比賽場地為{venue}。",
            f"{home_team}主力球員狀態{home_player_status}，{away_team}主力球員狀態{away_player_status}。",
            f"雙方過去10次交手，{home_team}獲勝{h2h}場。"
        ]

        # 添加額外的分析資訊
        if home_form > away_form:
            descriptions.append(f"{home_team}近期表現較佳，具有心理優勢。")
        elif away_form > home_form:
            descriptions.append(f"{away_team}近期表現較佳，狀態正盛。")
        else:
            descriptions.append("雙方近期狀態相當，實力接近。")

        if venue == '主場':
            descriptions.append(f"{home_team}享有主場優勢，球迷支持度高。")

        return " ".join(descriptions)

    def _determine_result(self, home_form, away_form, venue, home_player_form, away_player_form, h2h):
        """基於邏輯決定比賽結果"""

        # 計算主隊優勢分數
        home_advantage = 0

        # 近期狀態影響
        home_advantage += (home_form - away_form) * 0.3

        # 主場優勢
        if venue == '主場':
            home_advantage += 1.5
        elif venue == '客場':
            home_advantage -= 1.5

        # 球員狀態影響
        home_advantage += (home_player_form - away_player_form) * 0.2

        # 歷史對戰影響
        home_advantage += (h2h - 5) * 0.1

        # 加入隨機因素
        random_factor = random.uniform(-1, 1)
        home_advantage += random_factor

        # 決定結果
        if home_advantage > 1:
            return 2  # 主勝
        elif home_advantage < -1:
            return 0  # 客勝
        else:
            return 1  # 平局

class SportsDataset(Dataset):
    """運動預測數據集"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SportsPredictor:
    """運動賽事預測器"""

    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_encoder = LabelEncoder()

    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        """準備訓練數據"""

        # 編碼標籤
        df['encoded_labels'] = self.label_encoder.fit_transform(df['result'])

        # 分割數據
        X = df['text'].values
        y = df['encoded_labels'].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )

        return {
            'train': {'texts': X_train, 'labels': y_train},
            'val': {'texts': X_val, 'labels': y_val},
            'test': {'texts': X_test, 'labels': y_test}
        }

    def create_datasets(self, data_splits):
        """創建數據集"""

        train_dataset = SportsDataset(
            data_splits['train']['texts'],
            data_splits['train']['labels'],
            self.tokenizer
        )

        val_dataset = SportsDataset(
            data_splits['val']['texts'],
            data_splits['val']['labels'],
            self.tokenizer
        )

        test_dataset = SportsDataset(
            data_splits['test']['texts'],
            data_splits['test']['labels'],
            self.tokenizer
        )

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, output_dir='./sports_model',
              num_epochs=5, batch_size=16, learning_rate=2e-5):
        """訓練模型"""

        # 初始化模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        # 設定訓練參數
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy="steps",  # 修正參數名稱
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # 禁用wandb等報告
            learning_rate=learning_rate,
        )

        # 初始化訓練器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # 開始訓練
        print("開始訓練運動預測模型...")
        trainer.train()

        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        return trainer

    def evaluate(self, test_dataset):
        """評估模型"""
        if self.model is None:
            raise ValueError("模型尚未訓練或載入")

        # 創建數據加載器
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        self.model.eval()
        predictions = []
        true_labels = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                predicted = torch.argmax(logits, dim=-1)

                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # 計算評估指標
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"測試準確率: {accuracy:.4f}")
        print(f"F1分數: {f1:.4f}")

        # 分類報告
        target_names = ['客勝', '平局', '主勝']
        report = classification_report(true_labels, predictions, target_names=target_names)
        print("分類報告:")
        print(report)

        return accuracy, f1, report

    def predict_match(self, match_description, threshold=0.5):
        """預測單場比賽結果"""
        if self.model is None:
            raise ValueError("模型尚未訓練或載入")

        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # 編碼輸入文本
        encoding = self.tokenizer(
            match_description,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities).item()

        # 解碼預測結果
        result_mapping = {0: '客勝', 1: '平局', 2: '主勝'}
        predicted_result = result_mapping[predicted_class]

        return {
            'prediction': predicted_result,
            'confidence': confidence,
            'probabilities': {
                '客勝': probabilities[0][0].item(),
                '平局': probabilities[0][1].item(),
                '主勝': probabilities[0][2].item()
            }
        }

    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未訓練")

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # 保存標籤編碼器
        import joblib
        joblib.dump(self.label_encoder, f"{path}/label_encoder.pkl")

    def load_model(self, path):
        """載入模型"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # 載入標籤編碼器
        import joblib
        self.label_encoder = joblib.load(f"{path}/label_encoder.pkl")

def main():
    """主函數"""
    print("=== 運動彩券賽事預測 LLM 系統 ===")

    # 1. 生成訓練數據
    print("1. 生成訓練數據...")
    data_generator = SportsDataGenerator()
    df = data_generator.generate_match_context(sport='football', num_samples=2000)

    print(f"生成了 {len(df)} 條訓練數據")
    print("數據樣本:")
    print(df.head())

    # 2. 初始化預測器
    print("\n2. 初始化預測器...")
    predictor = SportsPredictor()

    # 3. 準備數據
    print("3. 準備訓練數據...")
    data_splits = predictor.prepare_data(df)
    train_dataset, val_dataset, test_dataset = predictor.create_datasets(data_splits)

    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")
    print(f"測試集大小: {len(test_dataset)}")

    # 4. 訓練模型
    print("\n4. 開始訓練模型...")
    trainer = predictor.train(train_dataset, val_dataset, num_epochs=3)

    # 5. 評估模型
    print("\n5. 評估模型...")
    accuracy, f1, report = predictor.evaluate(test_dataset)

    # 6. 測試預測功能
    print("\n6. 測試預測功能...")
    test_match = "本場比賽由曼城主場迎戰利物浦。曼城近期狀態為8/10，利物浦近期狀態為7/10。比賽當天天氣條件為晴天，比賽場地為主場。曼城主力球員狀態excellent，利物浦主力球員狀態good。雙方過去10次交手，曼城獲勝6場。曼城近期表現較佳，具有心理優勢。曼城享有主場優勢，球迷支持度高。"

    prediction = predictor.predict_match(test_match)
    print(f"預測結果: {prediction['prediction']}")
    print(f"信心度: {prediction['confidence']:.4f}")
    print("各結果機率:")
    for result, prob in prediction['probabilities'].items():
        print(f"  {result}: {prob:.4f}")

    # 7. 保存模型
    print("\n7. 保存模型...")
    predictor.save_model('./sports_prediction_model')
    print("模型已保存到 ./sports_prediction_model")

if __name__ == "__main__":
    main()