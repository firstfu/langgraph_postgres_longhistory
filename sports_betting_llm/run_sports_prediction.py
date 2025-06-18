"""
=============================================================================
運動彩券賽事預測系統 - 主要運行腳本
功能：整合數據收集、模型訓練、預測等功能的完整工作流程
使用：python run_sports_prediction.py
=============================================================================
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(__file__))

from config import SUPPORTED_MODELS, config
from data_collector import SportsDataCollector
# 導入自定義模塊
from sports_prediction_llm import SportsDataGenerator, SportsPredictor

warnings.filterwarnings('ignore')

class SportsPredictionPipeline:
    """運動預測完整流水線"""

    def __init__(self, model_name: str = None, use_real_data: bool = False):
        """
        初始化預測流水線

        Args:
            model_name: 預訓練模型名稱
            use_real_data: 是否使用真實數據收集器
        """
        # 設定模型
        if model_name and model_name in SUPPORTED_MODELS:
            config.model.model_name = SUPPORTED_MODELS[model_name]
        elif model_name:
            config.model.model_name = model_name

        # 初始化組件
        self.predictor = SportsPredictor(
            model_name=config.model.model_name,
            num_labels=config.model.num_labels
        )

        self.use_real_data = use_real_data
        if use_real_data:
            self.data_collector = SportsDataCollector()
        else:
            self.data_generator = SportsDataGenerator()

        self.training_data = None
        self.model_trained = False

    def collect_training_data(self, sport: str = 'football', num_samples: int = 2000):
        """收集訓練數據"""

        print(f"🔄 收集 {sport} 訓練數據 ({num_samples} 樣本)...")

        if self.use_real_data:
            # 使用真實數據收集器
            self.training_data = self.data_collector.prepare_training_data(
                sport=sport,
                num_samples=num_samples
            )
        else:
            # 使用模擬數據生成器
            df = self.data_generator.generate_match_context(
                sport=sport,
                num_samples=num_samples
            )
            self.training_data = df

        print(f"✅ 成功收集 {len(self.training_data)} 條訓練數據")

        # 顯示數據分布
        if 'result' in self.training_data.columns:
            result_counts = self.training_data['result'].value_counts()
            print("📊 結果分布:")
            result_mapping = {0: '客勝', 1: '平局', 2: '主勝'}
            for result, count in result_counts.items():
                label = result_mapping.get(result, f'未知({result})')
                print(f"   {label}: {count} ({count/len(self.training_data)*100:.1f}%)")

        return self.training_data

    def train_model(self, num_epochs: int = None, batch_size: int = None):
        """訓練模型"""

        if self.training_data is None:
            raise ValueError("請先收集訓練數據")

        print("🚀 開始訓練運動預測模型...")

        # 使用配置或傳入的參數
        epochs = num_epochs or config.model.num_epochs
        batch = batch_size or config.model.batch_size

        # 準備數據
        data_splits = self.predictor.prepare_data(self.training_data)
        train_dataset, val_dataset, test_dataset = self.predictor.create_datasets(data_splits)

        print(f"📚 數據集大小:")
        print(f"   訓練集: {len(train_dataset)}")
        print(f"   驗證集: {len(val_dataset)}")
        print(f"   測試集: {len(test_dataset)}")

        # 訓練模型
        trainer = self.predictor.train(
            train_dataset,
            val_dataset,
            num_epochs=epochs,
            batch_size=batch
        )

        # 評估模型
        print("📈 評估模型性能...")
        accuracy, f1, report = self.predictor.evaluate(test_dataset)

        self.model_trained = True
        print("✅ 模型訓練完成!")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }

    def predict_match(self, match_description: str, show_details: bool = True):
        """預測單場比賽"""

        if not self.model_trained:
            raise ValueError("請先訓練模型")

        print("🔮 進行比賽預測...")
        prediction = self.predictor.predict_match(match_description)

        if show_details:
            print("\n" + "="*60)
            print("📋 比賽描述:")
            print(match_description)
            print("\n🎯 預測結果:")
            print(f"   預測結果: {prediction['prediction']}")
            print(f"   信心度: {prediction['confidence']:.3f}")
            print("\n📊 各結果機率:")
            for result, prob in prediction['probabilities'].items():
                print(f"   {result}: {prob:.3f} ({prob*100:.1f}%)")
            print("="*60)

        return prediction

    def batch_predict(self, match_descriptions: list):
        """批量預測多場比賽"""

        print(f"🔮 批量預測 {len(match_descriptions)} 場比賽...")
        predictions = []

        for i, description in enumerate(match_descriptions, 1):
            print(f"\n預測第 {i} 場比賽:")
            try:
                prediction = self.predict_match(description, show_details=False)
                predictions.append({
                    'match_id': i,
                    'description': description,
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    **prediction['probabilities']
                })
                print(f"   結果: {prediction['prediction']} (信心度: {prediction['confidence']:.3f})")
            except Exception as e:
                print(f"   ❌ 預測失敗: {e}")
                predictions.append({
                    'match_id': i,
                    'description': description,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                })

        return predictions

    def save_model(self, model_path: str = None):
        """保存模型"""

        if not self.model_trained:
            raise ValueError("請先訓練模型")

        save_path = model_path or config.get_model_path()
        self.predictor.save_model(save_path)
        print(f"💾 模型已保存到: {save_path}")

    def load_model(self, model_path: str = None):
        """載入模型"""

        load_path = model_path or config.get_model_path()
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型路徑不存在: {load_path}")

        self.predictor.load_model(load_path)
        self.model_trained = True
        print(f"📁 模型已從 {load_path} 載入")

    def interactive_mode(self):
        """互動模式"""

        print("\n" + "="*60)
        print("🎮 進入互動預測模式")
        print("輸入 'quit' 或 'exit' 退出")
        print("="*60)

        while True:
            try:
                # 獲取用戶輸入
                match_input = input("\n請輸入比賽描述: ").strip()

                if match_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 退出互動模式")
                    break

                if not match_input:
                    print("⚠️  請輸入比賽描述")
                    continue

                # 進行預測
                self.predict_match(match_input)

            except KeyboardInterrupt:
                print("\n👋 用戶中斷，退出互動模式")
                break
            except Exception as e:
                print(f"❌ 預測錯誤: {e}")

def create_sample_predictions():
    """創建示例預測"""

    sample_matches = [
        "本場比賽由曼城主場迎戰利物浦。曼城近期狀態為8/10，利物浦近期狀態為7/10。比賽當天天氣條件為晴天，比賽場地為主場。曼城主力球員狀態excellent，利物浦主力球員狀態good。雙方過去10次交手，曼城獲勝6場。曼城近期表現較佳，具有心理優勢。曼城享有主場優勢，球迷支持度高。",

        "本場比賽由湖人主場迎戰勇士。湖人近期狀態為6/10，勇士近期狀態為8/10。比賽當天天氣條件為陰天，比賽場地為主場。湖人主力球員狀態average，勇士主力球員狀態excellent。雙方過去10次交手，湖人獲勝4場。勇士近期表現較佳，狀態正盛。湖人享有主場優勢，球迷支持度高。",

        "本場比賽由阿森納主場迎戰切爾西。阿森納近期狀態為7/10，切爾西近期狀態為5/10。比賽當天天氣條件為小雨，比賽場地為主場。阿森納主力球員狀態good，切爾西主力球員狀態average。雙方過去10次交手，阿森納獲勝5場。阿森納近期表現較佳，具有心理優勢。阿森納享有主場優勢，球迷支持度高。"
    ]

    return sample_matches

def main():
    """主函數"""

    parser = argparse.ArgumentParser(description='運動彩券賽事預測系統')
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive', 'demo'],
                       default='demo', help='運行模式')
    parser.add_argument('--model', choices=list(SUPPORTED_MODELS.keys()) + ['custom'],
                       default='chinese-roberta', help='預訓練模型')
    parser.add_argument('--sport', choices=['football', 'basketball', 'baseball'],
                       default='football', help='運動類型')
    parser.add_argument('--samples', type=int, default=2000, help='訓練樣本數量')
    parser.add_argument('--epochs', type=int, default=3, help='訓練週期')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--real-data', action='store_true', help='使用真實數據收集器')
    parser.add_argument('--model-path', type=str, help='模型保存/載入路徑')
    parser.add_argument('--match-text', type=str, help='要預測的比賽描述')

    args = parser.parse_args()

    print("🏆 運動彩券賽事預測系統")
    print("="*60)
    print(f"運行模式: {args.mode}")
    print(f"使用模型: {args.model}")
    print(f"運動類型: {args.sport}")
    if args.real_data:
        print("數據來源: 真實數據收集器")
    else:
        print("數據來源: 模擬數據生成器")
    print("="*60)

    # 初始化流水線
    pipeline = SportsPredictionPipeline(
        model_name=args.model if args.model != 'custom' else None,
        use_real_data=args.real_data
    )

    try:
        if args.mode == 'train':
            # 訓練模式
            pipeline.collect_training_data(args.sport, args.samples)
            results = pipeline.train_model(args.epochs, args.batch_size)

            # 保存模型
            if args.model_path:
                pipeline.save_model(args.model_path)
            else:
                pipeline.save_model()

            print(f"\n📊 訓練結果:")
            print(f"準確率: {results['accuracy']:.4f}")
            print(f"F1分數: {results['f1_score']:.4f}")

        elif args.mode == 'predict':
            # 預測模式
            if args.model_path:
                pipeline.load_model(args.model_path)
            else:
                # 如果沒有指定模型路徑，嘗試載入默認路徑
                try:
                    pipeline.load_model()
                except FileNotFoundError:
                    print("⚠️  找不到預訓練模型，將先進行訓練...")
                    pipeline.collect_training_data(args.sport, 1000)  # 使用較少樣本快速訓練
                    pipeline.train_model(2, args.batch_size)  # 2個epoch快速訓練

            if args.match_text:
                # 預測指定比賽
                pipeline.predict_match(args.match_text)
            else:
                # 預測示例比賽
                sample_matches = create_sample_predictions()
                pipeline.batch_predict(sample_matches)

        elif args.mode == 'interactive':
            # 互動模式
            if args.model_path:
                pipeline.load_model(args.model_path)
            else:
                try:
                    pipeline.load_model()
                except FileNotFoundError:
                    print("⚠️  找不到預訓練模型，將先進行訓練...")
                    pipeline.collect_training_data(args.sport, 1000)
                    pipeline.train_model(2, args.batch_size)

            pipeline.interactive_mode()

        elif args.mode == 'demo':
            # 演示模式 - 完整流程
            print("🎬 演示完整預測流程...")

            # 1. 收集數據
            pipeline.collect_training_data(args.sport, args.samples)

            # 2. 訓練模型
            results = pipeline.train_model(args.epochs, args.batch_size)

            # 3. 進行預測
            print("\n🔮 進行示例預測...")
            sample_matches = create_sample_predictions()
            predictions = pipeline.batch_predict(sample_matches)

            # 4. 保存結果
            output_file = config.get_output_path(f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'training_results': results,
                    'predictions': predictions,
                    'config': config.to_dict()
                }, f, ensure_ascii=False, indent=2)

            print(f"📁 結果已保存到: {output_file}")

            # 5. 保存模型
            pipeline.save_model()

    except Exception as e:
        print(f"❌ 運行錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✅ 程序執行完成!")
    return 0

if __name__ == "__main__":
    exit(main())
