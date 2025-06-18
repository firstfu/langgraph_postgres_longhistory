"""
=============================================================================
運動彩券賽事預測系統 - 配置文件
包含模型參數、訓練設定、數據路徑等配置選項
=============================================================================
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """模型配置類"""

    # 基礎模型設定
    model_name: str = 'hfl/chinese-roberta-wwm-ext'  # 預訓練模型
    num_labels: int = 3  # 分類數量：客勝、平局、主勝
    max_length: int = 512  # 最大序列長度
    dropout_rate: float = 0.3  # Dropout 率

    # 訓練參數
    num_epochs: int = 5  # 訓練週期
    batch_size: int = 16  # 批次大小
    learning_rate: float = 2e-5  # 學習率
    warmup_steps: int = 500  # 預熱步數
    weight_decay: float = 0.01  # 權重衰減

    # 驗證和早停
    validation_split: float = 0.1  # 驗證集比例
    test_split: float = 0.2  # 測試集比例
    early_stopping_patience: int = 3  # 早停耐心值

    # 評估設定
    eval_steps: int = 500  # 評估步數
    save_steps: int = 500  # 保存步數
    logging_steps: int = 100  # 日誌步數

    # 預測閾值
    prediction_threshold: float = 0.5  # 預測信心閾值

@dataclass
class DataConfig:
    """數據配置類"""

    # 數據生成設定
    num_samples: int = 2000  # 生成樣本數量
    sports_types: List[str] = None  # 支援的運動類型

    # 球隊設定
    teams: Dict[str, List[str]] = None

    # 數據路徑
    data_dir: str = './data'  # 數據目錄
    model_dir: str = './models'  # 模型保存目錄
    output_dir: str = './outputs'  # 輸出目錄

    def __post_init__(self):
        """初始化後處理"""
        if self.sports_types is None:
            self.sports_types = ['football', 'basketball', 'baseball']

        if self.teams is None:
            self.teams = {
                'football': [
                    '曼城', '利物浦', '切爾西', '阿森納', '曼聯', '熱刺',
                    '紐卡斯爾', '布萊頓', '水晶宮', '富勒姆', '西漢姆',
                    '埃弗頓', '萊斯特城', '伯恩利', '狼隊', '諾丁漢森林'
                ],
                'basketball': [
                    '湖人', '勇士', '塞爾提克', '熱火', '公牛', '馬刺',
                    '快艇', '太陽', '76人', '籃網', '公鹿', '爵士',
                    '國王', '魔術', '黃蜂', '活塞'
                ],
                'baseball': [
                    '洋基', '道奇', '紅襪', '巨人', '老虎', '天使',
                    '遊騎兵', '太空人', '運動家', '水手', '皇家', '雙城',
                    '白襪', '響尾蛇', '馬林魚', '教士'
                ]
            }

@dataclass
class FeatureConfig:
    """特徵配置類"""

    # 球隊特徵
    use_team_form: bool = True  # 使用球隊狀態
    use_head_to_head: bool = True  # 使用歷史對戰
    use_home_advantage: bool = True  # 使用主場優勢

    # 球員特徵
    use_player_stats: bool = True  # 使用球員統計
    use_injury_info: bool = True  # 使用傷病資訊

    # 環境特徵
    use_weather: bool = True  # 使用天氣資訊
    use_venue_info: bool = True  # 使用場地資訊

    # 時間特徵
    use_season_info: bool = True  # 使用賽季資訊
    use_match_importance: bool = True  # 使用比賽重要性

@dataclass
class SystemConfig:
    """系統配置類"""

    # 計算設定
    device: str = 'auto'  # 設備選擇：'auto', 'cpu', 'cuda'
    num_workers: int = 4  # 數據加載器工作線程數
    pin_memory: bool = True  # 是否使用釘住內存

    # 日誌設定
    log_level: str = 'INFO'  # 日誌級別
    log_file: Optional[str] = None  # 日誌文件路徑

    # 隨機種子
    random_seed: int = 42  # 隨機種子

    # API 設定（如果需要）
    api_host: str = '0.0.0.0'  # API 主機
    api_port: int = 8000  # API 端口

class Config:
    """主配置類"""

    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.system = SystemConfig()

        # 創建必要的目錄
        self._create_directories()

    def _create_directories(self):
        """創建必要的目錄"""
        directories = [
            self.data.data_dir,
            self.data.model_dir,
            self.data.output_dir,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_model_path(self, model_name: str = 'sports_prediction_model') -> str:
        """獲取模型保存路徑"""
        return os.path.join(self.data.model_dir, model_name)

    def get_data_path(self, filename: str) -> str:
        """獲取數據文件路徑"""
        return os.path.join(self.data.data_dir, filename)

    def get_output_path(self, filename: str) -> str:
        """獲取輸出文件路徑"""
        return os.path.join(self.data.output_dir, filename)

    def update_from_dict(self, config_dict: Dict):
        """從字典更新配置"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'features': self.features.__dict__,
            'system': self.system.__dict__
        }

    def save_config(self, filepath: str):
        """保存配置到文件"""
        import json

        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

    def load_config(self, filepath: str):
        """從文件載入配置"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        self.update_from_dict(config_dict)

# 全局配置實例
config = Config()

# 環境變量支援
def load_config_from_env():
    """從環境變量載入配置"""

    # 模型配置
    if os.getenv('MODEL_NAME'):
        config.model.model_name = os.getenv('MODEL_NAME')

    if os.getenv('BATCH_SIZE'):
        config.model.batch_size = int(os.getenv('BATCH_SIZE'))

    if os.getenv('LEARNING_RATE'):
        config.model.learning_rate = float(os.getenv('LEARNING_RATE'))

    if os.getenv('NUM_EPOCHS'):
        config.model.num_epochs = int(os.getenv('NUM_EPOCHS'))

    # 數據配置
    if os.getenv('NUM_SAMPLES'):
        config.data.num_samples = int(os.getenv('NUM_SAMPLES'))

    if os.getenv('DATA_DIR'):
        config.data.data_dir = os.getenv('DATA_DIR')

    if os.getenv('MODEL_DIR'):
        config.data.model_dir = os.getenv('MODEL_DIR')

    # 系統配置
    if os.getenv('DEVICE'):
        config.system.device = os.getenv('DEVICE')

    if os.getenv('RANDOM_SEED'):
        config.system.random_seed = int(os.getenv('RANDOM_SEED'))

# 支援的預訓練模型列表
SUPPORTED_MODELS = {
    'chinese-roberta': 'hfl/chinese-roberta-wwm-ext',
    'chinese-bert': 'bert-base-chinese',
    'chinese-macbert': 'hfl/chinese-macbert-base',
    'multilingual-bert': 'bert-base-multilingual-cased',
    'xlm-roberta': 'xlm-roberta-base'
}

# 運動類型映射
SPORT_MAPPINGS = {
    'football': '足球',
    'basketball': '籃球',
    'baseball': '棒球',
    'tennis': '網球',
    'volleyball': '排球'
}

# 結果標籤映射
RESULT_LABELS = {
    0: '客勝',
    1: '平局',
    2: '主勝'
}

# 初始化環境變量配置
load_config_from_env()