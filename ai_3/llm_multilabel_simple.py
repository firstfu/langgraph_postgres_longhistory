# =============================================================================
# 簡化版 LLM 風格多標籤文本分類模型
# 功能：模擬 Transformer 架構的多標籤分類
# 技術：自定義 Transformer + 中文文本處理 + 多標籤分類
# 優勢：不需要下載大型預訓練模型，快速測試和學習
# =============================================================================

import matplotlib

matplotlib.use('Agg')  # 使用非互動式後端避免顯示問題
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, f1_score, hamming_loss,
                             jaccard_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# 嘗試導入 jieba，如果沒有則使用簡單的文本處理
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

# 設定隨機種子確保結果可重現
torch.manual_seed(42)
np.random.seed(42)

# 1. 中文文本預處理
class ChineseTextProcessor:
    def __init__(self):
        # 停用詞列表
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個',
            '上', '也', '很', '到', '說', '要', '去', '你', '會', '着', '沒有', '看', '好',
            '自己', '這', '年', '那', '現在', '可以', '但是', '這個', '中', '大', '為',
            '來', '個', '能', '對', '更', '等', '還', '這樣', '進行', '相關', '今天',
            '今日', '宣布', '表示', '指出', '認為', '發現', '顯示', '報告'
        }

    def process_text(self, text):
        """處理中文文本：使用 jieba 分詞或基本字符處理"""
        if JIEBA_AVAILABLE:
            # 使用 jieba 分詞
            words = jieba.cut(text)
            filtered_words = [
                word.strip() for word in words
                if word.strip() and len(word.strip()) > 1 and word.strip() not in self.stop_words
            ]
            return ' '.join(filtered_words)
        else:
            # 簡單的字符分割
            words = []
            i = 0
            while i < len(text):
                # 嘗試提取2-3字詞
                for length in [3, 2, 1]:
                    if i + length <= len(text):
                        word = text[i:i+length]
                        if word not in self.stop_words and len(word.strip()) > 0:
                            words.append(word)
                            i += length
                            break
                else:
                    i += 1

            return ' '.join(words)

# 2. 準備訓練數據
def create_comprehensive_data():
    """創建更豐富的多標籤訓練數據"""
    texts = [
        # 政治相關
        "總統宣布新的經濟刺激政策以促進國家發展",
        "國會議員討論修改稅收法案的相關條款",
        "外交部長與各國代表舉行重要會談",
        "政府推出全民健康保險改革方案",
        "選舉委員會公布最新的民意調查結果",
        "立法院通過新的環境保護法案",
        "市長宣布都市更新計畫投資金額",
        "總理出訪鄰國討論貿易合作協議",

        # 經濟相關
        "股市今日收盤創下歷史新高紀錄",
        "央行宣布調降基準利率刺激經濟",
        "國際油價上漲影響通膨預期",
        "房地產市場交易量大幅增加",
        "企業財報顯示獲利大幅成長",
        "就業市場出現明顯復甦跡象",
        "消費者物價指數持續上升",
        "匯率波動對出口產業造成衝擊",

        # 科技相關
        "人工智慧技術在醫療診斷領域取得突破",
        "5G網路建設帶動相關產業發展",
        "量子計算研究獲得重大進展",
        "區塊鏈技術應用於金融服務創新",
        "自動駕駛汽車完成路測驗證",
        "虛擬實境設備銷量創新高",
        "機器學習演算法優化生產效率",
        "雲端運算服務市場快速擴張",

        # 體育相關
        "NBA總冠軍賽進入決戰時刻",
        "世界盃足球賽小組賽激烈競爭",
        "奧運游泳比賽刷新世界紀錄",
        "職業籃球聯賽季後賽開打",
        "網球四大滿貫賽事精彩對決",
        "棒球世界經典賽國家隊集訓",
        "馬拉松國際賽事吸引萬人參與",
        "羽毛球公開賽冠軍爭奪戰",

        # 健康相關
        "新冠疫苗接種計畫全面啟動",
        "醫院引進最新癌症治療技術",
        "健康飲食習慣預防慢性疾病",
        "心理健康諮商服務需求增加",
        "運動傷害防護知識宣導活動",
        "老年照護政策獲得社會關注",
        "醫療器材創新技術獲得認證",
        "疫情防控措施調整最新消息",

        # 多標籤複合文本 - 政治+經濟
        "政府投資基礎建設促進經濟復甦",
        "央行貨幣政策影響股市表現",
        "國際貿易談判關係經濟發展",
        "稅制改革對企業營運產生影響",

        # 多標籤複合文本 - 科技+經濟
        "科技公司股價因AI發展大漲",
        "電動車產業獲得政府投資支持",
        "數位貨幣技術推動金融創新",
        "半導體產業鏈供應短缺問題",

        # 多標籤複合文本 - 體育+經濟
        "職業運動員簽約代言費創新高",
        "體育賽事轉播權價格持續上漲",
        "運動產業帶動相關經濟發展",
        "電競比賽獎金池突破歷史紀錄",

        # 多標籤複合文本 - 健康+科技
        "AI醫療診斷系統提高準確率",
        "穿戴式健康監測設備普及化",
        "遠距醫療服務技術不斷進步",
        "基因編輯技術治療罕見疾病",

        # 多標籤複合文本 - 三重標籤
        "政府推動健康科技產業發展計畫",  # 政治+健康+科技
        "體育明星投資科技新創公司股權",  # 體育+科技+經濟
        "央行研發數位貨幣區塊鏈平台",   # 政治+經濟+科技
        "醫療保健政策獲得預算支持",      # 政治+健康+經濟
    ]

    labels = [
        # 政治相關
        ["政治", "經濟"],
        ["政治"],
        ["政治"],
        ["政治", "健康"],
        ["政治"],
        ["政治", "環境"],
        ["政治", "經濟"],
        ["政治", "經濟"],

        # 經濟相關
        ["經濟"],
        ["經濟", "政治"],
        ["經濟"],
        ["經濟", "房地產"],
        ["經濟"],
        ["經濟"],
        ["經濟"],
        ["經濟"],

        # 科技相關
        ["科技", "健康", "AI"],
        ["科技", "通訊"],
        ["科技", "量子"],
        ["科技", "經濟"],
        ["科技"],
        ["科技"],
        ["科技", "AI"],
        ["科技", "雲端"],

        # 體育相關
        ["體育", "籃球"],
        ["體育", "足球"],
        ["體育", "游泳"],
        ["體育", "籃球"],
        ["體育", "網球"],
        ["體育", "棒球"],
        ["體育", "跑步"],
        ["體育"],

        # 健康相關
        ["健康", "政治"],
        ["健康", "科技"],
        ["健康"],
        ["健康"],
        ["健康", "體育"],
        ["健康", "政治"],
        ["健康", "科技"],
        ["健康", "政治"],

        # 多標籤複合文本
        ["政治", "經濟"],
        ["政治", "經濟"],
        ["政治", "經濟"],
        ["政治", "經濟"],

        ["科技", "經濟", "AI"],
        ["科技", "經濟", "政治"],
        ["科技", "經濟"],
        ["科技", "經濟"],

        ["體育", "經濟"],
        ["體育", "經濟"],
        ["體育", "經濟"],
        ["體育", "科技", "經濟"],

        ["健康", "科技", "AI"],
        ["健康", "科技"],
        ["健康", "科技"],
        ["健康", "科技"],

        # 三重標籤
        ["政治", "健康", "科技"],
        ["體育", "科技", "經濟"],
        ["政治", "經濟", "科技"],
        ["政治", "健康", "經濟"],
    ]

    return texts, labels

# 3. 自定義數據集類
class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, text_processor, vectorizer=None, mlb=None, is_training=True):
        self.texts = texts
        self.text_processor = text_processor
        self.is_training = is_training

        # 預處理文本
        processed_texts = [self.text_processor.process_text(text) for text in texts]

        if is_training:
            # 訓練時建立新的向量化器和標籤編碼器
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words=None,
                ngram_range=(1, 2),  # 使用 1-2 元組合
                min_df=1
            )
            self.mlb = MultiLabelBinarizer()

            # 文本向量化
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            self.X = tfidf_matrix.toarray()
            # 標籤編碼
            self.y = self.mlb.fit_transform(labels)
        else:
            # 測試時使用已有的向量化器和編碼器
            self.vectorizer = vectorizer
            self.mlb = mlb
            tfidf_matrix = self.vectorizer.transform(processed_texts)
            self.X = tfidf_matrix.toarray()
            self.y = self.mlb.transform(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

# 4. 簡化版 Transformer 風格模型
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_heads=8, dropout_rate=0.3):
        super(SimpleTransformerClassifier, self).__init__()

        # 輸入投影層
        self.input_projection = nn.Linear(input_size, hidden_size)

        # 多頭注意力機制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # 前饋網路
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # 層正規化
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # 分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 輸入投影
        x = self.input_projection(x)  # [batch_size, hidden_size]

        # 添加序列維度用於注意力機制
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 多頭注意力 + 殘差連接
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # 前饋網路 + 殘差連接
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)

        # 移除序列維度並分類
        x = x.squeeze(1)  # [batch_size, hidden_size]
        x = self.classifier(x)

        return x

# 5. 訓練函數
def train_model(model, train_loader, val_loader, device, num_epochs=30, learning_rate=0.001):
    """訓練模型"""

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5
    )

    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 驗證階段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 學習率調度
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print()

    return train_losses, val_losses

# 6. 評估函數
def evaluate_model(model, test_loader, mlb, device, threshold=0.5):
    """評估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            probabilities = outputs.cpu().numpy()
            predictions = (probabilities > threshold).astype(int)

            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 計算評估指標
    hamming = hamming_loss(all_targets, all_predictions)
    jaccard = jaccard_score(all_targets, all_predictions, average='samples', zero_division=0)
    f1_micro = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    print("🔍 模型評估結果：")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Jaccard Score: {jaccard:.4f}")
    print(f"  F1 Score (Micro): {f1_micro:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print()

    # 詳細分類報告
    print("📊 各標籤詳細報告：")
    print(classification_report(
        all_targets, all_predictions,
        target_names=mlb.classes_,
        zero_division=0
    ))

    return all_predictions, all_targets, all_probabilities

# 7. 預測新文本
def predict_text(model, text, text_processor, vectorizer, mlb, device, threshold=0.3):
    """預測單個文本的標籤"""
    model.eval()

    # 預處理文本
    processed_text = text_processor.process_text(text)

    # 向量化
    text_vector = vectorizer.transform([processed_text]).toarray()
    text_tensor = torch.FloatTensor(text_vector).to(device)

    with torch.no_grad():
        outputs = model(text_tensor)
        probabilities = outputs.cpu().numpy()[0]

    # 自適應閾值
    predictions = (probabilities > threshold).astype(int)
    if predictions.sum() == 0:  # 如果沒有預測到任何標籤
        # 選擇機率最高的2個標籤
        top_indices = np.argsort(probabilities)[-2:]
        predictions[top_indices] = 1

    predicted_labels = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            predicted_labels.append(mlb.classes_[i])

    # 顯示結果
    print(f"\n📝 文本: '{text}'")
    print(f"🔧 處理後: '{processed_text}'")
    print(f"🎯 閾值: {threshold}")
    print("📈 預測結果（按機率排序）：")

    # 按機率排序顯示
    prob_indices = np.argsort(probabilities)[::-1]
    for i in prob_indices:
        status = "✓" if predictions[i] == 1 else "✗"
        print(f"  {status} {mlb.classes_[i]}: {probabilities[i]:.3f}")

    print(f"🏷️  最終預測標籤: {predicted_labels}")
    return predicted_labels

# 8. 繪製訓練曲線
def plot_training_curves(train_losses, val_losses):
    """繪製訓練和驗證損失曲線"""
    plt.figure(figsize=(12, 5))

    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Transformer-style Training Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('transformer_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 訓練曲線已保存為 'transformer_training_curves.png'")

# 9. 主函數
def main():
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用設備: {device}")

    # 初始化文本處理器
    if JIEBA_AVAILABLE:
        print("✅ jieba 中文分詞工具已安裝")
    else:
        print("❌ jieba 未安裝，使用簡化版文本處理")
        print("💡 如需更好效果，請安裝: pip install jieba")

    text_processor = ChineseTextProcessor()

    # 準備數據
    print("\n📚 準備訓練數據...")
    texts, labels = create_comprehensive_data()
    print(f"✅ 共 {len(texts)} 個訓練樣本")

    # 分割數據
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    print(f"📊 數據分割: 訓練 {len(train_texts)}, 驗證 {len(val_texts)}, 測試 {len(test_texts)}")

    # 創建數據集
    train_dataset = MultiLabelTextDataset(
        train_texts, train_labels, text_processor, is_training=True
    )
    val_dataset = MultiLabelTextDataset(
        val_texts, val_labels, text_processor,
        train_dataset.vectorizer, train_dataset.mlb, is_training=False
    )
    test_dataset = MultiLabelTextDataset(
        test_texts, test_labels, text_processor,
        train_dataset.vectorizer, train_dataset.mlb, is_training=False
    )

    # 創建數據載入器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 獲取模型參數
    input_size = train_dataset.X.shape[1]
    num_labels = len(train_dataset.mlb.classes_)

    print(f"🏷️  輸入特徵維度: {input_size}")
    print(f"🏷️  標籤數量: {num_labels}")
    print(f"🏷️  標籤類別: {list(train_dataset.mlb.classes_)}")

    # 創建模型
    print(f"\n🏗️  創建 Transformer 風格模型...")
    model = SimpleTransformerClassifier(
        input_size=input_size,
        hidden_size=256,
        num_labels=num_labels,
        num_heads=8,
        dropout_rate=0.3
    ).to(device)
    print("✅ 模型創建成功")

    # 訓練模型
    print(f"\n🎯 開始訓練...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=30, learning_rate=0.001
    )

    # 繪製訓練曲線
    plot_training_curves(train_losses, val_losses)

    # 評估模型
    print("\n🔍 評估模型...")
    predictions, targets, probabilities = evaluate_model(
        model, test_loader, train_dataset.mlb, device
    )

    # 測試新文本預測
    print("\n" + "="*60)
    print("🧪 測試新文本預測：")

    test_texts_new = [
        "政府宣布科技產業減稅政策刺激經濟發展",
        "NBA球星投資區塊鏈新創公司獲得豐厚回報",
        "央行推出數位貨幣技術研發計畫",
        "人工智慧醫療診斷系統提高治療效果",
        "電競選手在亞運會奪金創造歷史",
        "綠能科技股票因政策利多大漲",
        "職業足球員簽約運動品牌代言合約",
        "遠距醫療服務技術獲得投資關注",
        "量子計算突破帶動相關產業發展",
        "體育產業數位轉型獲政府支持"
    ]

    for text in test_texts_new:
        predict_text(
            model, text, text_processor,
            train_dataset.vectorizer, train_dataset.mlb,
            device, threshold=0.3
        )

    print(f"\n🎉 Transformer 風格多標籤分類訓練完成！")

if __name__ == "__main__":
    main()