# =============================================================================
# 基於 LLM 微調的多標籤文本分類模型
# 功能：使用預訓練的中文語言模型進行多標籤分類
# 技術：BERT/RoBERTa + 微調 + 多標籤分類頭
# 優勢：更好的語義理解能力，支援中文語言特性
# =============================================================================

import matplotlib

matplotlib.use('Agg')  # 使用非互動式後端避免顯示問題
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             hamming_loss, jaccard_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
# 處理不同版本的 transformers 套件匯入
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

warnings.filterwarnings('ignore')

# 設定隨機種子確保結果可重現
torch.manual_seed(42)
np.random.seed(42)

# 1. 準備訓練數據
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

# 2. 自定義數據集類
class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, mlb=None, max_length=128, is_training=True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

        if is_training:
            self.mlb = MultiLabelBinarizer()
            labels_encoded = self.mlb.fit_transform(labels)
            # 確保轉換為密集數組格式
            if sparse.issparse(labels_encoded):
                self.labels = labels_encoded.toarray()  # type: ignore
            else:
                self.labels = labels_encoded
        else:
            self.mlb = mlb
            labels_encoded = self.mlb.transform(labels)
            # 確保轉換為密集數組格式
            if sparse.issparse(labels_encoded):
                self.labels = labels_encoded.toarray()  # type: ignore
            else:
                self.labels = labels_encoded

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # 使用 tokenizer 編碼文本
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
            'labels': torch.FloatTensor(self.labels[idx])  # type: ignore
        }

# 3. 多標籤分類模型
class LLMMultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super(LLMMultiLabelClassifier, self).__init__()

        # 載入預訓練模型配置
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # 分類頭
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 凍結部分預訓練層（可選）
        # self._freeze_layers()

    def _freeze_layers(self):
        """凍結前幾層以穩定訓練"""
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        for layer in self.backbone.encoder.layer[:8]:  # 凍結前8層
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用 [CLS] token 的表示
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return torch.sigmoid(logits)  # 多標籤分類使用 sigmoid

# 4. 訓練函數
def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=2e-5):
    """訓練 LLM 多標籤分類模型"""

    # 優化器和學習率調度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 驗證階段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print()

    return train_losses, val_losses

# 5. 評估函數
def evaluate_model(model, test_loader, mlb, device, threshold=0.5):
    """評估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids, attention_mask)
            probabilities = outputs.cpu().numpy()
            predictions = (probabilities > threshold).astype(int)

            all_predictions.extend(predictions)
            all_targets.extend(labels.numpy())
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

# 6. 預測新文本
def predict_text(model, tokenizer, text, mlb, device, threshold=0.3):
    """預測單個文本的標籤"""
    model.eval()

    # 編碼文本
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = outputs.cpu().numpy()[0]

    # 自適應閾值
    predictions = (probabilities > threshold).astype(int)
    if predictions.sum() == 0:  # 如果沒有預測到任何標籤
        # 選擇機率最高的標籤
        top_indices = np.argsort(probabilities)[-2:]  # 選擇前2個最高機率的標籤
        predictions[top_indices] = 1

    predicted_labels = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            predicted_labels.append(mlb.classes_[i])

    # 顯示結果
    print(f"\n📝 文本: '{text}'")
    print(f"🎯 閾值: {threshold}")
    print("📈 預測結果（按機率排序）：")

    # 按機率排序顯示
    prob_indices = np.argsort(probabilities)[::-1]
    for i in prob_indices:
        status = "✓" if predictions[i] == 1 else "✗"
        print(f"  {status} {mlb.classes_[i]}: {probabilities[i]:.3f}")

    print(f"🏷️  最終預測標籤: {predicted_labels}")
    return predicted_labels

# 7. 繪製訓練曲線
def plot_training_curves(train_losses, val_losses):
    """繪製訓練和驗證損失曲線"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 1, 1)
    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('llm_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 訓練曲線已保存為 'llm_training_curves.png'")

# 8. 主函數
def main():
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用設備: {device}")

    # 選擇預訓練模型（中文 BERT）
    model_name = "bert-base-chinese"  # 或使用 "hfl/chinese-roberta-wwm-ext"
    print(f"🤖 使用模型: {model_name}")

    try:
        # 載入 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer 載入成功")
    except Exception as e:
        print(f"❌ 載入 tokenizer 失敗: {e}")
        print("💡 請先安裝 transformers: pip install transformers")
        return

    # 準備數據
    print("\n📚 準備訓練數據...")
    texts, labels = create_comprehensive_data()
    print(f"✅ 共 {len(texts)} 個訓練樣本")

    # 分割數據
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, random_state=42, stratify=None
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    print(f"📊 數據分割: 訓練 {len(train_texts)}, 驗證 {len(val_texts)}, 測試 {len(test_texts)}")

    # 創建數據集
    train_dataset = MultiLabelTextDataset(train_texts, train_labels, tokenizer, is_training=True)
    val_dataset = MultiLabelTextDataset(val_texts, val_labels, tokenizer, train_dataset.mlb, is_training=False)
    test_dataset = MultiLabelTextDataset(test_texts, test_labels, tokenizer, train_dataset.mlb, is_training=False)

    # 創建數據載入器
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 獲取標籤信息
    num_labels = len(train_dataset.mlb.classes_)
    print(f"🏷️  標籤數量: {num_labels}")
    print(f"🏷️  標籤類別: {list(train_dataset.mlb.classes_)}")

    # 創建模型
    print(f"\n🏗️  創建模型...")
    model = LLMMultiLabelClassifier(model_name, num_labels).to(device)
    print("✅ 模型創建成功")

    # 訓練模型
    print(f"\n🎯 開始訓練...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=5, learning_rate=2e-5
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
        predict_text(model, tokenizer, text, train_dataset.mlb, device, threshold=0.3)

    print(f"\n🎉 LLM 多標籤分類訓練完成！")

    # 保存模型（可選）
    # torch.save(model.state_dict(), 'llm_multilabel_model.pth')
    # print("💾 模型已保存")

if __name__ == "__main__":
    main()