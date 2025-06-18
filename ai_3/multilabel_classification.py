# =============================================================================
# 多標籤文本分類模型 - 使用PyTorch實現
# 功能：對中文文本進行多標籤分類，支援一個文本對應多個標籤
# 技術：TF-IDF特徵提取 + 神經網路分類器
# =============================================================================

import matplotlib

matplotlib.use('Agg')  # 使用非互動式後端避免顯示問題
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse  # 添加 scipy 稀疏矩陣支援
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, f1_score, hamming_loss,
                             jaccard_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

# 設定隨機種子確保結果可重現
torch.manual_seed(42)
np.random.seed(42)

# 1. 準備範例數據
def create_sample_data():
    """創建範例文本數據和多標籤"""
    texts = [
        # 政治相關
        "總統今天發表經濟政策演說",
        "國會通過新法案",
        "市長宣布都市更新計畫",
        "外交部長會見各國大使",
        "政府推出社會福利政策",
        "選舉投票率創新高",
        "立法院審查預算案",
        "總理訪問友邦國家",

        # 體育相關
        "NBA總冠軍賽今晚開打",
        "足球世界盃精彩進球集錦",
        "奧運會游泳比賽破紀錄",
        "籃球明星轉隊引發關注",
        "網球公開賽決賽對戰",
        "棒球季後賽激烈競爭",
        "馬拉松比賽創佳績",
        "羽毛球冠軍賽精彩對決",
        "高爾夫球公開賽開打",
        "排球聯賽總決賽",

        # 科技相關
        "人工智慧技術突破創新高",
        "5G網路建設加速推進",
        "量子計算研究新進展",
        "雲端服務市場擴張",
        "機器學習演算法優化",
        "區塊鏈技術新應用",
        "虛擬實境產品發布",
        "自動駕駛技術測試",
        "智慧型手機新功能",
        "物聯網設備普及化",

        # 經濟相關
        "股市今日大漲科技股領漲",
        "央行調整利率政策",
        "國際貿易協議談判",
        "房地產市場趨勢分析",
        "通膨率持續上升",
        "就業率創歷史新高",
        "匯率波動影響出口",
        "消費者信心指數下滑",
        "企業財報表現亮眼",
        "經濟成長率超出預期",

        # 健康相關
        "新冠疫情防控措施更新",
        "醫院推出健康檢查方案",
        "疫苗接種率達到目標",
        "新藥獲得政府核准",
        "醫療設備技術升級",
        "健康飲食觀念推廣",
        "運動傷害預防宣導",
        "心理健康諮商服務",

        # 多標籤複合文本
        "政府投資科技產業促進經濟發展",  # 政治+科技+經濟
        "NBA球星投資區塊鏈新創公司",  # 體育+籃球+科技+經濟
        "醫療AI診斷技術獲政府補助",  # 健康+科技+政治
        "電競比賽獎金創新高紀錄",  # 體育+科技+經濟
        "央行研發數位貨幣技術",  # 經濟+科技+政治
        "運動員代言健康食品廣告",  # 體育+健康+經濟
        "智慧醫療設備銷量大增",  # 科技+健康+經濟
        "政府推動體育產業發展",  # 政治+體育+經濟
        "房地產科技應用趨勢",  # 房地產+科技+經濟
        "健康科技股票表現強勁",  # 健康+科技+經濟
        "體育明星投資房地產",  # 體育+房地產+經濟
        "政府推廣運動健康政策",  # 政治+體育+健康
        "AI技術改善醫療診斷",  # 科技+AI+健康
        "電動車產業政策支持",  # 科技+經濟+政治
        "運動器材科技創新",  # 體育+科技
    ]

    # 每個文本對應的多個標籤
    labels = [
        # 政治相關
        ["政治", "經濟"],
        ["政治"],
        ["政治", "房地產"],
        ["政治"],
        ["政治", "健康"],
        ["政治"],
        ["政治", "經濟"],
        ["政治"],

        # 體育相關
        ["體育", "籃球"],
        ["體育", "足球"],
        ["體育", "游泳"],
        ["體育", "籃球"],
        ["體育", "網球"],
        ["體育", "棒球"],
        ["體育", "跑步"],
        ["體育"],
        ["體育"],
        ["體育"],

        # 科技相關
        ["科技", "AI"],
        ["科技", "通訊"],
        ["科技", "量子"],
        ["科技", "雲端"],
        ["科技", "AI"],
        ["科技"],
        ["科技"],
        ["科技"],
        ["科技"],
        ["科技"],

        # 經濟相關
        ["經濟", "科技"],
        ["經濟", "政治"],
        ["政治", "經濟"],
        ["經濟", "房地產"],
        ["經濟"],
        ["經濟"],
        ["經濟"],
        ["經濟"],
        ["經濟"],
        ["經濟"],

        # 健康相關
        ["健康", "政治"],
        ["健康"],
        ["健康", "政治"],
        ["健康"],
        ["健康", "科技"],
        ["健康"],
        ["健康", "體育"],
        ["健康"],

        # 多標籤複合文本
        ["政治", "科技", "經濟"],
        ["體育", "籃球", "科技", "經濟"],
        ["健康", "科技", "政治"],
        ["體育", "科技", "經濟"],
        ["經濟", "科技", "政治"],
        ["體育", "健康", "經濟"],
        ["科技", "健康", "經濟"],
        ["政治", "體育", "經濟"],
        ["房地產", "科技", "經濟"],
        ["健康", "科技", "經濟"],
        ["體育", "房地產", "經濟"],
        ["政治", "體育", "健康"],
        ["科技", "AI", "健康"],
        ["科技", "經濟", "政治"],
        ["體育", "科技"],
    ]

    return texts, labels

# 2. 自定義數據集類
class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer=None, mlb=None, is_training=True):
        self.texts = texts
        self.labels = labels
        self.is_training = is_training

        if is_training:
            # 訓練時建立新的向量化器和標籤編碼器
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            self.mlb = MultiLabelBinarizer()

            # 文本向量化 - 確保轉換為 numpy 密集矩陣
            tfidf_sparse = self.vectorizer.fit_transform(texts)
            self.X = tfidf_sparse.toarray()  # 直接轉換為密集矩陣
            # 標籤編碼
            self.y = self.mlb.fit_transform(labels)
        else:
            # 測試時使用已有的向量化器和編碼器
            self.vectorizer = vectorizer
            self.mlb = mlb
            tfidf_sparse = self.vectorizer.transform(texts)
            self.X = tfidf_sparse.toarray()  # 直接轉換為密集矩陣
            self.y = self.mlb.transform(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

# 3. 定義多標籤分類模型
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, dropout_rate=0.2):
        super(MultiLabelClassifier, self).__init__()

        # 簡化網路架構，防止過度擬合
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加批次正規化
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels),
            nn.Sigmoid()  # 關鍵：使用Sigmoid激活函數
        )

    def forward(self, x):
        return self.network(x)

# 4. 訓練函數
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向傳播
            outputs = model(data)

            # 計算損失
            loss = criterion(outputs, targets)

            # 反向傳播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return train_losses

# 5. 評估函數
def evaluate_model(model, test_loader, mlb, device, threshold=0.5):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            # 根據閾值進行預測
            predictions = (outputs > threshold).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 計算評估指標
    hamming = hamming_loss(all_targets, all_predictions)
    jaccard = jaccard_score(all_targets, all_predictions, average='samples', zero_division=0)
    f1_micro = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    print(f"\n評估結果：")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")

    # 顯示每個標籤的詳細報告
    print(f"\n各標籤詳細報告：")
    print(classification_report(all_targets, all_predictions,
                              target_names=mlb.classes_, zero_division=0))

    return all_predictions, all_targets

# 6. 預測新文本的函數
def predict_new_text(model, text, vectorizer, mlb, device, threshold=0.3):
    """預測新文本的標籤"""
    model.eval()

    # 向量化新文本
    text_sparse = vectorizer.transform([text])
    text_vector = text_sparse.toarray()  # 轉換為numpy密集矩陣
    text_tensor = torch.FloatTensor(text_vector).to(device)

    with torch.no_grad():
        output = model(text_tensor)
        probabilities = output.cpu().numpy()[0]

        # 使用多種閾值進行預測
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        best_threshold = threshold

        # 如果原始閾值沒有預測到任何標籤，嘗試較低的閾值
        original_predictions = (probabilities > threshold).astype(int)
        if original_predictions.sum() == 0:
            for t in [0.2, 0.15, 0.1, 0.05]:
                test_predictions = (probabilities > t).astype(int)
                if test_predictions.sum() > 0:
                    best_threshold = t
                    break

        predictions = (probabilities > best_threshold).astype(int)

    # 獲取預測的標籤
    predicted_labels = mlb.inverse_transform(predictions.reshape(1, -1))[0]

    # 顯示結果
    print(f"\n文本: '{text}'")
    print(f"使用閾值: {best_threshold}")
    print("預測標籤及機率：")

    # 排序顯示（按機率從高到低）
    prob_with_labels = list(zip(mlb.classes_, probabilities, predictions))
    prob_with_labels.sort(key=lambda x: x[1], reverse=True)

    for label, prob, pred in prob_with_labels:
        status = "✓" if pred == 1 else "✗"
        print(f"  {status} {label}: {prob:.3f}")
    print(f"最終預測標籤: {list(predicted_labels)}")

    return predicted_labels

# 7. 繪製訓練損失曲線
def plot_training_loss(train_losses):
    """繪製並保存訓練損失曲線"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss Curve', fontsize=14)  # 使用英文避免字體問題
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 保存圖片而不是顯示
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()  # 關閉圖形釋放記憶體
    print("📊 訓練損失曲線已保存為 'training_loss.png'")

# 8. 主函數
def main():
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    import sys
    sys.stdout.flush()  # 強制刷新輸出

    # 準備數據
    print("準備數據...")
    texts, labels = create_sample_data()

    # 分割訓練和測試數據
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    # 創建數據集
    train_dataset = MultiLabelTextDataset(train_texts, train_labels, is_training=True)
    test_dataset = MultiLabelTextDataset(test_texts, test_labels,
                                       train_dataset.vectorizer,
                                       train_dataset.mlb,
                                       is_training=False)

    # 創建數據加載器 - 調整批次大小
    batch_size = min(16, len(train_dataset) // 3)  # 動態調整批次大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 獲取輸入維度和標籤數量
    input_size = train_dataset.X.shape[1]
    num_labels = len(train_dataset.mlb.classes_)

    print(f"輸入特徵維度: {input_size}")
    print(f"標籤數量: {num_labels}")
    print(f"標籤類別: {list(train_dataset.mlb.classes_)}")

    # 創建模型 - 降低模型複雜度
    model = MultiLabelClassifier(input_size, 128, num_labels).to(device)  # 降低hidden_size從256到128

    # 定義損失函數和優化器 - 調整學習率
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)  # 提高學習率和weight_decay

    # 訓練模型 - 減少訓練週期
    print("\n開始訓練...")
    train_losses = train_model(model, train_loader, criterion, optimizer, 50, device)  # 降低epochs從100到50

    # 繪製訓練損失
    plot_training_loss(train_losses)

    # 評估模型
    print("\n評估模型...")
    predictions, targets = evaluate_model(model, test_loader, train_dataset.mlb, device)

    # 測試新文本預測
    print("\n" + "="*50)
    print("測試新文本預測：")
    test_texts_new = [
        # 明確的多標籤文本
        "政府宣布科技產業減稅政策刺激經濟發展",  # 政治+經濟+科技
        "NBA球星投資區塊鏈新創公司",  # 體育+籃球+科技+經濟
        "央行數位貨幣技術研發進展順利",  # 經濟+政治+科技
        "電競選手在亞運會奪金引發投資熱潮",  # 體育+科技+經濟

        # 單一領域文本
        "醫院推出新冠肺炎快篩服務",  # 健康
        "房地產市場出現回溫跡象",  # 經濟+房地產
        "學校停課改為線上教學模式",  # 教育+科技
        "環保團體抗議工廠污染問題",  # 環境+政治

        # 體育相關多標籤
        "足球明星簽約代言運動品牌合約",  # 體育+足球+經濟
        "馬拉松賽事帶動觀光產業發展",  # 體育+跑步+經濟

        # 其他多樣化內容
        "音樂節門票銷售創歷史新高",  # 娛樂+經濟
        "氣候變遷影響農業生產政策"   # 環境+政治+經濟
    ]

    # 使用較低的閾值來提高多標籤預測的可能性
    for text in test_texts_new:
        predict_new_text(model, text, train_dataset.vectorizer,
                        train_dataset.mlb, device, threshold=0.3)  # 降低閾值從0.5到0.3

    print("\n訓練完成！")

if __name__ == "__main__":
    main()