import matplotlib.pyplot as plt
import numpy as np

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def coffee_shop_backprop():
    """咖啡店訂價的反向傳播例子"""

    print("☕ 咖啡店訂價的反向傳播")
    print("="*50)
    print()

    print("📖 故事背景：")
    print("你開了一家咖啡店，想要自動調整價格來最大化利潤")
    print("但是價格太高顧客會減少，價格太低利潤會降低")
    print()

    print("🔗 業務邏輯鏈：")
    print("咖啡價格 → 顧客數量 → 營業額 → 利潤")
    print()

    # 設定初始狀況
    print("📊 當前狀況：")
    price = 50  # 咖啡價格50元
    print(f"咖啡價格：{price} 元")

    # 計算過程（前向傳播）
    customers = 100 - price * 0.8  # 價格越高，顧客越少
    revenue = customers * price     # 營業額 = 顧客數 × 價格
    cost = customers * 20          # 成本：每杯20元
    profit = revenue - cost        # 利潤 = 營業額 - 成本

    print(f"顧客數量：{customers:.1f} 人")
    print(f"營業額：{revenue:.1f} 元")
    print(f"成本：{cost:.1f} 元")
    print(f"利潤：{profit:.1f} 元")
    print()

    # 目標利潤
    target_profit = 2000
    print(f"🎯 目標利潤：{target_profit} 元")
    print(f"❌ 當前差距：{target_profit - profit:.1f} 元")
    print()

    print("🤔 問題：如何調整價格來達到目標利潤？")
    print()

    # 開始反向傳播
    print("🔄 反向傳播分析：")
    print("="*30)

    # 步驟1：計算利潤誤差
    profit_error = profit - target_profit
    print(f"步驟1：計算利潤誤差")
    print(f"利潤誤差 = 實際利潤 - 目標利潤")
    print(f"利潤誤差 = {profit:.1f} - {target_profit} = {profit_error:.1f}")
    print("（負數表示利潤不夠）")
    print()

    # 步驟2：分析價格對利潤的影響
    print(f"步驟2：分析價格對利潤的影響")
    print("利潤 = 營業額 - 成本")
    print("利潤 = (顧客數 × 價格) - (顧客數 × 20)")
    print("利潤 = 顧客數 × (價格 - 20)")
    print("利潤 = (100 - 價格×0.8) × (價格 - 20)")
    print()

    # 計算價格對利潤的影響率（梯度）
    # 利潤 = (100 - 0.8*價格) × (價格 - 20)
    # d利潤/d價格 = -0.8 × (價格-20) + (100-0.8*價格) × 1
    gradient = -0.8 * (price - 20) + (100 - 0.8 * price)

    print(f"價格對利潤的影響率：")
    print(f"d利潤/d價格 = -0.8×({price}-20) + (100-0.8×{price})")
    print(f"d利潤/d價格 = {gradient:.2f}")
    print("（正數表示漲價會增加利潤）")
    print()

    # 步驟3：計算價格調整量
    learning_rate = 0.01  # 學習率
    price_adjustment = -learning_rate * profit_error * gradient

    print(f"步驟3：計算價格調整量")
    print(f"價格調整 = -學習率 × 利潤誤差 × 影響率")
    print(f"價格調整 = -{learning_rate} × {profit_error:.1f} × {gradient:.2f}")
    print(f"價格調整 = {price_adjustment:.2f}")
    print()

    # 步驟4：更新價格
    new_price = price + price_adjustment
    print(f"步驟4：更新價格")
    print(f"新價格 = 舊價格 + 調整量")
    print(f"新價格 = {price} + {price_adjustment:.2f} = {new_price:.2f}")
    print()

    # 驗證新價格的效果
    print("✅ 驗證新價格效果：")
    new_customers = 100 - new_price * 0.8
    new_revenue = new_customers * new_price
    new_cost = new_customers * 20
    new_profit = new_revenue - new_cost

    print(f"新顧客數量：{new_customers:.1f} 人")
    print(f"新利潤：{new_profit:.1f} 元")
    print(f"改善程度：{new_profit - profit:.1f} 元")
    print()

    return price, new_price, profit, new_profit

def student_study_backprop():
    """學生讀書時間的反向傳播例子"""

    print("\n" + "="*60)
    print("📚 學生讀書時間優化")
    print("="*60)
    print()

    print("📖 故事：")
    print("小明想要考到85分，但現在只考了70分")
    print("需要調整讀書時間分配")
    print()

    print("🔗 學習鏈：")
    print("數學讀書時間 → 數學分數 → 總分 → 與目標的差距")
    print()

    # 初始狀況
    math_study_hours = 2  # 數學讀書2小時
    english_study_hours = 3  # 英文讀書3小時

    print(f"📊 當前狀況：")
    print(f"數學讀書時間：{math_study_hours} 小時")
    print(f"英文讀書時間：{english_study_hours} 小時")

    # 前向計算
    math_score = 50 + math_study_hours * 8  # 基礎50分，每小時+8分
    english_score = 60 + english_study_hours * 5  # 基礎60分，每小時+5分
    total_score = (math_score + english_score) / 2

    print(f"數學分數：{math_score} 分")
    print(f"英文分數：{english_score} 分")
    print(f"平均分數：{total_score} 分")
    print()

    target_score = 85
    print(f"🎯 目標分數：{target_score} 分")
    print(f"❌ 分數差距：{target_score - total_score} 分")
    print()

    # 反向傳播
    print("🔄 反向傳播：如何調整讀書時間？")
    print("="*40)

    # 計算誤差
    score_error = total_score - target_score
    print(f"步驟1：計算分數誤差")
    print(f"誤差 = {total_score} - {target_score} = {score_error}")
    print()

    # 計算梯度
    print(f"步驟2：計算數學讀書時間的影響")
    print("總分 = (數學分數 + 英文分數) ÷ 2")
    print("數學分數 = 50 + 數學讀書時間 × 8")
    print("所以：數學讀書時間對總分的影響 = 8 ÷ 2 = 4")

    math_gradient = 4  # 數學讀書時間每增加1小時，總分增加4分

    print(f"數學讀書時間梯度：{math_gradient}")
    print()

    # 計算調整量
    learning_rate = 0.1
    math_adjustment = -learning_rate * score_error * math_gradient

    print(f"步驟3：計算調整量")
    print(f"調整量 = -學習率 × 誤差 × 梯度")
    print(f"調整量 = -{learning_rate} × {score_error} × {math_gradient}")
    print(f"調整量 = {math_adjustment:.2f} 小時")
    print()

    # 更新讀書時間
    new_math_hours = math_study_hours + math_adjustment
    print(f"步驟4：更新讀書時間")
    print(f"新的數學讀書時間 = {math_study_hours} + {math_adjustment:.2f} = {new_math_hours:.2f} 小時")
    print()

    # 驗證
    new_math_score = 50 + new_math_hours * 8
    new_total = (new_math_score + english_score) / 2

    print("✅ 驗證結果：")
    print(f"新數學分數：{new_math_score:.1f} 分")
    print(f"新總分：{new_total:.1f} 分")
    print(f"改善：{new_total - total_score:.1f} 分")
    print()

def simple_neural_network_backprop():
    """最簡單的神經網路反向傳播"""

    print("\n" + "="*60)
    print("🧠 最簡單的神經網路反向傳播")
    print("="*60)
    print()

    print("🏗️ 網路結構：")
    print("輸入(x) → 權重(w) → 輸出(y)")
    print("非常簡單：y = x × w")
    print()

    # 初始設定
    x = 3.0      # 輸入
    w = 0.5      # 權重
    target = 2.0 # 目標輸出

    print(f"📊 初始狀況：")
    print(f"輸入 x = {x}")
    print(f"權重 w = {w}")
    print(f"目標輸出 = {target}")
    print()

    # 前向傳播
    y = x * w
    print(f"🚀 前向傳播：")
    print(f"實際輸出 y = x × w = {x} × {w} = {y}")
    print()

    # 計算損失
    error = y - target
    loss = 0.5 * error ** 2  # 均方誤差

    print(f"❌ 計算誤差：")
    print(f"誤差 = 實際輸出 - 目標 = {y} - {target} = {error}")
    print(f"損失 = 0.5 × 誤差² = 0.5 × {error}² = {loss}")
    print()

    # 反向傳播
    print(f"🔄 反向傳播：")
    print("="*20)

    print(f"步驟1：計算損失對輸出的梯度")
    print(f"損失 = 0.5 × (y - target)²")
    print(f"d損失/dy = y - target = {error}")

    dL_dy = error
    print(f"d損失/dy = {dL_dy}")
    print()

    print(f"步驟2：計算輸出對權重的梯度")
    print(f"y = x × w")
    print(f"dy/dw = x = {x}")

    dy_dw = x
    print(f"dy/dw = {dy_dw}")
    print()

    print(f"步驟3：使用鏈式法則")
    print(f"d損失/dw = d損失/dy × dy/dw")
    print(f"d損失/dw = {dL_dy} × {dy_dw} = {dL_dy * dy_dw}")

    dL_dw = dL_dy * dy_dw
    print()

    # 更新權重
    learning_rate = 0.1
    w_adjustment = -learning_rate * dL_dw
    new_w = w + w_adjustment

    print(f"步驟4：更新權重")
    print(f"權重調整 = -學習率 × 梯度")
    print(f"權重調整 = -{learning_rate} × {dL_dw} = {w_adjustment}")
    print(f"新權重 = {w} + {w_adjustment} = {new_w}")
    print()

    # 驗證
    new_y = x * new_w
    new_error = new_y - target
    new_loss = 0.5 * new_error ** 2

    print(f"✅ 驗證新權重：")
    print(f"新輸出 = {x} × {new_w} = {new_y}")
    print(f"新誤差 = {new_error:.3f}")
    print(f"新損失 = {new_loss:.3f}")
    print(f"損失改善 = {loss - new_loss:.3f}")
    print()

def visualize_backprop_process():
    """視覺化反向傳播過程"""

    print("📊 視覺化反向傳播過程")
    print("="*30)

    # 模擬多次訓練過程
    x = 3.0
    target = 2.0
    w = 0.5
    learning_rate = 0.1

    weights = [w]
    outputs = []
    losses = []

    # 訓練10次
    for i in range(10):
        # 前向傳播
        y = x * w
        error = y - target
        loss = 0.5 * error ** 2

        outputs.append(y)
        losses.append(loss)

        # 反向傳播
        dL_dw = error * x
        w = w - learning_rate * dL_dw
        weights.append(w)

    # 繪圖
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 權重變化
    ax1.plot(range(len(weights)), weights, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=target/x, color='r', linestyle='--', label=f'目標權重 ({target/x:.3f})')
    ax1.set_title('權重在訓練過程中的變化', fontsize=14, fontweight='bold')
    ax1.set_xlabel('訓練次數')
    ax1.set_ylabel('權重值')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 輸出變化
    ax2.plot(range(len(outputs)), outputs, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=target, color='r', linestyle='--', label=f'目標輸出 ({target})')
    ax2.set_title('輸出在訓練過程中的變化', fontsize=14, fontweight='bold')
    ax2.set_xlabel('訓練次數')
    ax2.set_ylabel('輸出值')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 損失變化
    ax3.plot(range(len(losses)), losses, 'ro-', linewidth=2, markersize=8)
    ax3.set_title('損失在訓練過程中的變化', fontsize=14, fontweight='bold')
    ax3.set_xlabel('訓練次數')
    ax3.set_ylabel('損失值')
    ax3.grid(True, alpha=0.3)

    # 反向傳播流程圖
    ax4.text(0.1, 0.8, '前向傳播', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
             ha='center')
    ax4.text(0.1, 0.6, '計算損失', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
             ha='center')
    ax4.text(0.1, 0.4, '反向傳播', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
             ha='center')
    ax4.text(0.1, 0.2, '更新權重', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
             ha='center')

    # 添加箭頭
    for i in range(3):
        ax4.annotate('', xy=(0.1, 0.75-i*0.2), xytext=(0.1, 0.85-i*0.2),
                    arrowprops=dict(arrowstyle='->', lw=3, color='blue'))

    # 循環箭頭
    ax4.annotate('', xy=(0.2, 0.8), xytext=(0.2, 0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                              connectionstyle="arc3,rad=0.3"))
    ax4.text(0.35, 0.5, '重複循環\n直到收斂', fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))

    ax4.set_xlim(0, 0.6)
    ax4.set_ylim(0, 1)
    ax4.set_title('反向傳播流程', fontsize=14, fontweight='bold')
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"最終結果：")
    print(f"初始權重：0.5")
    print(f"最終權重：{weights[-1]:.3f}")
    print(f"理想權重：{target/x:.3f}")
    print(f"最終損失：{losses[-1]:.6f}")

if __name__ == "__main__":
    # 執行所有例子
    coffee_shop_backprop()
    student_study_backprop()
    simple_neural_network_backprop()
    visualize_backprop_process()

    print("\n" + "🎉"*25)
    print("反向傳播總結")
    print("🎉"*25)
    print()
    print("🔑 核心概念：")
    print("1. 📊 計算誤差：看看結果差多少")
    print("2. 🔍 追溯責任：往回找每個參數的責任")
    print("3. ⚙️ 計算調整：用鏈式法則算出調整量")
    print("4. 🔧 更新參數：根據責任大小調整參數")
    print("5. 🔄 重複過程：直到達到目標")
    print()
    print("💡 關鍵理解：")
    print("• 反向傳播 = 自動化的「責任分配」系統")
    print("• 從結果往回推，找出每個參數該調多少")
    print("• 就像管理問題：出錯了要找根源並改善")
    print("• 鏈式法則幫我們計算連鎖影響")
    print()
    print("🚀 實際應用：")
    print("• 神經網路用反向傳播自動學習")
    print("• 每次訓練都會讓預測更準確")
    print("• 這就是AI能夠「學習」的原理！")
    print()
    print("🎯 記憶口訣：")
    print("前向算結果，反向找責任")
    print("誤差往回傳，權重跟著調")
    print("鏈式法則幫忙，學習自動化")
    print()
    print("🎉 恭喜！你已經完全理解反向傳播了！")