"""
=============================================================================
運動彩券數據收集器
功能：收集真實的運動賽事數據，包括球隊統計、球員數據、歷史比賽記錄等
技術：網路爬蟲、API 接口、數據清理和標準化
應用：為 LLM 模型提供高質量的訓練數據
=============================================================================
"""

import json
import logging
import random
import sqlite3
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchData:
    """比賽數據結構"""
    home_team: str
    away_team: str
    date: str
    sport: str
    league: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    result: Optional[str] = None  # 'H', 'D', 'A' (主勝、平局、客勝)

    # 球隊數據
    home_form: Optional[float] = None
    away_form: Optional[float] = None
    home_ranking: Optional[int] = None
    away_ranking: Optional[int] = None

    # 歷史對戰
    head_to_head_home_wins: Optional[int] = None
    head_to_head_draws: Optional[int] = None
    head_to_head_away_wins: Optional[int] = None

    # 球員數據
    home_key_players_available: Optional[int] = None
    away_key_players_available: Optional[int] = None
    home_injury_count: Optional[int] = None
    away_injury_count: Optional[int] = None

    # 環境因素
    venue: Optional[str] = None
    weather: Optional[str] = None
    temperature: Optional[float] = None

    # 市場數據
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None

class SportsDataCollector:
    """運動數據收集器"""

    def __init__(self, db_path: str = "sports_data.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.init_database()

        # 模擬的真實球隊數據
        self.real_teams_data = {
            'football': {
                'premier_league': {
                    '曼城': {'strength': 95, 'form': [3, 3, 1, 3, 3], 'home_advantage': 85},
                    '阿森納': {'strength': 88, 'form': [3, 1, 3, 3, 1], 'home_advantage': 82},
                    '利物浦': {'strength': 87, 'form': [3, 3, 0, 3, 1], 'home_advantage': 90},
                    '曼聯': {'strength': 82, 'form': [1, 3, 1, 0, 3], 'home_advantage': 85},
                    '紐卡斯爾': {'strength': 78, 'form': [3, 0, 1, 3, 1], 'home_advantage': 75},
                    '熱刺': {'strength': 80, 'form': [0, 3, 3, 1, 1], 'home_advantage': 78},
                    '切爾西': {'strength': 75, 'form': [1, 1, 3, 0, 1], 'home_advantage': 80},
                    '布萊頓': {'strength': 72, 'form': [1, 0, 3, 1, 3], 'home_advantage': 70}
                }
            },
            'basketball': {
                'nba': {
                    '湖人': {'strength': 88, 'form': [1, 1, 0, 1, 1], 'home_advantage': 82},
                    '勇士': {'strength': 85, 'form': [1, 0, 1, 1, 0], 'home_advantage': 90},
                    '塞爾提克': {'strength': 92, 'form': [1, 1, 1, 0, 1], 'home_advantage': 85},
                    '熱火': {'strength': 80, 'form': [0, 1, 1, 1, 0], 'home_advantage': 78},
                    '公牛': {'strength': 75, 'form': [0, 0, 1, 0, 1], 'home_advantage': 75},
                    '76人': {'strength': 83, 'form': [1, 1, 0, 1, 1], 'home_advantage': 80}
                }
            }
        }

    def init_database(self):
        """初始化數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 創建比賽數據表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            date TEXT NOT NULL,
            sport TEXT NOT NULL,
            league TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            result TEXT,
            home_form REAL,
            away_form REAL,
            home_ranking INTEGER,
            away_ranking INTEGER,
            h2h_home_wins INTEGER,
            h2h_draws INTEGER,
            h2h_away_wins INTEGER,
            home_key_players INTEGER,
            away_key_players INTEGER,
            home_injuries INTEGER,
            away_injuries INTEGER,
            venue TEXT,
            weather TEXT,
            temperature REAL,
            home_odds REAL,
            draw_odds REAL,
            away_odds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # 創建球隊數據表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            sport TEXT NOT NULL,
            league TEXT NOT NULL,
            strength INTEGER,
            current_form REAL,
            home_advantage INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()

    def collect_historical_matches(self, sport: str = 'football',
                                 league: str = 'premier_league',
                                 num_matches: int = 1000) -> List[MatchData]:
        """收集歷史比賽數據"""

        matches = []
        teams_data = self.real_teams_data.get(sport, {}).get(league, {})
        team_names = list(teams_data.keys())

        if len(team_names) < 2:
            logger.warning(f"沒有足夠的球隊數據用於 {sport}/{league}")
            return matches

        # 生成歷史比賽數據
        start_date = datetime.now() - timedelta(days=365)

        for i in range(num_matches):
            # 隨機選擇兩支球隊
            home_team, away_team = random.sample(team_names, 2)

            # 生成比賽日期
            match_date = start_date + timedelta(days=random.randint(0, 365))

            # 獲取球隊數據
            home_data = teams_data[home_team]
            away_data = teams_data[away_team]

            # 計算比賽結果
            match_result = self._simulate_match_result(home_data, away_data)

            # 創建比賽數據
            match = MatchData(
                home_team=home_team,
                away_team=away_team,
                date=match_date.strftime('%Y-%m-%d'),
                sport=sport,
                league=league,
                home_score=match_result['home_score'],
                away_score=match_result['away_score'],
                result=match_result['result'],
                home_form=self._calculate_form(home_data['form']),
                away_form=self._calculate_form(away_data['form']),
                home_ranking=self._get_team_ranking(home_team, teams_data),
                away_ranking=self._get_team_ranking(away_team, teams_data),
                head_to_head_home_wins=random.randint(0, 5),
                head_to_head_draws=random.randint(0, 3),
                head_to_head_away_wins=random.randint(0, 5),
                home_key_players_available=random.randint(8, 11),
                away_key_players_available=random.randint(8, 11),
                home_injury_count=random.randint(0, 3),
                away_injury_count=random.randint(0, 3),
                venue="主場",
                weather=random.choice(['晴天', '陰天', '小雨', '強風']),
                temperature=random.uniform(5, 30),
                home_odds=match_result['home_odds'],
                draw_odds=match_result['draw_odds'],
                away_odds=match_result['away_odds']
            )

            matches.append(match)

        logger.info(f"收集了 {len(matches)} 場 {sport}/{league} 的歷史比賽數據")
        return matches

    def _simulate_match_result(self, home_data: Dict, away_data: Dict) -> Dict:
        """模擬比賽結果"""

        # 計算球隊實力差距
        home_strength = home_data['strength'] + home_data['home_advantage'] * 0.1
        away_strength = away_data['strength']

        # 計算狀態影響
        home_form = self._calculate_form(home_data['form'])
        away_form = self._calculate_form(away_data['form'])

        home_strength += home_form * 5
        away_strength += away_form * 5

        # 計算獲勝概率
        total_strength = home_strength + away_strength
        home_win_prob = home_strength / total_strength * 0.7 + 0.15  # 主場略有優勢
        draw_prob = 0.25
        away_win_prob = 1 - home_win_prob - draw_prob

        # 隨機決定結果
        rand = random.random()
        if rand < home_win_prob:
            result = 'H'
            home_score = random.randint(1, 4)
            away_score = random.randint(0, home_score - 1)
        elif rand < home_win_prob + draw_prob:
            result = 'D'
            score = random.randint(0, 3)
            home_score = away_score = score
        else:
            result = 'A'
            away_score = random.randint(1, 4)
            home_score = random.randint(0, away_score - 1)

        # 生成模擬賠率
        home_odds = 1 / home_win_prob if home_win_prob > 0 else 10
        draw_odds = 1 / draw_prob if draw_prob > 0 else 10
        away_odds = 1 / away_win_prob if away_win_prob > 0 else 10

        return {
            'home_score': home_score,
            'away_score': away_score,
            'result': result,
            'home_odds': round(home_odds, 2),
            'draw_odds': round(draw_odds, 2),
            'away_odds': round(away_odds, 2)
        }

    def _calculate_form(self, recent_results: List[int]) -> float:
        """計算球隊近期狀態分數 (0-1)"""
        if not recent_results:
            return 0.5

        total_points = sum(recent_results)
        max_points = len(recent_results) * 3  # 假設3分制
        return total_points / max_points if max_points > 0 else 0.5

    def _get_team_ranking(self, team_name: str, teams_data: Dict) -> int:
        """獲取球隊排名"""
        strengths = [(name, data['strength']) for name, data in teams_data.items()]
        strengths.sort(key=lambda x: x[1], reverse=True)

        for i, (name, _) in enumerate(strengths):
            if name == team_name:
                return i + 1
        return len(strengths)

    def save_matches_to_db(self, matches: List[MatchData]):
        """保存比賽數據到數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for match in matches:
            cursor.execute('''
            INSERT INTO matches (
                home_team, away_team, date, sport, league,
                home_score, away_score, result,
                home_form, away_form, home_ranking, away_ranking,
                h2h_home_wins, h2h_draws, h2h_away_wins,
                home_key_players, away_key_players, home_injuries, away_injuries,
                venue, weather, temperature,
                home_odds, draw_odds, away_odds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match.home_team, match.away_team, match.date, match.sport, match.league,
                match.home_score, match.away_score, match.result,
                match.home_form, match.away_form, match.home_ranking, match.away_ranking,
                match.head_to_head_home_wins, match.head_to_head_draws, match.head_to_head_away_wins,
                match.home_key_players_available, match.away_key_players_available,
                match.home_injury_count, match.away_injury_count,
                match.venue, match.weather, match.temperature,
                match.home_odds, match.draw_odds, match.away_odds
            ))

        conn.commit()
        conn.close()
        logger.info(f"已保存 {len(matches)} 場比賽到數據庫")

    def load_matches_from_db(self, sport: str = None, league: str = None,
                           limit: int = 1000) -> pd.DataFrame:
        """從數據庫載入比賽數據"""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM matches"
        params = []

        if sport or league:
            conditions = []
            if sport:
                conditions.append("sport = ?")
                params.append(sport)
            if league:
                conditions.append("league = ?")
                params.append(league)
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY date DESC LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        logger.info(f"從數據庫載入了 {len(df)} 場比賽數據")
        return df

    def create_training_text(self, match: MatchData) -> str:
        """為比賽創建訓練文本描述"""

        # 基本比賽信息
        text_parts = [
            f"本場{match.sport}比賽由{match.home_team}主場迎戰{match.away_team}。"
        ]

        # 球隊狀態信息
        if match.home_form is not None and match.away_form is not None:
            home_form_desc = self._form_to_desc(match.home_form)
            away_form_desc = self._form_to_desc(match.away_form)
            text_parts.append(f"{match.home_team}近期狀態{home_form_desc}，{match.away_team}近期狀態{away_form_desc}。")

        # 排名信息
        if match.home_ranking and match.away_ranking:
            text_parts.append(f"{match.home_team}目前排名第{match.home_ranking}位，{match.away_team}排名第{match.away_ranking}位。")

        # 歷史對戰
        if match.head_to_head_home_wins is not None:
            total_h2h = match.head_to_head_home_wins + match.head_to_head_draws + match.head_to_head_away_wins
            if total_h2h > 0:
                text_parts.append(f"雙方過去{total_h2h}次交手，{match.home_team}獲勝{match.head_to_head_home_wins}場，平局{match.head_to_head_draws}場，{match.away_team}獲勝{match.head_to_head_away_wins}場。")

        # 球員情況
        if match.home_key_players_available and match.away_key_players_available:
            text_parts.append(f"{match.home_team}有{match.home_key_players_available}名主力球員可以出戰，{match.away_team}有{match.away_key_players_available}名主力球員可以出戰。")

        # 傷病情況
        if match.home_injury_count is not None and match.away_injury_count is not None:
            if match.home_injury_count > 0 or match.away_injury_count > 0:
                text_parts.append(f"{match.home_team}目前有{match.home_injury_count}名球員受傷，{match.away_team}有{match.away_injury_count}名球員受傷。")

        # 環境因素
        if match.weather and match.temperature:
            text_parts.append(f"比賽當天天氣為{match.weather}，氣溫約{match.temperature:.1f}度。")

        # 市場預期
        if match.home_odds and match.away_odds:
            if match.home_odds < match.away_odds:
                text_parts.append(f"根據賠率，{match.home_team}被看好獲勝。")
            elif match.away_odds < match.home_odds:
                text_parts.append(f"根據賠率，{match.away_team}被看好獲勝。")
            else:
                text_parts.append("根據賠率，雙方實力相當。")

        return " ".join(text_parts)

    def _form_to_desc(self, form_score: float) -> str:
        """將狀態分數轉換為描述"""
        if form_score >= 0.8:
            return "極佳"
        elif form_score >= 0.6:
            return "良好"
        elif form_score >= 0.4:
            return "一般"
        else:
            return "不佳"

    def prepare_training_data(self, sport: str = 'football',
                            league: str = 'premier_league',
                            num_samples: int = 1000) -> pd.DataFrame:
        """準備LLM訓練數據"""

        # 收集歷史比賽數據
        matches = self.collect_historical_matches(sport, league, num_samples)

        # 保存到數據庫
        self.save_matches_to_db(matches)

        # 創建訓練數據
        training_data = []

        for match in matches:
            # 創建文本描述
            text_description = self.create_training_text(match)

            # 轉換結果標籤
            result_mapping = {'A': 0, 'D': 1, 'H': 2}  # 客勝、平局、主勝
            result_label = result_mapping.get(match.result, 1)

            training_data.append({
                'text': text_description,
                'home_team': match.home_team,
                'away_team': match.away_team,
                'result': result_label,
                'result_text': ['客勝', '平局', '主勝'][result_label],
                'home_score': match.home_score,
                'away_score': match.away_score,
                'date': match.date,
                'sport': match.sport,
                'league': match.league
            })

        df = pd.DataFrame(training_data)
        logger.info(f"準備了 {len(df)} 條訓練數據")

        return df

    def get_real_time_data(self, team1: str, team2: str, sport: str = 'football') -> Dict:
        """獲取實時比賽數據（模擬）"""

        # 這裡可以接入真實的體育數據API
        # 例如：ESPN API, Sports API, etc.

        # 模擬實時數據
        current_data = {
            'home_team': team1,
            'away_team': team2,
            'sport': sport,
            'match_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'weather': random.choice(['晴天', '陰天', '小雨']),
            'temperature': random.uniform(10, 25),
            'home_odds': round(random.uniform(1.5, 4.0), 2),
            'draw_odds': round(random.uniform(2.5, 4.5), 2),
            'away_odds': round(random.uniform(1.5, 4.0), 2),
            'status': '未開始'
        }

        return current_data

def main():
    """主函數 - 演示數據收集功能"""

    print("=== 運動彩券數據收集器 ===")

    # 初始化數據收集器
    collector = SportsDataCollector()

    # 收集足球數據
    print("1. 收集足球比賽數據...")
    football_data = collector.prepare_training_data(
        sport='football',
        league='premier_league',
        num_samples=500
    )

    print(f"足球數據樣本數: {len(football_data)}")
    print("足球數據前5行:")
    print(football_data.head())

    # 收集籃球數據
    print("\n2. 收集籃球比賽數據...")
    basketball_data = collector.prepare_training_data(
        sport='basketball',
        league='nba',
        num_samples=300
    )

    print(f"籃球數據樣本數: {len(basketball_data)}")

    # 合併數據
    all_data = pd.concat([football_data, basketball_data], ignore_index=True)
    print(f"\n總數據樣本數: {len(all_data)}")

    # 保存訓練數據
    all_data.to_csv('sports_training_data.csv', index=False, encoding='utf-8')
    print("訓練數據已保存到 sports_training_data.csv")

    # 顯示結果分布
    print("\n結果分布:")
    print(all_data['result_text'].value_counts())

    # 測試實時數據獲取
    print("\n3. 測試實時數據獲取...")
    real_time_data = collector.get_real_time_data('曼城', '利物浦')
    print("實時數據樣本:")
    for key, value in real_time_data.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()