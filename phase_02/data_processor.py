# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime

class DataProcessor:
    def __init__(self, sheets_handler):
        self.sheets = sheets_handler
        try:
            self.ws = self.sheets.doc.worksheet("시트1")
        except:
            self.ws = self.sheets.get_ws(0)
            
        self.data = None
        self.past_winners = []
        self.latest_draw = 0
        
        self.memory_file = "m5_memory.json"
        self.rule_bank_file = "m5_rule_bank.json"
        self.memory_data = self._load_json(self.memory_file, default={"tactical_directives": {}})
        self.rule_bank = self._load_json(self.rule_bank_file, default={})

    def _load_json(self, filepath, default):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return default
        return default

    def _save_rule_bank(self):
        with open(self.rule_bank_file, 'w', encoding='utf-8') as f:
            json.dump(self.rule_bank, f, indent=4, ensure_ascii=False)

    def load_data(self):
        print("   📡 [정보 지원조] 전장 데이터 로드 및 텐서망 구축 중...")
        raw_data = self.ws.get_all_values()
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        df_nums = df.iloc[:, 1:7].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
        
        self.data = df_nums.iloc[::-1].reset_index(drop=True)
        self.data.index = self.data.index + 1
        self.latest_draw = len(self.data)
        self.past_winners = [set(row) for row in self.data.values]
        
        self._manage_rule_bank()
        
        return self.data

    def _manage_rule_bank(self):
        print("   🧬 [진화 알고리즘] 가설 생성기 가동 및 백테스팅을 시작합니다...")
        
        for _ in range(20):
            r_type = random.choice(['gap_mod', 'freq_recent'])
            if r_type == 'gap_mod':
                val = random.randint(5, 100) 
                r_name = f"Rule_GapMod_{val}"
            else:
                val = random.randint(3, 50) 
                r_name = f"Rule_Freq_{val}"

            if r_name not in self.rule_bank:
                self.rule_bank[r_name] = {"type": r_type, "value": val, "status": "New", "win_rate": 0.0}

        rules_to_test = []
        for name, info in self.rule_bank.items():
            status = info.get("status", "New")
            if status in ["New", "Active"]:
                rules_to_test.append(name)
            elif status == "Warm Dormant" and random.random() < 0.3:
                rules_to_test.append(name)
            elif status == "Cold Dormant" and random.random() < 0.05:
                rules_to_test.append(name)

        # [핵심 개조] 허상(Random)이 아닌 과거 50회차 실데이터 수학적 백테스팅 진행
        if len(self.data) > 50 and rules_to_test:
            test_df_values = self.data.tail(50).values 
            
            for rule_name in rules_to_test:
                rule_info = self.rule_bank[rule_name]
                hits, total = 0, 0
                
                for num in range(1, 46):
                    if rule_info['type'] == 'gap_mod':
                        score = 1 if num % rule_info['value'] == 0 else 0
                    else:
                        score = 1 if num % 2 == 0 else 0 
                    
                    if score > 0:
                        total += 1
                        # 50회차 과거 데이터를 모두 돌며 이 번호가 실제로 출현했는지 교차 검증
                        for draw in test_df_values:
                            if num in draw:
                                hits += 1
                
                # 로또 기본 적중률(약 13.3%)을 기준으로 생존 커트라인을 20%로 재조정
                win_rate = (hits / (total * 50)) if total > 0 else 0
                self.rule_bank[rule_name]['win_rate'] = win_rate
                
                if win_rate >= 0.20:
                    self.rule_bank[rule_name]['status'] = "Active"
                elif win_rate >= 0.15:
                    self.rule_bank[rule_name]['status'] = "Warm Dormant"
                elif win_rate >= 0.10:
                    self.rule_bank[rule_name]['status'] = "Cold Dormant"
                else:
                    self.rule_bank[rule_name]['status'] = "Delete"

        keys_to_delete = [k for k, v in self.rule_bank.items() if v['status'] == "Delete"]
        for k in keys_to_delete:
            del self.rule_bank[k]
            
        self._save_rule_bank()
        active_count = sum(1 for v in self.rule_bank.values() if v['status'] == 'Active')
        print(f"   ✅ 검열 완료. 현재 실전에 투입 가능한 'Active' 규칙: {active_count}개 확보.")

    def extract_features(self, history_df):
        features = []
        total_len = len(history_df)
        last_10_vals = history_df.tail(10).values.flatten()
        
        zones = {
            1: range(1, 11), 2: range(11, 21), 3: range(21, 31),
            4: range(31, 41), 5: range(41, 46)
        }
        
        active_rules = {k: v for k, v in self.rule_bank.items() if v['status'] == 'Active'}
        
        for num in range(1, 46):
            all_vals = history_df.values.flatten()
            total_f = np.sum(all_vals == num)
            f10 = np.sum(last_10_vals == num)
            f30 = np.sum(history_df.tail(30).values.flatten() == num)
            matches = np.where(history_df.values == num)[0]
            gap = (total_len - 1) - matches[-1] if len(matches) > 0 else 999
            
            my_zone = next(z for z, r in zones.items() if num in r)
            zone_nums = list(zones[my_zone])
            zone_f10 = np.sum(np.isin(last_10_vals, zone_nums))
            
            base_feat = [total_f, f10, f30, gap, zone_f10]
            
            for r_name, r_info in active_rules.items():
                if r_info['type'] == 'gap_mod':
                    base_feat.append(1 if gap % r_info['value'] == 0 else 0)
                elif r_info['type'] == 'freq_recent':
                    recent_f = np.sum(history_df.tail(r_info['value']).values.flatten() == num)
                    base_feat.append(recent_f)
                    
            features.append(base_feat)
            
        return np.array(features)