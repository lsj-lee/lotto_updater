#!/bin/bash

# [0] 중복 실행 방지 (하루 딱 한 번만)
TODAY=$(date +%Y-%m-%d)
TRACKER_FILE="/Users/lsj/Desktop/로또/run_tracker.txt"

if [ -f "$TRACKER_FILE" ]; then
    LAST_RUN=$(cat "$TRACKER_FILE")
    if [ "$LAST_RUN" == "$TODAY" ]; then
        exit 0
    fi
fi

echo "========================================"
echo "☀️ 일요일 로또 AI 통합 자율 주행 시작"
echo "시작 시간: $(date)"
echo "========================================"

cd /Users/lsj/Desktop/로또
source /Users/lsj/Desktop/로또/.venv/bin/activate

# ▶️ 1단계: 9차원 라이트급 훈련 (lotto_brain.py)
echo "🚀 [STEP 1] 8가지 시야 훈련 시작..."
python lotto_brain.py
sleep 60 # M5 칩 냉각

# ▶️ 2단계: 앙상블 번호 예측 및 시트 기록 (lotto_predict.py)
echo "🚀 [STEP 2] 앙상블 예측 및 결과 전송 시작..."
python lotto_predict.py
sleep 10 # 통신 대기

# ▶️ 3단계: 최종 데이터 심화 분석 (로또_분석.py)
# 상진 님이 요청하신 '순서대로'의 마침표입니다.
echo "🚀 [STEP 3] 최종 데이터 심화 분석 시작..."
python 로또_분석.py

# [2] 완료 도장 찍기
echo "$TODAY" > "$TRACKER_FILE"

echo "========================================"
echo "✅ 모든 순차 작업이 완료되었습니다!"
echo "종료 시간: $(date)"
echo "========================================"