# Prediction Market AI — Agent Context

## 프로젝트 개요
예측시장 차익거래 + AI 미스프라이싱 탐지 자동매매 봇

## 현재 구조
- scanner.py: 멀티플랫폼 스캐너 (Polymarket, Limitless, Predict.fun)
  - Polymarket: gamma-api.polymarket.com (공개 API)
  - Limitless: api.limitless.exchange (공개 API)
  - Predict.fun: api-testnet.predict.fun (공개 API)
  - 5분 간격 스캔, systemd 서비스로 24/7 실행
  - YES+NO < $0.98 탐지 (intra-platform)
  - 크로스 플랫폼 fuzzy matching + 가격 비교

## 코딩 규칙
- Python 3.9+
- 타입 힌트 필수
- 함수/클래스에 docstring 필수
- 에러 핸들링 포괄적으로
- 로깅은 logging 모듈 사용
- 설정은 환경변수 또는 config.json
- 테스트: pytest 사용

## API 키 참고
- Anthropic API: ANTHROPIC_API_KEY 환경변수
- Polymarket/Limitless/Predict.fun: 공개 API (키 불필요)
- Smart Router (로컬): http://localhost:4001/v1/messages

## 파일 구조 규칙
- 모듈별 분리 (scanner.py, analyzer.py, trader.py 등)
- 공통 유틸: utils.py
- 설정: config.py
- 테스트: tests/ 디렉토리
