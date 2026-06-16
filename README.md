# Earpiece

실시간 영어 STT + 한글 번역 — 완전 로컬, API 불필요

---

## 파일 구조

```
earpiece.py          ← 메인 스크립트
initialize.sh        ← 최초 환경 세팅 (OS/CPU/CUDA 자동 감지)
contexts/
  example.md         ← 미팅 컨텍스트 템플릿
README.md
```

---

## 설치

```bash
bash initialize.sh
```

OS, CPU 아키텍처, CUDA 유무를 자동으로 감지하여 적합한 백엔드를 설치합니다.

| 환경 | STT 백엔드 |
|------|-----------|
| Mac Apple Silicon | mlx-whisper (Metal 가속) |
| Linux CPU | faster-whisper (int8) |
| Linux CUDA | faster-whisper (float16, GPU) |

---

## 오디오 설정 (Mac — Zoom/Meet 캡처)

1. **BlackHole 2ch 설치** — [existential.audio/blackhole](https://existential.audio/blackhole/) → `.pkg` 인스톨러 실행
2. **오디오 MIDI 설정** → `+` → Multi-Output 장치 생성 → BlackHole 2ch ✅ + 스피커 ✅ (드리프트 보정: BlackHole에 체크)
3. **Zoom** → 설정 → 오디오 → 스피커: Multi-Output Device

---

## 실행

```bash
source .venv/bin/activate

# 오디오 장치 확인
python earpiece.py --list-devices

# BlackHole로 Zoom 캡처 (Mac)
python earpiece.py --device blackhole

# 미팅 컨텍스트(키워드) 포함
python earpiece.py --device blackhole --context contexts/example.md

# 정확도 높이기
python earpiece.py --device blackhole --model base
```

---

## 미팅 컨텍스트

`contexts/example.md` 를 복사해서 미팅별로 작성합니다.

```bash
cp contexts/example.md contexts/2026-06-16_meeting.md
# 참가자, 키워드, 아젠다 작성 후 실행 시 --context 로 지정
```

Keywords 섹션에 등록한 고유명사/약어가 Whisper 전사 정확도를 높이는 데 사용됩니다.

---

## 출력 예시

```
[17:06:02] Can you walk us through the deployment?
          → 배포 과정을 안내해 주시겠어요?
[17:06:05] Thank you.
          → 감사합니다.
```

---

## 모델 옵션

| 모델 | 크기 | 속도 |
|------|------|------|
| tiny (기본) | 74MB | 매우 빠름 |
| base | 142MB | 빠름 |
| small | 466MB | 보통 |
| medium | 1.5GB | 느림 |

---

## 다음 단계

- [ ] 자막 오버레이 UI (맥 화면 하단)
- [ ] 미팅 후 요약 자동 생성
- [ ] Obsidian vault에 트랜스크립트 자동 저장
