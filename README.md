# Neural D Meeting Transcriber

실시간 영어 STT + 한글 번역 — 완전 로컬, API 불필요

---

## 파일 구조

```
transcribe_local.py     ← 메인 스크립트
debug_audio.py          ← 오디오 입력 디버거
template_meeting.md     ← 미팅별 복사해서 편집
requirements.txt
```

---

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 첫 실행 시 mlx-whisper(~74MB)와 argostranslate EN→KO 패키지(~100MB)가 자동 다운로드됩니다.

---

## 오디오 설정 (Zoom/Meet 캡처)

1. **BlackHole 2ch 설치** — [existential.audio/blackhole](https://existential.audio/blackhole/) → `.pkg` 인스톨러 실행
2. **Audio MIDI Setup** → `+` → Multi-Output 장치 생성 → BlackHole 2ch ✅ + 스피커 ✅ (드리프트 보정: BlackHole에 체크)
3. **Zoom** → 설정 → 오디오 → 스피커: Multi-Output Device

---

## 실행

```bash
source .venv/bin/activate

# 오디오 장치 확인
python transcribe_local.py --list-devices

# BlackHole로 Zoom 캡처
python transcribe_local.py --device blackhole

# 미팅 컨텍스트(키워드) 포함
python transcribe_local.py --device blackhole --meeting template_meeting.md

# 정확도 높이기 (모델 크기 업)
python transcribe_local.py --device blackhole --model base
```

---

## 출력 예시

```
[17:06:02] Can you walk us through the KEPCO deployment?
          → KEPCO 배포 과정을 안내해 주시겠어요?
[17:06:05] Thank you.
          → 감사합니다.
```

---

## 오디오 디버깅

BlackHole에 신호가 들어오는지 확인:

```bash
python debug_audio.py --device blackhole
```

레벨 바가 움직이면 정상. 무음이면 Zoom 스피커 출력이 Multi-Output Device로 설정됐는지 확인.

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
- [ ] Zoom/Meet 오디오 자동 캡처 설정 자동화
- [ ] 미팅 후 요약 자동 생성
- [ ] Obsidian vault에 트랜스크립트 자동 저장
