# Neural D Meeting Transcriber

실시간 영어 STT + 한글 번역 — 완전 로컬, API 불필요

---

## 파일 구조

```
transcribe_local.py     ← 메인 스크립트 (Mac/Linux 공통)
debug_audio.py          ← 오디오 입력 레벨 디버거
template_meeting.md     ← 미팅별 복사해서 편집

setup_mac.sh            ← Mac (Apple Silicon) 초기 설치
setup_linux.sh          ← Linux bare metal 초기 설치
Dockerfile              ← Docker 이미지 빌드

requirements.txt        ← 공통 의존성
requirements_mac.txt    ← Mac 전용 (mlx-whisper)
requirements_linux.txt  ← Linux 전용 (faster-whisper)
```

---

## 플랫폼별 설치

### Mac (Apple Silicon)

```bash
bash setup_mac.sh
```

- 가상환경 생성 (`.venv`)
- mlx-whisper + argostranslate EN→KO 모델 사전 다운로드

### Linux (bare metal)

```bash
bash setup_linux.sh
```

- PortAudio, Python 패키지 설치
- faster-whisper + argostranslate EN→KO 모델 사전 다운로드

### Docker

```bash
docker build -t neural-d .

# 마이크 입력
docker run --rm -it --device /dev/snd neural-d

# PulseAudio 시스템 오디오 캡처
docker run --rm -it \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  neural-d --device 0
```

---

## 오디오 설정 (Mac — Zoom/Meet 캡처)

1. **BlackHole 2ch 설치** — [existential.audio/blackhole](https://existential.audio/blackhole/) → `.pkg` 인스톨러 실행
2. **오디오 MIDI 설정** → `+` → Multi-Output 장치 생성 → BlackHole 2ch ✅ + 스피커 ✅ (드리프트 보정: BlackHole에 체크)
3. **Zoom** → 설정 → 오디오 → 스피커: Multi-Output Device

---

## 실행

```bash
source .venv/bin/activate   # Linux: 동일

# 오디오 장치 확인
python transcribe_local.py --list-devices

# BlackHole로 Zoom 캡처 (Mac)
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

```bash
python debug_audio.py --device blackhole   # Mac
python debug_audio.py --device 0           # Linux/Docker
```

레벨 바가 움직이면 정상. 무음이면 Zoom 스피커 출력이 Multi-Output Device로 설정됐는지 확인.

---

## 모델 옵션

| 모델 | 크기 | Mac (mlx) | Linux (faster) |
|------|------|-----------|----------------|
| tiny (기본) | 74MB | 매우 빠름 | 빠름 |
| base | 142MB | 빠름 | 보통 |
| small | 466MB | 보통 | 느림 |
| medium | 1.5GB | 느림 | 매우 느림 |

---

## 다음 단계

- [ ] 자막 오버레이 UI (맥 화면 하단)
- [ ] Zoom/Meet 오디오 자동 캡처 설정 자동화
- [ ] 미팅 후 요약 자동 생성
- [ ] Obsidian vault에 트랜스크립트 자동 저장
