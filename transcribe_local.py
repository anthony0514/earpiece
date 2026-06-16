#!/usr/bin/env python3
"""
Neural D — Local Meeting Transcriber
=====================================
BlackHole → WebRTC VAD → mlx-whisper (완전 오프라인)

Setup:
  pip install sounddevice webrtcvad-wheels mlx-whisper numpy

Usage:
  python transcribe_local.py                         # 기본 마이크
  python transcribe_local.py --device blackhole      # Zoom/Meet 오디오
  python transcribe_local.py --list-devices
  python transcribe_local.py --meeting template_meeting.md --device blackhole
  python transcribe_local.py --model base            # 정확도↑ (기본: tiny)
  python transcribe_local.py --lang ko               # 한국어
"""

import sys
import queue
import argparse
import collections
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

# ── 의존성 체크 ────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("❌ pip install sounddevice numpy"); sys.exit(1)

try:
    import webrtcvad
except ImportError:
    print("❌ pip install webrtcvad-wheels"); sys.exit(1)

try:
    import mlx_whisper
except ImportError:
    print("❌ pip install mlx-whisper"); sys.exit(1)


# ── 컬러 ──────────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GRAY   = "\033[90m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RED    = "\033[91m"


# ── 오디오 장치 ───────────────────────────────────────────────────────────────
def list_devices():
    print(f"\n{'─'*65}")
    print(f"{'ID':>3}  {'이름':<45} {'입력ch':>6}")
    print(f"{'─'*65}")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            marker = " ◀ default" if i == sd.default.device[0] else ""
            print(f"{i:>3}  {d['name']:<45} {int(d['max_input_channels']):>6}{marker}")
    print(f"{'─'*65}\n")


def find_blackhole() -> Optional[int]:
    for i, d in enumerate(sd.query_devices()):
        if "blackhole" in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    return None


def resolve_device(arg: Optional[str]) -> Optional[int]:
    if arg is None:
        return None
    if arg.lower() == "blackhole":
        dev_id = find_blackhole()
        if dev_id is None:
            print(f"{C.RED}❌ BlackHole 장치 없음. sudo killall coreaudiod 후 재시도{C.RESET}")
            sys.exit(1)
        return dev_id
    return int(arg)


# ── 미팅 .md 파싱 ─────────────────────────────────────────────────────────────
def parse_meeting_md(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    ctx = {"keywords": [], "context": ""}
    kw = re.search(r"## Keywords.*?```(.*?)```", text, re.DOTALL)
    if kw:
        ctx["keywords"] = [k.strip() for k in re.split(r"[,\n]+", kw.group(1)) if k.strip()]
    cx = re.search(r"## Company Context(.*?)(^##|\Z)", text, re.DOTALL | re.MULTILINE)
    if cx:
        ctx["context"] = cx.group(1).strip()
    return ctx


# ── VAD + 버퍼링 ──────────────────────────────────────────────────────────────
RATE          = 16000
FRAME_MS      = 30                        # WebRTC VAD 프레임 단위
FRAME_SAMPLES = RATE * FRAME_MS // 1000  # 480 samples
FRAME_BYTES   = FRAME_SAMPLES * 2        # int16 = 2 bytes

PREROLL_FRAMES  = 10   # 발화 시작 전 패딩 (~300ms)
SILENCE_FRAMES  = 20   # 이 프레임 수 이상 무음이면 전사 (~600ms)
MIN_SPEECH_FRAMES = 5  # 이 이하는 노이즈로 무시


class VADBuffer:
    """WebRTC VAD로 발화 구간 감지 → 버퍼 반환"""

    def __init__(self, aggressiveness: int = 2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.pre_roll = collections.deque(maxlen=PREROLL_FRAMES)
        self.speech_buf: list[bytes] = []
        self.silence_count = 0
        self.speaking = False
        self._leftover = b""  # 프레임 정렬용

    def feed(self, pcm_bytes: bytes) -> Optional[np.ndarray]:
        """
        int16 PCM bytes를 받아서 발화 구간이 완성되면 float32 ndarray 반환.
        아직 진행 중이면 None 반환.
        """
        data = self._leftover + pcm_bytes
        self._leftover = b""
        result = None

        while len(data) >= FRAME_BYTES:
            frame = data[:FRAME_BYTES]
            data  = data[FRAME_BYTES:]

            is_speech = self.vad.is_speech(frame, RATE)

            if is_speech:
                if not self.speaking:
                    self.speaking = True
                    self.speech_buf = list(self.pre_roll)  # pre-roll 붙이기
                self.speech_buf.append(frame)
                self.silence_count = 0
            else:
                if self.speaking:
                    self.speech_buf.append(frame)
                    self.silence_count += 1
                    if self.silence_count >= SILENCE_FRAMES:
                        # 발화 종료 → 반환
                        if len(self.speech_buf) >= MIN_SPEECH_FRAMES:
                            raw = b"".join(self.speech_buf)
                            result = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                        self.speech_buf = []
                        self.silence_count = 0
                        self.speaking = False
                else:
                    self.pre_roll.append(frame)

        self._leftover = data
        return result


# ── Whisper 모델 이름 매핑 ─────────────────────────────────────────────────────
MODELS = {
    "tiny":   "mlx-community/whisper-tiny-mlx",
    "base":   "mlx-community/whisper-base-mlx",
    "small":  "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
}


# ── 로컬 번역 (argostranslate EN→KO) ─────────────────────────────────────────
def load_translator():
    """argostranslate EN→KO 패키지 설치. 첫 실행 시 ~100MB 다운로드."""
    try:
        from argostranslate import package, translate

        def _get_translation(langs):
            en = next((l for l in langs if l.code == "en"), None)
            ko = next((l for l in langs if l.code == "ko"), None)
            if en and ko:
                return en.get_translation(ko)
            return None

        # 이미 설치된 경우 바로 반환
        trans = _get_translation(translate.get_installed_languages())
        if trans:
            return trans

        # 패키지 인덱스 업데이트 & en→ko 설치
        print(f"{C.GRAY}  번역 패키지 다운로드 중... (~100MB){C.RESET}")
        package.update_package_index()
        available = package.get_available_packages()
        pkg = next((p for p in available if p.from_code == "en" and p.to_code == "ko"), None)
        if pkg is None:
            raise RuntimeError("en→ko 패키지를 찾을 수 없습니다")
        package.install_from_path(pkg.download())

        return _get_translation(translate.get_installed_languages())

    except Exception as e:
        print(f"{C.YELLOW}⚠  번역 로드 실패: {e}{C.RESET}")
        return None


def translate_to_korean(text: str, translator, ctx: dict) -> str:
    try:
        return translator.translate(text)
    except Exception:
        return ""


# ── 트랜스크립트 저장 ──────────────────────────────────────────────────────────
class TranscriptLog:
    def __init__(self):
        self.entries: list[dict] = []

    def add(self, en: str, ko: str = ""):
        self.entries.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "en": en,
            "ko": ko,
        })

    def save(self):
        if not self.entries:
            return
        path = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Meeting Transcript — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            for e in self.entries:
                f.write(f"[{e['time']}] {e['en']}\n")
                if e["ko"]:
                    f.write(f"          → {e['ko']}\n")
                f.write("\n")
        print(f"\n{C.CYAN}💾 저장됨: {path}{C.RESET}")


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run(device_id: Optional[int], model_key: str, ctx: dict):
    model_id = MODELS.get(model_key, MODELS["tiny"])
    dev_name = sd.query_devices(device_id)["name"] if device_id is not None else "기본 마이크"

    print(f"\n{C.CYAN}{'─'*60}{C.RESET}")
    print(f"{C.BOLD}🎙  Neural D — Local Transcriber{C.RESET}")
    print(f"{C.CYAN}{'─'*60}{C.RESET}")
    print(f"{C.GRAY}모델   : {model_key} ({model_id}){C.RESET}")
    print(f"{C.GRAY}장치   : {dev_name}{C.RESET}")
    print(f"{C.GRAY}언어   : en (고정){C.RESET}")
    if ctx["keywords"]:
        print(f"{C.GRAY}키워드 : {', '.join(ctx['keywords'][:8])}{C.RESET}")
    print(f"{C.CYAN}{'─'*60}{C.RESET}")
    print(f"{C.YELLOW}모델 로딩 중... (첫 실행 시 다운로드){C.RESET}")

    dummy = np.zeros(RATE, dtype=np.float32)
    mlx_whisper.transcribe(dummy, path_or_hf_repo=model_id, verbose=False)

    print(f"{C.YELLOW}번역 모델 로딩 중... (argostranslate EN→KO){C.RESET}")
    import logging, io, contextlib, os
    logging.getLogger("stanza").setLevel(logging.ERROR)
    # stanza가 OS 레벨 fd=2에 직접 쓰므로 dup2로 억제
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_stderr = os.dup(2)
    os.dup2(_null_fd, 2)
    os.close(_null_fd)
    try:
        translator = load_translator()
        if translator:
            translator.translate("warming up")  # stanza 초기화 완료
    finally:
        os.dup2(_saved_stderr, 2)
        os.close(_saved_stderr)
    has_translator = translator is not None
    print(f"{C.GRAY}번역   : {'✅ argostranslate EN→KO (로컬)' if has_translator else '⏭  OFF'}{C.RESET}")
    print(f"{C.GREEN}✅ 준비 완료 — 말하세요 (Ctrl+C로 종료){C.RESET}\n")

    audio_q: queue.Queue[bytes] = queue.Queue()
    transcribe_q: queue.Queue = queue.Queue()
    translate_q: queue.Queue = queue.Queue()
    vad_buf = VADBuffer(aggressiveness=2)
    log = TranscriptLog()

    def audio_callback(indata, frames, time_info, status):
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        audio_q.put(pcm)

    def transcribe_worker():
        while True:
            audio = transcribe_q.get()
            if audio is None:
                break
            sys.stdout.write("\r" + " " * 70 + "\r")
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=model_id,
                language="en",
                verbose=False,
                initial_prompt=", ".join(ctx["keywords"]) if ctx["keywords"] else None,
            )
            text = result["text"].strip()
            if not text:
                continue
            ts = datetime.now().strftime("%H:%M:%S")
            # 영어 즉시 출력
            print(f"{C.GRAY}[{ts}]{C.RESET} {C.GREEN}{C.BOLD}{text}{C.RESET}")
            log.add(text)
            # 번역 큐에 넘기기
            if has_translator:
                translate_q.put((text, ts))

    def translate_worker():
        while True:
            item = translate_q.get()
            if item is None:
                break
            text, ts = item
            ko = translate_to_korean(text, translator, ctx)
            if ko:
                print(f"          {C.CYAN}→ {ko}{C.RESET}")
                for e in reversed(log.entries):
                    if e["en"] == text:
                        e["ko"] = ko
                        break

    t_transcribe = threading.Thread(target=transcribe_worker, daemon=True)
    t_translate  = threading.Thread(target=translate_worker,  daemon=True)
    t_transcribe.start()
    t_translate.start()

    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SAMPLES, callback=audio_callback,
                            device=device_id):
            while True:
                pcm = audio_q.get()
                audio_chunk = vad_buf.feed(pcm)
                if vad_buf.speaking:
                    elapsed = len(vad_buf.speech_buf) * FRAME_MS / 1000
                    sys.stdout.write(f"\r{C.GRAY}  🎤 발화 중 {elapsed:.1f}s ...{C.RESET}")
                    sys.stdout.flush()
                if audio_chunk is not None:
                    transcribe_q.put(audio_chunk)

    except KeyboardInterrupt:
        pass
    finally:
        transcribe_q.put(None)
        translate_q.put(None)
        t_transcribe.join(timeout=10)
        t_translate.join(timeout=10)
        log.save()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Neural D Local Transcriber")
    parser.add_argument("--device", "-d", default=None,
                        help="장치 ID 또는 'blackhole'")
    parser.add_argument("--model", "-m", default="tiny",
                        choices=list(MODELS.keys()),
                        help="Whisper 모델 크기 (기본: tiny)")
    parser.add_argument("--meeting", default=None,
                        help="미팅 컨텍스트 .md 파일 경로")
    parser.add_argument("--list-devices", action="store_true",
                        help="오디오 장치 목록 출력")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    ctx = {"keywords": [], "context": ""}
    if args.meeting:
        if not Path(args.meeting).exists():
            print(f"{C.RED}❌ 파일 없음: {args.meeting}{C.RESET}")
            sys.exit(1)
        ctx = parse_meeting_md(args.meeting)
        print(f"{C.GREEN}✅ 미팅 컨텍스트: 키워드 {len(ctx['keywords'])}개{C.RESET}")

    device_id = resolve_device(args.device)
    run(device_id, args.model, ctx)


if __name__ == "__main__":
    main()
