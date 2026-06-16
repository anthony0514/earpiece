#!/usr/bin/env python3
"""
BlackHole 오디오 입력 디버거
— Deepgram 없이 오디오 신호가 실제로 들어오는지 확인

Usage:
  python debug_audio.py                  # 장치 목록 출력
  python debug_audio.py --device blackhole  # BlackHole 레벨 모니터
  python debug_audio.py --device 3          # 장치 ID 직접 지정
"""

import sys
import time
import argparse
from typing import Optional

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("❌ pip install sounddevice numpy")
    sys.exit(1)


def list_devices():
    devices = sd.query_devices()
    print(f"\n{'─'*65}")
    print(f"{'ID':>3}  {'이름':<45} {'입력ch':>6}")
    print(f"{'─'*65}")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ◀ default" if i == sd.default.device[0] else ""
            print(f"{i:>3}  {d['name']:<45} {int(d['max_input_channels']):>6}{marker}")
    print(f"{'─'*65}\n")


def find_blackhole() -> Optional[int]:
    for i, d in enumerate(sd.query_devices()):
        if "blackhole" in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    return None


def level_monitor(device_id: Optional[int], duration: int = 30):
    if device_id is not None:
        dev = sd.query_devices(device_id)
        dev_name = dev["name"]
    else:
        dev_name = f"기본 장치 (ID {sd.default.device[0]})"

    print(f"\n장치: {dev_name}")
    print(f"Ctrl+C로 종료 | 소리가 들어오면 레벨 바가 움직입니다\n")

    RATE = 16000
    CHUNK = 2048

    def show_level(indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata ** 2)))
        db = 20 * np.log10(rms + 1e-9)
        # -60dB ~ 0dB 범위를 40칸 바로 표시
        bars = max(0, int((db + 60) / 60 * 40))
        bar = "█" * bars + "░" * (40 - bars)
        label = "🔊 신호있음" if rms > 0.001 else "🔇 무음    "
        sys.stdout.write(f"\r[{bar}] {db:6.1f}dB  {label}")
        sys.stdout.flush()

    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                            blocksize=CHUNK, callback=show_level,
                            device=device_id):
            time.sleep(duration)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        print("   --list-devices 로 장치 ID 다시 확인해보세요")
        return

    print("\n\n✅ 완료")


def main():
    parser = argparse.ArgumentParser(description="BlackHole 오디오 입력 디버거")
    parser.add_argument("--device", "-d", default=None,
                        help="장치 ID 또는 'blackhole'")
    parser.add_argument("--duration", type=int, default=60,
                        help="모니터링 시간(초), 기본 60")
    args = parser.parse_args()

    if args.device is None:
        list_devices()
        print("사용법: python debug_audio.py --device blackhole")
        return

    if args.device.lower() == "blackhole":
        dev_id = find_blackhole()
        if dev_id is None:
            print("❌ BlackHole 장치를 찾을 수 없습니다.")
            print("   sudo killall coreaudiod 후 재시도, 또는 --list-devices 확인")
            return
        print(f"✅ BlackHole 자동 감지: ID {dev_id}")
    else:
        try:
            dev_id = int(args.device)
        except ValueError:
            print("❌ 장치 ID는 숫자 또는 'blackhole'")
            return

    level_monitor(dev_id, args.duration)


if __name__ == "__main__":
    main()
