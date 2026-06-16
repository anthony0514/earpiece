#!/bin/bash
# Neural D Meeting Transcriber — Linux Setup (bare metal)
# Ubuntu 22.04+ / Debian 12+

set -e

echo "────────────────────────────────────────"
echo "  Neural D Meeting Transcriber — Linux Setup"
echo "────────────────────────────────────────"

# Python 확인
if ! command -v python3 &>/dev/null; then
  echo "❌ python3가 없습니다."
  echo "   sudo apt install python3 python3-venv python3-pip"
  exit 1
fi
echo "✅ Python: $(python3 --version)"

# PortAudio (sounddevice 의존성)
if ! dpkg -s portaudio19-dev &>/dev/null 2>&1; then
  echo "→ PortAudio 설치 중..."
  sudo apt-get install -y portaudio19-dev libsndfile1
fi
echo "✅ PortAudio 설치됨"

# 가상환경
if [ ! -d ".venv" ]; then
  echo "→ 가상환경 생성 중..."
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "✅ 가상환경 활성화"

# 패키지 설치
echo "→ 패키지 설치 중..."
pip install -q --upgrade pip
pip install -q -r requirements_linux.txt
echo "✅ 패키지 설치 완료"

# faster-whisper 모델 다운로드 (tiny)
echo "→ Whisper 모델 다운로드 중... (faster-whisper tiny, ~74MB)"
python3 - <<'EOF'
from faster_whisper import WhisperModel
import sys
try:
    WhisperModel("tiny", device="cpu", compute_type="int8")
    print("✅ Whisper 모델 준비 완료")
except Exception as e:
    print(f"❌ Whisper 모델 다운로드 실패: {e}")
    sys.exit(1)
EOF

# argostranslate EN→KO 패키지 다운로드
echo "→ 번역 패키지 다운로드 중... (EN→KO, ~100MB)"
python3 - <<'EOF'
import sys
from argostranslate import package, translate

installed = translate.get_installed_languages()
en = next((l for l in installed if l.code == "en"), None)
ko = next((l for l in installed if l.code == "ko"), None)
if en and ko and en.get_translation(ko):
    print("✅ 번역 패키지 이미 설치됨")
    sys.exit(0)

package.update_package_index()
available = package.get_available_packages()
pkg = next((p for p in available if p.from_code == "en" and p.to_code == "ko"), None)
if not pkg:
    print("❌ EN→KO 패키지를 찾을 수 없습니다")
    sys.exit(1)
package.install_from_path(pkg.download())
print("✅ 번역 패키지 설치 완료")
EOF

echo ""
echo "────────────────────────────────────────"
echo "  ✅ 설치 완료!"
echo ""
echo "  실행 방법:"
echo "  source .venv/bin/activate"
echo "  python transcribe_local.py --device 0"
echo ""
echo "  ※ PulseAudio 시스템 오디오 캡처:"
echo "  pactl load-module module-loopback"
echo "────────────────────────────────────────"
