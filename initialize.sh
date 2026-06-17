#!/bin/bash
# Earpiece — Initialize
# OS/CPU/CUDA 자동 감지 후 환경 세팅 + 모델 다운로드까지 한번에

set -e

# ── 컬러 ──────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${RESET}"; }
info() { echo -e "${CYAN}→  $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠  $*${RESET}"; }
die()  { echo -e "${RED}❌ $*${RESET}"; exit 1; }

echo -e "\n${BOLD}────────────────────────────────────────${RESET}"
echo -e "${BOLD}  Earpiece — Initialize${RESET}"
echo -e "${BOLD}────────────────────────────────────────${RESET}\n"

# ── 환경 감지 ─────────────────────────────────────────────────────────────────
OS=$(uname -s)         # Darwin / Linux
ARCH=$(uname -m)       # arm64 / x86_64
IS_APPLE_SILICON=false
BACKEND="faster"       # default
DEVICE_FLAG="cpu"
COMPUTE_TYPE="int8"

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  IS_APPLE_SILICON=true
  BACKEND="mlx"
fi

# CUDA 감지 (Linux x86_64)
HAS_CUDA=false
if [ "$OS" = "Linux" ] && command -v nvidia-smi &>/dev/null; then
  CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  if [ -n "$CUDA_VER" ]; then
    HAS_CUDA=true
    DEVICE_FLAG="cuda"
    COMPUTE_TYPE="float16"
  fi
fi

echo -e "  OS      : $OS ($ARCH)"
echo -e "  Backend : $BACKEND"
if [ "$HAS_CUDA" = true ]; then
  echo -e "  CUDA    : ✅ (driver $CUDA_VER) — GPU 가속 활성화"
else
  echo -e "  CUDA    : ⏭  없음 — CPU 사용"
fi
echo ""

# ── Python 확인 ───────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || true)
[ -z "$PYTHON" ] && die "Python 3.9+ 이 필요합니다. https://python.org"
PY_VER=$($PYTHON --version 2>&1)
PY_OK=$($PYTHON -c "import sys; print(1 if sys.version_info >= (3,9) else 0)")
[ "$PY_OK" != "1" ] && die "Python 3.9+ 이 필요합니다. 현재: $PY_VER"
ok "Python: $PY_VER"

# ── 시스템 패키지 ────────────────────────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
  # tkinter: brew python-tk 버전이 Python 버전과 일치해야 함
  if ! $PYTHON -c "import tkinter" &>/dev/null 2>&1; then
    if command -v brew &>/dev/null; then
      PY_MINOR=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
      info "tkinter 설치 중... (brew install python-tk@${PY_MINOR})"
      brew install "python-tk@${PY_MINOR}" || warn "python-tk 설치 실패 — --gui 옵션 사용 불가"
    else
      warn "tkinter 없음 + brew 없음 — --gui 사용 시 https://brew.sh 설치 후 재시도"
    fi
  fi
fi

if [ "$OS" = "Linux" ]; then
  if command -v apt-get &>/dev/null; then
    info "시스템 패키지 설치 중... (portaudio, tkinter)"
    sudo apt-get install -y --no-install-recommends portaudio19-dev libsndfile1 python3-tk -q
    ok "시스템 패키지 설치됨"
  else
    warn "apt-get 없음 — portaudio19-dev, python3-tk 를 수동으로 설치하세요"
  fi
fi

# ── 가상환경 ──────────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  info "가상환경 생성 중..."
  $PYTHON -m venv .venv
fi
source .venv/bin/activate
ok "가상환경 활성화"

# ── Python 패키지 설치 ────────────────────────────────────────────────────────
info "공통 패키지 설치 중..."
pip install -q --upgrade pip
pip install -q sounddevice numpy webrtcvad-wheels argostranslate

if [ "$BACKEND" = "mlx" ]; then
  info "mlx-whisper 설치 중... (Apple Silicon)"
  pip install -q mlx-whisper
else
  info "faster-whisper 설치 중..."
  pip install -q faster-whisper
  # CUDA 버전이면 GPU 지원 torch 설치
  if [ "$HAS_CUDA" = true ]; then
    info "CUDA 지원 패키지 설치 중..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
  fi
fi
ok "패키지 설치 완료"

# ── Whisper 모델 다운로드 ─────────────────────────────────────────────────────
info "Whisper 모델 다운로드 중... (tiny, ~74MB)"
if [ "$BACKEND" = "mlx" ]; then
  python - <<EOF
import numpy as np, mlx_whisper, sys
try:
    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),
                           path_or_hf_repo="mlx-community/whisper-tiny-mlx", verbose=False)
    print("✅ Whisper 모델 준비 완료 (mlx)")
except Exception as e:
    print(f"❌ {e}"); sys.exit(1)
EOF
else
  python - <<EOF
from faster_whisper import WhisperModel
import sys
try:
    WhisperModel("tiny", device="${DEVICE_FLAG}", compute_type="${COMPUTE_TYPE}")
    print("✅ Whisper 모델 준비 완료 (faster-whisper, ${DEVICE_FLAG})")
except Exception as e:
    print(f"❌ {e}"); sys.exit(1)
EOF
fi

# ── argostranslate EN→KO 설치 ─────────────────────────────────────────────────
info "번역 패키지 다운로드 중... (EN→KO, ~100MB)"
python - <<'EOF'
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
    print("❌ EN→KO 패키지를 찾을 수 없습니다"); sys.exit(1)
package.install_from_path(pkg.download())
print("✅ 번역 패키지 설치 완료")
EOF

# ── 완료 ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}────────────────────────────────────────${RESET}"
echo -e "${GREEN}${BOLD}  ✅ 초기화 완료!${RESET}"
echo ""
echo -e "  실행 방법:"
echo -e "  ${CYAN}source .venv/bin/activate${RESET}"
echo -e "  ${CYAN}python earpiece.py --device blackhole${RESET}   # Mac"
echo -e "  ${CYAN}python earpiece.py --device 0${RESET}           # Linux"
echo -e "${BOLD}────────────────────────────────────────${RESET}"
