FROM python:3.11-slim

# ── 시스템 의존성 ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python 패키지 ──────────────────────────────────────────────────────────────
COPY requirements.txt requirements_linux.txt ./
RUN pip install --no-cache-dir -r requirements_linux.txt

# ── Whisper 모델 사전 다운로드 (tiny) ─────────────────────────────────────────
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('tiny', device='cpu', compute_type='int8'); \
print('✅ Whisper 모델 준비 완료')"

# ── argostranslate EN→KO 사전 설치 ────────────────────────────────────────────
RUN python -c "\
from argostranslate import package, translate; \
package.update_package_index(); \
available = package.get_available_packages(); \
pkg = next((p for p in available if p.from_code == 'en' and p.to_code == 'ko'), None); \
package.install_from_path(pkg.download()) if pkg else None; \
print('✅ 번역 패키지 준비 완료')"

# ── 소스 복사 ─────────────────────────────────────────────────────────────────
COPY transcribe_local.py debug_audio.py template_meeting.md ./

CMD ["python", "transcribe_local.py"]
