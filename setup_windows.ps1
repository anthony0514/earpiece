# Neural D Meeting Transcriber — Windows Setup
# PowerShell 5.1+  /  Windows 10/11
# 실행 정책 오류 시: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

Write-Host "----------------------------------------"
Write-Host "  Neural D Meeting Transcriber — Windows Setup"
Write-Host "----------------------------------------"

# Python 확인
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python이 없습니다. https://python.org 에서 설치하세요."
    exit 1
}
Write-Host "✅ Python: $(python --version)"

# 가상환경
if (-not (Test-Path ".venv")) {
    Write-Host "→ 가상환경 생성 중..."
    python -m venv .venv
}
& ".\.venv\Scripts\Activate.ps1"
Write-Host "✅ 가상환경 활성화"

# 패키지 설치
Write-Host "→ 패키지 설치 중..."
pip install -q --upgrade pip
pip install -q -r requirements_windows.txt
Write-Host "✅ 패키지 설치 완료"

# faster-whisper 모델 다운로드 (tiny)
Write-Host "→ Whisper 모델 다운로드 중... (faster-whisper tiny, ~74MB)"
python - @"
from faster_whisper import WhisperModel
import sys
try:
    WhisperModel("tiny", device="cpu", compute_type="int8")
    print("✅ Whisper 모델 준비 완료")
except Exception as e:
    print(f"❌ Whisper 모델 다운로드 실패: {e}")
    sys.exit(1)
"@

# argostranslate EN→KO 패키지 다운로드
Write-Host "→ 번역 패키지 다운로드 중... (EN→KO, ~100MB)"
python - @"
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
"@

Write-Host ""
Write-Host "----------------------------------------"
Write-Host "  ✅ 설치 완료!"
Write-Host ""
Write-Host "  실행 방법:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python transcribe_local.py --device 0"
Write-Host ""
Write-Host "  ※ Windows 오디오 캡처: VB-Audio Virtual Cable 사용"
Write-Host "     https://vb-audio.com/Cable/"
Write-Host "----------------------------------------"
