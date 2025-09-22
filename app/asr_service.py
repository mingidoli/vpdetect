# app/asr_service.py
# 웹에서 import해서 쓰는 STT 모듈
# - FFmpeg/FFprobe 경로 고정(환경변수/후보 경로/자동탐지)
# - pydub 변환 실패 시 ffmpeg 서브프로세스 폴백
# - GPU(fp16) -> GPU(int8_float16) -> CPU(int8) 단계 폴백
# - 비-wav 입력(m4a 등) 자동 변환
# 공개 API: load_asr(model_size="small"), transcribe_file(input_path)

import os
import tempfile
import subprocess
from typing import Optional

from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.utils import which

# 전역 모델 핸들
_model: Optional[WhisperModel] = None

# 디코딩 기본 옵션(안정성 위주)
TRANSCRIBE_KW = dict(
    language="ko",                         # 한국어 고정
    beam_size=5,                           # 빔서치(환각/삑사리 감소)
    vad_filter=True,                       # 무음 구간 정리
    vad_parameters={"min_silence_duration_ms": 500},
    temperature=0.0,                       # 일관성↑
)

# ---- FFmpeg/FFprobe 경로 관련 -------------------------------------------------

# 1) 환경변수로 직접 주입할 수 있음 (없으면 자동탐지/후보 경로 사용)
FFMPEG_ENV = os.getenv("FFMPEG_BIN")
FFPROBE_ENV = os.getenv("FFPROBE_BIN")

FFMPEG_CANDIDATES = [
    FFMPEG_ENV,
    which("ffmpeg"),
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
]
FFPROBE_CANDIDATES = [
    FFPROBE_ENV,
    which("ffprobe"),
    r"C:\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
    r"C:\ProgramData\chocolatey\bin\ffprobe.exe",
]


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def _setup_ffmpeg_paths() -> None:
    """
    pydub 및 하위 호출이 사용할 ffmpeg/ffprobe 경로를 확정한다.
    - 환경변수/자동탐지/후보 경로 순으로 검색
    - pydub 속성 및 환경변수에 모두 반영
    """
    ffmpeg_path = _first_existing(FFMPEG_CANDIDATES)
    ffprobe_path = _first_existing(FFPROBE_CANDIDATES)

    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError(
            "FFmpeg/FFprobe를 찾을 수 없습니다. ffmpeg 설치 후 환경변수 FFMPEG_BIN/FFPROBE_BIN "
            "또는 PATH를 설정하세요. (예: C:\\ffmpeg\\bin\\ffmpeg.exe)"
        )

    # pydub에 직접 경로 주입
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    # 환경변수에도 반영(서브프로세스가 참고)
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    os.environ["FFPROBE_BINARY"] = ffprobe_path
    ffdir = os.path.dirname(ffmpeg_path)
    if ffdir and ffdir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffdir + os.pathsep + os.environ.get("PATH", "")

    print(f"[FFMPEG] ffmpeg:  {ffmpeg_path}")
    print(f"[FFMPEG] ffprobe: {ffprobe_path}")


def _ensure_wav(input_path: str) -> str:
    """
    입력 파일이 wav가 아니면 임시 wav로 변환하여 경로 반환.
    1) pydub으로 시도
    2) 실패 시 ffmpeg 서브프로세스로 폴백
    """
    if input_path.lower().endswith(".wav"):
        return input_path

    tmp_wav = tempfile.mktemp(suffix=".wav")

    # 1) pydub 시도
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(tmp_wav, format="wav")
        print(f"[AUDIO] Converted to wav via pydub: {tmp_wav}")
        return tmp_wav
    except Exception as e:
        print("[AUDIO] pydub convert failed, fallback to ffmpeg:", repr(e))

    # 2) ffmpeg 직접 호출
    ffmpeg_path = getattr(AudioSegment, "ffmpeg", None) or os.environ.get("FFMPEG_BINARY") or "ffmpeg"
    cmd = [
        ffmpeg_path,
        "-y",                 # overwrite
        "-i", input_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz
        tmp_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[AUDIO] Converted to wav via ffmpeg: {tmp_wav}")
        return tmp_wav
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 변환 실패: {e.stderr.decode(errors='ignore')[:500]}") from e


# ---- ASR 로딩/전사 ------------------------------------------------------------

def load_asr(model_size: str = "small") -> bool:
    """
    Whisper 모델을 1회 로드.
    - CUDA fp16 → CUDA int8_float16 → CPU int8 순으로 폴백
    """
    global _model
    if _model is not None:
        return True

    _setup_ffmpeg_paths()

    # 1) CUDA fp16
    try:
        _model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print(f"[ASR] Loaded '{model_size}' on CUDA fp16")
        return True
    except Exception as e:
        print("[ASR] CUDA fp16 init failed:", repr(e))

    # 2) CUDA int8_float16 (VRAM 여유 적을 때 유리)
    try:
        _model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        print(f"[ASR] Loaded '{model_size}' on CUDA int8_float16")
        return True
    except Exception as e:
        print("[ASR] CUDA int8_float16 init failed:", repr(e))

    # 3) CPU int8
    try:
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[ASR] Loaded '{model_size}' on CPU int8")
        return True
    except Exception as e:
        print("[ASR] CPU int8 init failed:", repr(e))
        _model = None
        return False


def transcribe_file(input_path: str) -> str:
    """
    파일 경로를 받아 전체 텍스트(문장 연결)를 반환.
    (웹 서비스에서는 업로드 받은 임시경로를 그대로 넘겨주면 됨)
    """
    assert _model is not None, "ASR model is not loaded. 먼저 load_asr()를 호출하세요."
    wav = _ensure_wav(input_path)
    segments, info = _model.transcribe(wav, **TRANSCRIBE_KW)
    try:
        print(f"[ASR] Detected language: {info.language} (prob={info.language_probability:.2f})")
    except Exception:
        # 일부 버전에서 info.language_probability가 없을 수 있음
        print(f"[ASR] Detected language: {getattr(info, 'language', 'unknown')}")

    texts = [seg.text.strip() for seg in segments]
    return " ".join(t for t in texts if t)
