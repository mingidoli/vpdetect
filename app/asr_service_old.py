# app/asr_service.py

from pydub import AudioSegment
from pydub.utils import which
import os

from faster_whisper import WhisperModel
import os, tempfile
from pydub import AudioSegment

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffmpeg    = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"

_model = None

def _ensure_wav(input_path: str) -> str:
    try:
        if input_path.lower().endswith(".wav"):
            return input_path
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        import tempfile
        tmp_wav = tempfile.mktemp(suffix=".wav")
        audio.export(tmp_wav, format="wav")
        return tmp_wav
    except Exception as e:
        print("FFMPEG convert error:", e)
        raise

# app/asr_service.py (load_asr 교체)
def load_asr(model_size: str = "small"):
    global _model
    if _model is None:
        try:
            # 1순위: GPU
            _model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print("[ASR] GPU mode (CUDA, fp16) enabled.")
        except Exception as e:
            print("[ASR] GPU init failed -> CPU fallback:", e)
            _model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("[ASR] CPU mode (int8) enabled.")
    return True

def transcribe_file(input_path: str) -> str:
    assert _model is not None, "ASR model is not loaded"
    wav = _ensure_wav(input_path)
    segments, info = _model.transcribe(wav, language="ko")  # 한국어 고정
    texts = [seg.text.strip() for seg in segments]
    return " ".join(t for t in texts if t)

print(">>> Using ffmpeg at:", AudioSegment.converter)
print(">>> Using ffprobe at:", AudioSegment.ffprobe)
