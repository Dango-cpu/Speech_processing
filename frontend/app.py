from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.model_manager import ModelManager
from backend.models.model_1.metrics import (
    char_error_rate,
    real_time_factor,
    word_error_rate,
)
from backend.services.translation import LANGUAGES, TranslationService

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import numpy as np
    import soundfile as sf
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:
    np = None
    sf = None
    WebRtcMode = None
    webrtc_streamer = None

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def audio_frames_to_wav(frames):
    audio, sample_rate = audio_frames_to_audio(frames)
    if audio is None:
        return None

    return write_audio_to_wav(audio, sample_rate)


def audio_frames_to_audio(frames):
    arrays = []
    sample_rate = None

    for frame in frames:
        array = frame.to_ndarray()
        if array.ndim == 2:
            array = array.mean(axis=0)

        if np.issubdtype(array.dtype, np.integer):
            dtype_max = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
            array = array.astype("float32") / dtype_max
        else:
            array = array.astype("float32")

        arrays.append(array)
        sample_rate = frame.sample_rate

    if not arrays or sample_rate is None:
        return None, None

    audio = np.concatenate(arrays)
    max_value = np.max(np.abs(audio)) if audio.size else 0
    if max_value > 1:
        audio = audio / max_value

    return audio, sample_rate


def write_audio_to_wav(audio, sample_rate):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    sf.write(temp_path, audio, sample_rate)
    return temp_path


def audio_rms(audio):
    if audio is None or not audio.size:
        return 0
    return float(np.sqrt(np.mean(np.square(audio))))


def get_audio_duration_seconds(audio_path):
    try:
        if librosa is not None:
            return float(librosa.get_duration(path=str(audio_path)))

        if sf is not None:
            info = sf.info(str(audio_path))
            return float(info.frames / info.samplerate)
    except Exception:
        return None

    return None


def normalize_metric_text(text):
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = re.sub(r"[^\w\s\u00C0-\u1EF9]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def format_metric_percent(value):
    return f"{value * 100:.2f}%"


def audio_bytes_signature(source_name, audio_bytes):
    digest = hashlib.sha256(audio_bytes).hexdigest()
    return f"{source_name}:{len(audio_bytes)}:{digest}"


def append_with_overlap_cleanup(current_text, new_text, max_overlap_words=12):
    current_words = current_text.split()
    new_words = new_text.split()

    if not current_words:
        return new_text.strip()
    if not new_words:
        return current_text.strip()

    max_overlap = min(max_overlap_words, len(current_words), len(new_words))
    for overlap in range(max_overlap, 0, -1):
        current_tail = " ".join(current_words[-overlap:]).casefold()
        new_head = " ".join(new_words[:overlap]).casefold()
        if current_tail == new_head:
            merged_words = current_words + new_words[overlap:]
            return " ".join(merged_words).strip()

    return f"{current_text} {new_text}".strip()


def init_session_state(key, default_factory):
    if key not in st.session_state:
        st.session_state[key] = default_factory()

st.set_page_config(
    page_title="Conversation Transcription",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Conversation Transcription Demo")
st.caption("Giai đoạn 1: Thu âm hoặc upload audio, sau đó nhận dạng bằng Whisper")

if sys.version_info[:2] != (3, 12):
    st.warning(
        "This project is configured for Python 3.12. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Create a Python 3.12 virtual environment before loading speech models."
    )

with st.sidebar:
    st.subheader("Hugging Face")
    hf_token = st.text_input(
        "HF token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Used only for this running app session when downloading models.",
    )

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token.strip()
        st.success("HF token is set for this session.")
    else:
        os.environ.pop("HF_TOKEN", None)
        st.info("Paste a token if model downloads require authentication.")

init_session_state("model_manager", ModelManager)
init_session_state("translation_service", TranslationService)
init_session_state("loaded_model", lambda: None)
init_session_state("audio_path", lambda: None)
init_session_state("audio_signature", lambda: None)
init_session_state("transcript", lambda: None)
init_session_state("translation", lambda: None)
init_session_state("reference_transcript", str)
init_session_state("transcription_seconds", lambda: None)
init_session_state("audio_duration_seconds", lambda: None)
init_session_state("translation_target_language", lambda: None)
init_session_state("translation_model_loaded", lambda: False)
init_session_state("realtime_transcript", str)
init_session_state("realtime_translation", str)
init_session_state("realtime_running", lambda: False)
init_session_state("realtime_pending_translation", str)
init_session_state("realtime_last_translation_at", time.time)
init_session_state("realtime_audio_tail", lambda: None)
init_session_state("realtime_audio_tail_rate", lambda: None)

model_options = {
    "Model 3 - PhoWhisper Seq2Seq Encoder-Decoder": "model3_whisper",
    "Model 3 - Faster PhoWhisper / CTranslate2": "model3_faster_whisper",
    "Model 1 - CNN + BiLSTM + CTC": "model1_cnn_bilstm_ctc",
    "Model 2 - Wav2Vec2 Vietnamese": "model2_deepspeech",
}

selected_label = st.selectbox("Chọn mô hình ASR", list(model_options.keys()))
selected_model = model_options[selected_label]

asr_options = {}
if selected_model in {"model3_whisper", "model3_faster_whisper"}:
    with st.sidebar:
        st.subheader("ASR backend")
        configured_model_path = st.text_input(
            "Model path or id",
            value="",
            help=(
                "Leave empty for the default checkpoint. For faster-whisper, "
                "prefer a converted CTranslate2 model directory."
            ),
        )
        configured_device = st.selectbox("Device", ["auto", "cuda", "cpu"])
        configured_compute_type = st.selectbox(
            "Compute type",
            ["float16", "int8_float16", "int8", "float32"],
        )
        configured_language = st.text_input("Language", value="vi")

    if configured_model_path.strip():
        if selected_model == "model3_faster_whisper":
            asr_options["model_path"] = configured_model_path.strip()
        else:
            asr_options["model_name"] = configured_model_path.strip()
    if configured_device != "auto":
        asr_options["device"] = configured_device
    asr_options["language"] = configured_language.strip() or "vi"
    if selected_model == "model3_faster_whisper":
        asr_options["compute_type"] = configured_compute_type

translation_mode = st.selectbox(
    "Translation mode",
    ["Translate automatically", "Do not translate automatically"],
)
translation_enabled = translation_mode == "Translate automatically"

target_language_label = st.selectbox(
    "Target translation language",
    list(LANGUAGES.keys()),
)
target_language = LANGUAGES[target_language_label]

if st.session_state.translation_target_language != target_language:
    st.session_state.translation = None
    st.session_state.translation_target_language = target_language

translator_available_locally = (
    st.session_state.translation_service.local_model_exists()
)

if translation_enabled:
    if translator_available_locally:
        st.info("Translator checkpoint found locally.")
    else:
        st.warning(
            "Translator checkpoint was not found locally. It will be downloaded "
            "automatically when you click Load Selected Model."
        )

status_box = st.empty()

if st.button("Load Selected Model"):
    try:
        if translation_enabled:
            if translator_available_locally:
                load_message = "Loading speech-to-text and local translation models..."
            else:
                load_message = (
                    "Loading speech-to-text model and downloading translation model..."
                )

            status_box.info(load_message)
            with st.spinner(load_message):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    asr_future = executor.submit(
                        st.session_state.model_manager.load_model,
                        selected_model,
                        None,
                        **asr_options,
                    )
                    translation_future = executor.submit(
                        st.session_state.translation_service.load,
                    )

                    asr_future.result()
                    translation_future.result()

            st.session_state.translation_model_loaded = True
            status_box.success(
                f"{selected_label} and translator model are ready."
            )
        else:
            status_box.info("Loading speech-to-text model...")
            with st.spinner("Loading speech-to-text model..."):
                st.session_state.model_manager.load_model(
                    selected_model,
                    **asr_options,
                )

            status_box.success(f"{selected_label} loaded successfully.")

        st.session_state.loaded_model = selected_model

    except Exception as e:
        status_box.error(str(e))

if st.session_state.loaded_model is not None:
    st.success(f"Model đang được load: {st.session_state.loaded_model}")
else:
    st.info("Chưa có model nào được load. Hãy chọn model và bấm Load Selected Model.")

if st.session_state.translation_model_loaded:
    st.success("Translation model is loaded.")

st.divider()

st.subheader("Real-time microphone transcription")

realtime_mode = st.selectbox(
    "Realtime microphone mode",
    ["Off", "On"],
)

if realtime_mode == "Off":
    st.session_state.realtime_running = False
    st.info("Realtime microphone streaming is off.")
elif webrtc_streamer is None or WebRtcMode is None or np is None or sf is None:
    st.warning(
        "Real-time microphone mode requires streamlit-webrtc, numpy, and soundfile. "
        "Install the project dependencies, then restart Streamlit."
    )
else:
    realtime_chunk_seconds = int(
        st.selectbox(
            "Realtime chunk length",
            ["3", "5", "8", "10", "15"],
            index=1,
        )
    )
    realtime_context_seconds = int(
        st.selectbox(
            "Realtime context overlap",
            ["0", "1", "2", "3", "5"],
            index=2,
        )
    )
    realtime_silence_threshold = float(
        st.selectbox(
            "Realtime silence filter",
            ["0.003", "0.005", "0.010", "0.020"],
            index=1,
        )
    )
    realtime_translation_seconds = int(
        st.selectbox(
            "Realtime translation interval",
            ["3", "4", "5", "8", "10"],
            index=1,
        )
    )

    realtime_ctx = webrtc_streamer(
        key="realtime-transcription",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=4096,
        async_processing=True,
    )

    realtime_controls = st.columns(3)
    start_realtime = realtime_controls[0].button("Start live transcription")
    stop_realtime = realtime_controls[1].button("Stop live transcription")
    clear_realtime = realtime_controls[2].button("Clear live text")

    if start_realtime:
        st.session_state.realtime_running = True
        st.session_state.realtime_last_translation_at = time.time()

    if stop_realtime:
        st.session_state.realtime_running = False

    if clear_realtime:
        st.session_state.realtime_transcript = ""
        st.session_state.realtime_translation = ""
        st.session_state.realtime_pending_translation = ""
        st.session_state.realtime_audio_tail = None
        st.session_state.realtime_audio_tail_rate = None
        st.rerun()

    realtime_status = st.empty()
    realtime_transcript_column, realtime_translation_column = st.columns(2)
    realtime_transcript_column.markdown("**Live Vietnamese Transcript**")
    realtime_translation_column.markdown(
        f"**Live {target_language_label} Translation**"
    )
    realtime_transcript_placeholder = realtime_transcript_column.empty()
    realtime_translation_placeholder = realtime_translation_column.empty()

    def render_realtime_text():
        realtime_transcript_placeholder.code(
            st.session_state.realtime_transcript or " ",
            language=None,
        )
        realtime_translation_placeholder.code(
            st.session_state.realtime_translation or " ",
            language=None,
        )

    render_realtime_text()

    if st.session_state.realtime_running:
        if st.session_state.loaded_model is None:
            realtime_status.warning("Load the speech-to-text model first.")
        elif realtime_ctx.audio_receiver is None:
            realtime_status.warning(
                "Live transcription is armed. Start the microphone stream to begin."
            )
        elif not realtime_ctx.state.playing:
            realtime_status.warning(
                "Live transcription is armed. Waiting for the microphone stream."
            )
        else:
            realtime_status.info(
                "Listening continuously. Click Stop live transcription to end."
            )
            frames = []
            started_at = time.time()

            while time.time() - started_at < realtime_chunk_seconds:
                if not realtime_ctx.state.playing:
                    break
                try:
                    frames.extend(
                        realtime_ctx.audio_receiver.get_frames(timeout=1)
                    )
                except Exception:
                    pass

            if frames:
                chunk_audio, chunk_sample_rate = audio_frames_to_audio(frames)

                if chunk_audio is not None:
                    speech_seconds = len(chunk_audio) / chunk_sample_rate
                    speech_level = audio_rms(chunk_audio)
                    has_enough_speech = (
                        speech_seconds >= 0.75
                        and speech_level >= realtime_silence_threshold
                    )

                    if not has_enough_speech:
                        realtime_status.info(
                            "Listening. Skipped a quiet chunk "
                            f"(level {speech_level:.4f})."
                        )
                        st.rerun()

                    tail_audio = st.session_state.realtime_audio_tail
                    tail_rate = st.session_state.realtime_audio_tail_rate

                    if (
                        tail_audio is not None
                        and tail_rate == chunk_sample_rate
                        and realtime_context_seconds > 0
                    ):
                        model_audio = np.concatenate([tail_audio, chunk_audio])
                    else:
                        model_audio = chunk_audio

                    temp_audio_path = write_audio_to_wav(
                        model_audio,
                        chunk_sample_rate,
                    )

                    try:
                        chunk_transcript = st.session_state.model_manager.transcribe(
                            str(temp_audio_path)
                        )
                        chunk_translation = ""

                        if chunk_transcript:
                            st.session_state.realtime_transcript = (
                                append_with_overlap_cleanup(
                                    st.session_state.realtime_transcript,
                                    chunk_transcript,
                                )
                            )
                            st.session_state.realtime_pending_translation = (
                                append_with_overlap_cleanup(
                                    st.session_state.realtime_pending_translation,
                                    chunk_transcript,
                                )
                            )

                        should_translate = (
                            translation_enabled
                            and st.session_state.realtime_pending_translation
                            and time.time() - st.session_state.realtime_last_translation_at
                            >= realtime_translation_seconds
                        )

                        if should_translate:
                            st.session_state.translation_service.load()
                            chunk_translation = st.session_state.translation_service.translate(
                                st.session_state.realtime_pending_translation,
                                target_lang=target_language,
                            )
                            st.session_state.realtime_pending_translation = ""
                            st.session_state.realtime_last_translation_at = time.time()

                            if chunk_translation:
                                st.session_state.realtime_translation = (
                                    f"{st.session_state.realtime_translation} "
                                    f"{chunk_translation}"
                                ).strip()

                        render_realtime_text()
                        realtime_status.success(
                            "Updated at "
                            f"{time.strftime('%H:%M:%S')} "
                            f"(level {speech_level:.4f})"
                        )
                    except Exception as e:
                        realtime_status.error(str(e))
                    finally:
                        temp_audio_path.unlink(missing_ok=True)

                    if realtime_context_seconds > 0:
                        tail_samples = int(
                            chunk_sample_rate * realtime_context_seconds
                        )
                        st.session_state.realtime_audio_tail = chunk_audio[
                            -tail_samples:
                        ]
                        st.session_state.realtime_audio_tail_rate = chunk_sample_rate
                    else:
                        st.session_state.realtime_audio_tail = None
                        st.session_state.realtime_audio_tail_rate = None

            if st.session_state.realtime_running:
                st.rerun()
    else:
        realtime_status.info("Live transcription is stopped.")

st.divider()

input_mode = st.radio(
    "Chọn nguồn audio",
    ["Thu âm trực tiếp", "Upload file audio"],
    horizontal=True,
)

audio_path = None

if input_mode == "Thu âm trực tiếp":
    st.subheader("Thu âm từ microphone")

    if mic_recorder is None:
        st.warning(
            "Microphone recording is unavailable because streamlit-mic-recorder "
            "is not installed. Use audio upload or install the project dependencies."
        )
        audio = None
    else:
        audio = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=False,
            use_container_width=True,
            format="wav",
        )

    if audio is not None:
        audio_bytes = audio["bytes"]
        audio_path = UPLOAD_DIR / "recorded_audio.wav"
        audio_signature = audio_bytes_signature("microphone", audio_bytes)

        if st.session_state.audio_signature != audio_signature:
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            st.session_state.audio_path = str(audio_path)
            st.session_state.audio_signature = audio_signature
            st.session_state.transcript = None
            st.session_state.translation = None
            st.session_state.transcription_seconds = None
            st.session_state.audio_duration_seconds = None
        st.audio(audio_bytes, format="audio/wav")

elif input_mode == "Upload file audio":
    st.subheader("Upload file audio")

    uploaded_file = st.file_uploader(
        "Upload file audio tiếng Việt",
        type=["wav", "mp3", "m4a", "flac"],
    )

    if uploaded_file is not None:
        audio_path = UPLOAD_DIR / uploaded_file.name
        audio_bytes = uploaded_file.getvalue()
        audio_signature = audio_bytes_signature(uploaded_file.name, audio_bytes)

        if st.session_state.audio_signature != audio_signature:
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            st.session_state.audio_path = str(audio_path)
            st.session_state.audio_signature = audio_signature
            st.session_state.transcript = None
            st.session_state.translation = None
            st.session_state.transcription_seconds = None
            st.session_state.audio_duration_seconds = None
        st.audio(str(audio_path))

st.divider()

if st.session_state.audio_path is not None:
    st.write(f"Audio hiện tại: `{st.session_state.audio_path}`")

    if st.button("Transcribe"):

        if st.session_state.loaded_model is None:
            st.warning("Hãy load model trước.")
        else:
            try:
                with st.spinner("Đang nhận dạng..."):
                    transcription_started_at = time.perf_counter()
                    transcript = st.session_state.model_manager.transcribe(
                        st.session_state.audio_path
                    )
                    transcription_seconds = (
                        time.perf_counter() - transcription_started_at
                    )

                st.session_state.transcript = transcript
                st.session_state.translation = None
                st.session_state.transcription_seconds = transcription_seconds
                st.session_state.audio_duration_seconds = (
                    get_audio_duration_seconds(st.session_state.audio_path)
                )

                if translation_enabled and transcript.strip():
                    translation_status = st.empty()

                    def update_translation_status(message):
                        translation_status.info(message)

                    with st.spinner("Translating transcript..."):
                        st.session_state.translation_service.load(
                            progress_callback=update_translation_status,
                        )
                        st.session_state.translation = (
                            st.session_state.translation_service.translate(
                                transcript,
                                target_lang=target_language,
                            )
                        )

                    translation_status.success("Translation ready.")

            except Exception as e:
                st.error(str(e))

    if st.session_state.transcript:
        transcript_column, translation_column = st.columns(2)

        with transcript_column:
            st.subheader("Vietnamese Transcript")
            st.text_area(
                "Transcript",
                st.session_state.transcript,
                height=220,
                label_visibility="collapsed",
            )

        with translation_column:
            st.subheader(f"{target_language_label} Translation")

            if st.session_state.translation:
                st.text_area(
                    "Translation",
                    st.session_state.translation,
                    height=220,
                    label_visibility="collapsed",
                )
            else:
                st.info("No translation yet.")

            if st.button("Translate latest transcript"):
                try:
                    translation_status = st.empty()

                    def update_translation_status(message):
                        translation_status.info(message)

                    with st.spinner("Translating transcript..."):
                        st.session_state.translation_service.load(
                            progress_callback=update_translation_status,
                        )
                        st.session_state.translation = (
                            st.session_state.translation_service.translate(
                                st.session_state.transcript,
                                target_lang=target_language,
                            )
                        )
                    translation_status.success("Translation ready.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.subheader("Model evaluation")
        reference_transcript = st.text_area(
            "Reference transcript for WER",
            key="reference_transcript",
            height=120,
            help="Paste the correct Vietnamese transcript here to calculate WER.",
        )

        normalized_reference = normalize_metric_text(reference_transcript)
        normalized_hypothesis = normalize_metric_text(st.session_state.transcript)
        rtf = real_time_factor(
            st.session_state.transcription_seconds,
            st.session_state.audio_duration_seconds,
        )

        metric_columns = st.columns(5)
        if normalized_reference:
            wer = word_error_rate(normalized_reference, normalized_hypothesis)
            cer = char_error_rate(normalized_reference, normalized_hypothesis)
            metric_columns[0].metric("WER", format_metric_percent(wer))
            metric_columns[1].metric("CER", format_metric_percent(cer))
        else:
            metric_columns[0].metric("WER", "Add reference")
            metric_columns[1].metric("CER", "Add reference")

        metric_columns[2].metric(
            "RTF",
            f"{rtf:.3f}" if rtf is not None else "N/A",
        )
        metric_columns[3].metric(
            "Audio duration",
            (
                f"{st.session_state.audio_duration_seconds:.2f}s"
                if st.session_state.audio_duration_seconds is not None
                else "N/A"
            ),
        )
        metric_columns[4].metric(
            "Processing time",
            (
                f"{st.session_state.transcription_seconds:.2f}s"
                if st.session_state.transcription_seconds is not None
                else "N/A"
            ),
        )
else:
    st.info("Hãy thu âm hoặc upload file audio trước.")
