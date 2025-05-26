# CosyVoice2 예제 분석 (GestureLSM2/examples/libritts/cosyvoice2/)

## 1. 서론

`GestureLSM2/examples/libritts/cosyvoice2/` 예제는 LibriTTS 데이터셋을 사용하여 고품질 Text-to-Speech (TTS) 모델을 학습하고 추론하는 전체 파이프라인을 제공합니다. 이 시스템은 대규모 언어 모델(LLM), Flow 기반 음향 모델, 그리고 HiFiGAN 보코더를 결합한 현대적인 TTS 아키텍처를 특징으로 합니다. Supervised Fine-Tuning (SFT) 및 Zero-shot 추론 모드를 지원하여 다양한 TTS 시나리오에 활용될 수 있습니다.

## 2. 디렉토리 구조

주요 파일 및 디렉토리 구성은 다음과 같습니다:

*   `run.sh`: 전체 데이터 준비, 학습, 추론 파이프라인을 실행하는 메인 셸 스크립트.
*   `path.sh`: Python 환경 변수(`PYTHONPATH`) 및 인코딩 설정을 수행하는 스크립트.
*   `conf/`:
    *   `cosyvoice2.yaml`: 모델 아키텍처, 학습 하이퍼파라미터, 데이터 처리 파이프라인 등을 정의하는 핵심 설정 파일.
    *   `ds_stage2.json`: (존재한다면) Deepspeed 학습을 위한 설정 파일.
*   `local/`: LibriTTS 데이터셋에 특화된 초기 데이터 처리 스크립트 위치.
    *   `download_and_untar.sh`: 데이터셋 다운로드 및 압축 해제.
    *   `prepare_data.py`: Kaldi 스타일의 기본 데이터 파일(`wav.scp`, `text` 등) 생성.
*   `tools/`: 데이터 전처리 과정에서 사용되는 보조 스크립트 위치.
    *   `extract_embedding.py`: 화자 및 발화 임베딩 추출 스크립트.
    *   `extract_speech_token.py`: 이산 음성 토큰 추출 스크립트.
    *   `make_parquet_list.py`: 학습용 Parquet 데이터 파일 및 관련 리스트 생성 스크립트.
*   `cosyvoice/bin/`: 핵심 모델 학습, 추론 및 유틸리티 스크립트 위치.
    *   `train.py`: 모델 학습 실행.
    *   `inference.py`: 모델 추론 실행.
    *   `average_model.py`: 학습된 모델 체크포인트 평균화.
    *   `export_jit.py`: 모델을 TorchScript(JIT) 형식으로 내보내기.
    *   `export_onnx.py`: 모델을 ONNX 형식으로 내보내기.
*   `tts_text.json` (예상): 추론 시 입력 텍스트를 담는 파일.
*   `exp/`: 학습 및 추론 결과가 저장되는 디렉토리 (생성됨).
*   `tensorboard/`: TensorBoard 로그가 저장되는 디렉토리 (생성됨).
*   `data/`: 전처리된 데이터가 저장되는 디렉토리 (생성됨).

## 3. 실행 파이프라인 (`run.sh`)

`run.sh` 스크립트는 `stage` 변수를 통해 단계별 실행을 제어합니다.

*   **Stage -1: 데이터 다운로드**
    *   `local/download_and_untar.sh`를 사용하여 LibriTTS 데이터셋의 각 파티션 (dev-clean, test-clean, train-clean-100 등)을 다운로드하고 압축 해제합니다.
*   **Stage 0: 기본 데이터 준비**
    *   `local/prepare_data.py`를 사용하여 각 데이터 파티션에 대해 `wav.scp`, `text`, `utt2spk`, `spk2utt` 파일을 생성합니다.
*   **Stage 1: 화자 임베딩 추출**
    *   `tools/extract_embedding.py`와 사전 학습된 Campplus ONNX 모델 (`pretrained_models/CosyVoice2-0.5B/campplus.onnx`)을 사용하여 Fbank 특징으로부터 발화별 및 화자별 임베딩 (`utt2embedding.pt`, `spk2embedding.pt`)을 생성합니다.
*   **Stage 2: 이산 음성 토큰 추출**
    *   `tools/extract_speech_token.py`와 사전 학습된 Speech Tokenizer ONNX 모델 (`pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx`)을 사용하여 Whisper 스타일의 Log-Mel Spectrogram으로부터 발화별 이산 음성 토큰 (`utt2speech_token.pt`)을 생성합니다.
*   **Stage 3: Parquet 형식 데이터 준비**
    *   `tools/make_parquet_list.py`를 사용하여 이전 단계에서 생성된 모든 정보(오디오 원본 바이트 데이터 포함)를 취합하여 학습에 사용될 다수의 Parquet 파일 및 관련 인덱스 파일 (`data.list`, `utt2data.list`, `spk2data.list`)을 생성합니다.
*   **Stage 4: 추론 (Inference)**
    *   `cosyvoice/bin/inference.py`를 실행하여 TTS 추론을 수행합니다.
    *   `sft` 및 `zero_shot` 두 가지 모드를 지원합니다.
    *   `conf/cosyvoice2.yaml` 설정 파일을 사용하며, 사전 학습된 Qwen 모델 및 학습된 LLM, Flow, HiFiGAN 모델 경로를 인자로 받습니다.
*   **Stage 5: 모델 학습 (LLM, Flow)**
    *   `cosyvoice/bin/train.py`를 실행하여 모델을 학습합니다. (주로 LLM, Flow 모델 대상)
    *   `torchrun`을 이용한 분산 학습(DDP)을 지원하며, Deepspeed 사용도 가능합니다.
    *   `conf/cosyvoice2.yaml` 설정 및 Parquet 데이터 리스트를 사용합니다.
*   **Stage 6: 모델 평균화**
    *   `cosyvoice/bin/average_model.py`를 사용하여 학습된 여러 모델 체크포인트의 가중치를 평균화하여 성능을 개선합니다.
*   **Stage 7: 모델 내보내기**
    *   `cosyvoice/bin/export_jit.py` 및 `cosyvoice/bin/export_onnx.py`를 사용하여 추론 속도 향상을 위해 모델을 JIT 또는 ONNX 형식으로 변환합니다.

## 4. 핵심 설정 (`conf/cosyvoice2.yaml`)

YAML 파일을 사용하여 모델 아키텍처, 학습 파라미터, 데이터 처리 파이프라인 등을 상세하게 정의합니다 (`hyperpyyaml` 라이브러리 사용).

*   **전역 설정**: 샘플링 레이트, LLM 입출력 크기, 화자 임베딩 차원, 토큰 프레임 레이트, Qwen 사전 학습 모델 경로 등.
*   **모델 컴포넌트 정의**:
    *   **`llm`**: `cosyvoice.llm.llm.Qwen2LM` (내부적으로 `Qwen2Encoder` 사용). 텍스트 인코딩 및 의미 정보 추출.
    *   **`flow`**: `cosyvoice.flow.flow.CausalMaskedDiffWithXvec`. 음성 토큰으로부터 음향 특징(Mel-spectrogram)을 생성하는 Flow-matching 기반 모델.
        *   **Encoder**: `cosyvoice.transformer.upsample_encoder.UpsampleConformerEncoder`.
        *   **Decoder (CFM)**: `cosyvoice.flow.flow_matching.CausalConditionalCFM` (내부적으로 `CausalConditionalDecoder` 사용).
    *   **`hift` (Vocoder)**: `cosyvoice.hifigan.generator.HiFTGenerator`. HiFiGAN 기반 보코더로, 음향 특징으로부터 실제 음성 파형 생성. F0 예측기 포함.
*   **GAN 관련 모듈**: `hift` (Generator) 학습을 위한 판별자(`cosyvoice.hifigan.discriminator.MultipleDiscriminator` - MPD, MRD 구성) 및 Mel-spectrogram 변환 설정.
*   **데이터 처리 파이프라인 (`data_pipeline`, `data_pipeline_gan`)**:
    *   Parquet 파일 로딩, 토큰화, 필터링, 리샘플링, Fbank/F0 특징 추출, 임베딩 파싱, 셔플링, 정렬, 배치 구성, 패딩 등 일련의 데이터 전처리 단계를 정의합니다.
    *   각 단계는 `cosyvoice.dataset.processor` 또는 `matcha.utils.audio` 내의 함수/클래스로 정의됩니다.

## 5. 주요 스크립트 상세

*   **`path.sh`**:
    *   Python의 기본 인코딩을 `UTF-8`로 설정합니다.
    *   `PYTHONPATH`에 프로젝트 루트 디렉토리 (`../../../`)와 `../../../third_party/Matcha-TTS`를 추가하여, `cosyvoice` 및 `matcha` 관련 모듈을 임포트할 수 있도록 합니다.
*   **`local/download_and_untar.sh`**:
    *   `wget`을 사용하여 LibriTTS 데이터셋을 다운로드하고 `tar`로 압축을 해제합니다.
    *   파일 크기 검사, 완료 플래그 파일 생성 등의 기능을 포함합니다.
*   **`local/prepare_data.py`**:
    *   다운로드된 LibriTTS 데이터를 스캔하여 `wav.scp`, `text`, `utt2spk`, `spk2utt` 파일을 생성합니다.
*   **`tools/extract_embedding.py`**:
    *   ONNX 런타임과 사전 학습된 Campplus 모델을 사용하여 Fbank 특징으로부터 발화별 및 화자별 임베딩을 추출합니다.
    *   멀티스레딩을 지원하며, 결과를 `.pt` 파일로 저장합니다.
*   **`tools/extract_speech_token.py`**:
    *   ONNX 런타임과 사전 학습된 Speech Tokenizer 모델을 사용하여 Whisper 스타일의 Log-Mel Spectrogram으로부터 이산 음성 토큰을 추출합니다.
    *   GPU 사용을 시도하며, 30초 이상 오디오 처리는 제한합니다. 결과를 `.pt` 파일로 저장합니다.
*   **`tools/make_parquet_list.py`**:
    *   이전 단계들의 모든 출력(오디오 원본 바이트 포함)을 취합하여 Pandas DataFrame으로 구성 후, 다수의 Parquet 파일로 저장합니다.
    *   각 Parquet 파일에 대한 발화 및 화자 매핑 정보(JSON)와 전체 Parquet 파일 목록(`data.list` 등)을 생성합니다. 멀티프로세싱을 지원합니다.
*   **`cosyvoice/bin/train.py`**:
    *   분산 학습(DDP, Deepspeed) 및 AMP를 지원하는 학습 스크립트.
    *   `hyperpyyaml`로 YAML 설정을 로드하고, 커맨드라인 인자로 일부 설정을 오버라이드합니다.
    *   학습 대상 모델(LLM, Flow, HiFiGAN 등)을 선택적으로 로드하고 학습합니다.
    *   `cosyvoice.utils.executor.Executor`를 통해 학습 루프를 관리합니다.
    *   체크포인트 저장 및 로드, TensorBoard 로깅 기능을 포함합니다.
*   **`cosyvoice/bin/inference.py`**:
    *   학습된 모델을 로드하여 TTS 추론을 수행합니다.
    *   `CosyVoiceModel` 또는 `CosyVoice2Model` 래퍼 클래스를 사용하여 LLM, Flow, Vocoder를 통합적으로 관리합니다.
    *   `sft` 및 `zero_shot` 추론 모드를 지원하며, 각 모드에 따라 다른 입력 구성을 사용합니다.
    *   생성된 음성 파일을 `.wav` 형식으로 저장하고 `wav.scp` 파일을 생성합니다.

## 6. 주요 기술 및 특징

*   **모듈화된 TTS 파이프라인**: LLM (텍스트 이해) -> Flow 모델 (음향 특징 생성) -> HiFiGAN (음성 파형 생성)의 단계별 구성.
*   **YAML 기반 설정**: `hyperpyyaml`을 사용하여 모델, 학습 파라미터, 데이터 파이프라인 등을 유연하게 관리.
*   **Flow Matching**: Flow 모델 학습에 사용되어 고품질의 음향 특징 생성을 목표.
*   **다양한 추론 모드**: SFT 및 Zero-shot TTS 지원.
*   **분산 학습 및 효율성**: DDP, Deepspeed, AMP (Automatic Mixed Precision) 지원.
*   **사전 학습 모델 활용**: Qwen (LLM), Campplus (화자 임베딩), Speech Tokenizer (ONNX 모델) 등 사전 학습된 모델을 적극 활용.
*   **데이터 처리**: Parquet 컬럼 기반 저장 포맷을 사용하여 효율적인 데이터 관리 및 로딩. Whisper 라이브러리 활용 (Log-Mel Spectrogram).
*   **ONNX 런타임 활용**: 일부 전처리 단계(임베딩, 음성 토큰 추출)에서 ONNX 모델을 사용하여 이식성 및 실행 효율성 확보.

## 7. 분석 완료

이 문서 작성 시점에서 `GestureLSM2/examples/libritts/cosyvoice2/` 예제의 주요 구성 요소 및 실행 흐름에 대한 분석이 완료되었습니다. 