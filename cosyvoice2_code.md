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

## 8. 아키텍처 및 작동 방식 상세

CosyVoice2는 고품질 Text-to-Speech (TTS)를 위해 여러 신경망 모듈을 계층적으로 결합한 현대적인 아키텍처를 사용합니다. 핵심 구성 요소는 LLM(Large Language Model), Flow-matching 기반 음향 모델, 그리고 HiFiGAN 보코더입니다. 이들의 상호작용은 `conf/cosyvoice2.yaml` 파일에 상세히 정의되어 있습니다.

### 8.1. 입력 처리

*   **텍스트 입력**: 사용자가 음성으로 변환하고자 하는 주 텍스트입니다.
*   **선택적 입력 (Zero-shot/SFT 모드에 따라 다름)**:
    *   **화자 ID (`spk_id`)**: SFT(Supervised Fine-Tuning) 모드에서 특정 학습된 화자를 지정할 때 사용됩니다.
    *   **텍스트 프롬프트 (`text_prompt`)**: Zero-shot 모드에서 원하는 음성 스타일이나 운율을 텍스트로 설명하여 전달할 때 사용될 수 있습니다.
    *   **오디오 프롬프트 (`speech_prompt`)**: Zero-shot 모드에서 원하는 음성 스타일(음색, 리듬 등)을 가진 짧은 참조 오디오를 제공합니다. 이 오디오로부터 화자 임베딩(`xvector`) 또는 다른 스타일 특징이 추출되어 모델에 조건으로 주입됩니다. (`tools/extract_embedding.py`가 이러한 임베딩 추출에 사용될 수 있습니다.)

### 8.2. LLM (Large Language Model - `cosyvoice.llm.llm.Qwen2LM`)

*   **역할**: 입력된 텍스트 정보로부터 의미론적 특징과 운율적 특징을 추출하여, 후속 음향 모델이 사용할 중간 표현(Intermediate Representation)을 생성합니다.
*   **입력**: 주 텍스트, (Zero-shot/스타일 제어 시) 텍스트 프롬프트 또는 오디오 프롬프트에서 파생된 정보.
*   **처리 과정**:
    1.  **텍스트 인코딩**: 사전 학습된 `Qwen2Encoder` (Qwen 언어 모델 기반)를 사용하여 입력 텍스트를 인코딩합니다. 이를 통해 텍스트의 의미를 담고 있는 고수준의 벡터 시퀀스(텍스트 임베딩)를 생성합니다.
    2.  **음성 토큰 예측**: 인코딩된 텍스트 임베딩과 (필요시) 프롬프트 정보를 바탕으로, **음성 토큰(speech token) 시퀀스**를 예측/생성합니다. 이 음성 토큰은 이산적인 단위로, 각 토큰은 특정 음향적/운율적 특성(예: 음소, 리듬, 피치 변화 패턴, 스타일 등)을 나타낼 수 있습니다. `conf/cosyvoice2.yaml`의 `llm_model.output_names: ["xs", "speech_tokens"]` 설정은 `xs` (인코딩된 텍스트 특징)와 `speech_tokens` (예측된 음성 토큰)를 출력함을 명시합니다.
    3.  **Zero-shot/스타일 제어**: Zero-shot 모드에서는 오디오 프롬프트에서 추출된 화자 임베딩이나 텍스트 프롬프트의 스타일 정보가 음성 토큰 예측 과정에 영향을 미쳐, 생성될 음성의 스타일을 제어할 수 있습니다.
*   **출력**:
    *   `xs`: 인코딩된 텍스트 특징 벡터 시퀀스 (의미 정보).
    *   `speech_tokens`: 예측된 음성 토큰 시퀀스 (운율 및 스타일 정보).

### 8.3. Flow-matching 음향 모델 (`cosyvoice.flow.flow.CausalMaskedDiffWithXvec`)

*   **역할**: LLM으로부터 받은 텍스트 특징 및 음성 토큰 시퀀스, 그리고 (필요시) 화자 정보를 조건으로 받아 자연스러운 음향 특징인 Mel-spectrogram을 생성합니다.
*   **입력**:
    *   LLM이 생성한 `xs` (텍스트 특징) 및 `speech_tokens`.
    *   화자 임베딩 (`xvector`): SFT 모드에서는 지정된 화자 ID에 해당하는 임베딩을, Zero-shot 모드에서는 오디오 프롬프트에서 추출된 임베딩을 사용합니다.
*   **처리 과정**:
    1.  **인코더 (`flow_model.encoder: cosyvoice.transformer.upsample_encoder.UpsampleConformerEncoder`)**:
        *   입력된 `speech_tokens`를 업샘플링하여 시간적 해상도를 Mel-spectrogram과 유사하게 맞춥니다. 그 후, 텍스트 특징(`xs`)과 결합하여 Conformer 네트워크를 통해 더욱 정제되고 풍부한 조건부 특징 벡터 시퀀스를 생성합니다. 이 특징은 후속 디코더가 Mel-spectrogram을 생성하는 데 필요한 상세한 가이드 역할을 합니다.
    2.  **디코더 (`flow_model.decoder: cosyvoice.flow.flow_matching.CausalConditionalCFM`)**:
        *   인코더에서 생성된 조건부 특징 벡터 시퀀스와 화자 임베딩(`xvector`)을 조건으로 사용합니다.
        *   랜덤 노이즈로부터 시작하여, Conditional Flow Matching (CFM) 또는 이와 유사한 Score-based/Diffusion 기반의 점진적 생성 과정을 통해 목표 Mel-spectrogram을 반복적으로 정제해 나갑니다. "Causal" 특성은 각 시간 스텝의 Mel-frame이 과거의 정보와 현재 조건에만 의존하여 인과적으로 생성됨을 의미하며, 실시간 스트리밍 합성에 유리할 수 있습니다.
*   **출력**: Mel-spectrogram 시퀀스 (`mel`).

### 8.4. HiFiGAN Vocoder (`cosyvoice.hifigan.generator.HiFTGenerator`)

*   **역할**: 음향 모델이 생성한 Mel-spectrogram을 실제 사람이 들을 수 있는 고품질의 음성 파형(waveform)으로 변환합니다.
*   **입력**: Mel-spectrogram (`mel`), (선택적으로) F0(기본 주파수) 컨투어.
*   **처리 과정**:
    1.  입력된 Mel-spectrogram을 일련의 업샘플링 레이어와 잔차 블록으로 구성된 HiFiGAN의 Generator 구조에 통과시킵니다. 이 과정에서 주파수 및 시간 해상도가 점차 높아지며, 파형의 미세한 세부 특성들이 복원됩니다.
    2.  설정에서 `use_f0: true`이고 F0 정보가 제공되면 (외부에서 주어지거나 보코더 내부의 F0 예측기를 통해 추정), 이를 활용하여 더욱 자연스럽고 표현력 있는 억양과 운율을 가진 음성을 생성합니다.
*   **출력**: 최종 음성 파형 (raw audio waveform).

### 8.5. 요약된 데이터 흐름도

```
+-------------------------+     +-----------------------+
|      텍스트 입력        | --> | (선택적) 오디오/텍스트  |
| (및 화자 ID for SFT)    |     |       프롬프트        |
+-------------------------+     +----------+------------+
             |                              |
             |       +----------------------+----------------------+
             |       | 화자 임베딩 (xvector) 추출 (Zero-shot 시) |
             |       +--------------------------------------+
             v
+-------------------------------------------------------------+
|                    LLM (Qwen2LM)                            |
|  - 입력 텍스트 인코딩 (Qwen2Encoder) -> 텍스트 특징 (xs)      |
|  - (프롬프트 기반) 음성 토큰 (speech_tokens) 예측/생성        |
+-------------------------+-------------------------------------+
                          | (xs, speech_tokens)
                          v
+-------------------------------------------------------------+
|      Flow-matching 음향 모델 (CausalMaskedDiffWithXvec)     |     +-----------------+
|  - 인코더 (UpsampleConformerEncoder):                       |<----| 화자 임베딩     |
|    - speech_tokens 업샘플링                                 |     | (xvector)       |
|    - xs와 결합하여 조건부 특징 생성                         |     +-----------------+
|  - 디코더 (CausalConditionalCFM):                           |
|    - 조건부 특징 & xvector 기반으로 Mel-spectrogram 생성    |
+-------------------------+-------------------------------------+
                          | (mel: Mel-spectrogram)
                          v
+-------------------------------------------------------------+     +-----------------+
|                  HiFiGAN Vocoder (HiFTGenerator)            |<----| (선택적) F0     |
|  - Mel-spectrogram을 음성 파형으로 변환                       |     | 컨투어          |
+-------------------------+-------------------------------------+
                          |
                          v
+-------------------------+
|       음성 파형 출력    |
+-------------------------+
```

이러한 모듈화된 접근 방식은 각 구성 요소가 특정 작업에 최적화되도록 하여, 전체적으로 높은 수준의 음성 품질, 제어 가능성, 그리고 다양한 화자 및 스타일에 대한 적응력을 달성할 수 있게 합니다. LLM은 콘텐츠의 이해와 고수준의 스타일 및 운율 제어를 담당하고, Flow 모델은 섬세한 음향 특징을 생성하며, HiFiGAN 보코더는 이를 충실도 높은 음성 파형으로 최종 변환하는 역할을 수행합니다. 

## 9. 상세 데이터 흐름 및 텐서 변환

CosyVoice2 아키텍처에서 데이터는 여러 단계를 거치며 다양한 형태로 변환됩니다. 각 단계에서의 주요 텐서 모양(shape) 변화와 `unsqueeze`/`squeeze`와 같은 일반적인 연산에 초점을 맞춰 설명합니다. (`B`: 배치 크기, `T`: 시퀀스 길이(시간), `D`: 특징 차원, `S`: 다른 시퀀스 길이)

### 9.1. 데이터 로딩 및 전처리 (Dataset & Processor 파이프라인)

`cosyvoice.dataset.dataset.py`와 `cosyvoice.dataset.processor.py`의 함수들이 이 단계를 담당합니다.

1.  **`parquet_opener`**:
    *   Parquet 파일에서 데이터를 행 단위 딕셔너리로 읽어들입니다. (예: `audio_data` (bytes), `text` (str), `utt_embedding` (list/array), `spk_embedding` (list/array), `text_token` (list of int), `speech_token` (list of int)). 이 단계는 텐서가 아닙니다.

2.  **`filter`**:
    *   `torchaudio.load(BytesIO(sample['audio_data']))` → `speech` (오디오 파형), `sample_rate`.
        *   `speech` 초기 모양: `[채널 수, T_audio_raw]`. 보통 `speech.mean(dim=0, keepdim=True)`를 통해 `[1, T_audio_raw]`로 변환.
    *   길이 기반 필터링 수행.

3.  **`resample`**:
    *   `torchaudio.transforms.Resample`을 사용하여 `speech` 리샘플링.
    *   `speech` 모양: `[1, T_audio_resampled]`.
    *   `sample['speech'] /= max_val` (진폭 정규화).

4.  **`truncate`**:
    *   `speech`를 고정 길이 `truncate_length`로 자르거나 제로 패딩.
    *   `speech` 모양: `[1, truncate_length]`.

5.  **`compute_fbank`**:
    *   `feat_extractor(waveform)` (예: `torchaudio.transforms.MelSpectrogram`)
        *   입력 `waveform` (`speech`) 모양: `[1, truncate_length]`.
        *   `feat_extractor` 출력 `feat` 모양: `[1, D_mel, T_feat]` (채널, Mel 빈 수, 프레임 수).
        *   `.squeeze(dim=0).transpose(0, 1)` 적용 후, `speech_feat` 모양: `[T_feat, D_mel]`.
    *   `token_mel_ratio`에 따라 `feat` 및 `speech_token` (프롬프트용) 길이 조정.

6.  **`compute_f0`**:
    *   `pyworld`로 F0(pitch) 추출.
    *   `F.interpolate`로 F0 컨투어를 `speech_feat`의 시간 길이 `T_feat`에 맞게 보간.
    *   `pitch_feat` 모양: `[T_feat]`.

7.  **`parse_embedding`**:
    *   `utt_embedding`, `spk_embedding`을 `torch.tensor(..., dtype=torch.float32)`로 변환.
        *   `utt_embedding` 모양: `[D_embed]`.
        *   `spk_embedding` 모양: `[D_embed]`.
    *   `F.normalize(..., dim=0)`로 정규화.

8.  **`tokenize`**:
    *   `tokenizer.encode(...)`로 텍스트를 토큰 ID 시퀀스로 변환.
    *   `text_token` 모양: `[S_text]`.
    *   (추론 시) `tts_text_token` 모양: `[S_tts_text]`.

9.  **`padding`** (배치 후 최종 단계):
    *   배치 내 샘플들을 `pad_sequence(..., batch_first=True)`로 패딩하여 텐서 생성.
    *   **`speech`**: `[B, T_speech_padded]` (원래 `[1, T_speech]`였던 개별 샘플들이 배치화 및 패딩됨).
    *   **`speech_len`**: `[B]` (패딩 전 실제 길이).
    *   **`speech_token`** (프롬프트용): `[B, S_speech_token_padded]`.
    *   **`speech_token_len`**: `[B]`.
    *   **`speech_feat`** (Mel-spectrogram): `[B, T_feat_padded, D_mel]`.
    *   **`speech_feat_len`**: `[B]`.
    *   **`text_token`**: `[B, S_text_padded]`.
    *   **`text_token_len`**: `[B]`.
    *   **`utt_embedding`**: `[B, D_embed]` (`torch.stack` 사용).
    *   **`spk_embedding`**: `[B, D_embed]` (`torch.stack` 사용).
    *   **`pitch_feat`** (GAN 학습 시): `[B, T_feat_padded]` (패딩 후).
    *   **`pitch_feat_len`**: `[B]`.
    *   (추론 시) **`tts_text_token`**: `[B, S_tts_text_padded]`.
    *   **`embedding`** (실제 사용할 화자 임베딩): `[B, D_embed]` (`spk_embedding` 또는 `utt_embedding` 중 선택).

    *   **일반적인 `squeeze`/`unsqueeze`**:
        *   `compute_fbank`에서 Mel-spectrogram 계산 시 `squeeze(dim=0)`으로 불필요한 채널 차원 제거.
        *   `padding` 전 `speech`가 `[1, T_audio]`였다면, `.squeeze(dim=0)`으로 `[T_audio]`로 만들고 리스트에 담아 `pad_sequence`.

### 9.2. LLM (Large Language Model - `cosyvoice.llm.llm.Qwen2LM`)

*   **입력**:
    *   `text_token`: `[B, S_text_padded]` (데이터 로더 출력).
    *   `text_token_len`: `[B]`.
    *   화자 정보 (예: `embedding`): `[B, D_embed]`.
    *   (Zero-shot 시 참조 음성 토큰 `prompt_speech_token`: `[B, S_prompt_speech_token_padded]`).

*   **처리 과정**:
    1.  **텍스트 인코딩 (`Qwen2Encoder`)**:
        *   `text_token`을 임베딩 레이어 통과: `[B, S_text_padded, D_llm_hidden]`.
        *   Transformer 인코더 블록 통과 후 `xs` (텍스트 특징) 생성: `[B, S_text_padded, D_llm_hidden]`.
    2.  **음성 토큰 예측 (Transformer 디코더 기반)**:
        *   `xs`와 다른 조건(화자 임베딩, 프롬프트 토큰 등)을 받아 `speech_tokens`를 자기 회귀적으로 생성.
        *   생성된 `speech_tokens`는 이산적인 ID 시퀀스.

*   **출력**:
    *   `xs`: `[B, S_text_padded, D_llm_hidden]`.
    *   `speech_tokens`: `[B, S_gen_speech_token]` (생성된 음성 토큰의 길이).

### 9.3. Flow-matching 음향 모델 (`cosyvoice.flow.flow.CausalMaskedDiffWithXvec`)

*   **입력**:
    *   `xs`: `[B, S_text_padded, D_llm_hidden]` (LLM 출력).
    *   `speech_tokens`: `[B, S_gen_speech_token]` (LLM 출력).
    *   `xvector` (화자 임베딩, 데이터 로더의 `embedding`): `[B, D_embed]`.

*   **처리 과정**:
    1.  **인코더 (`UpsampleConformerEncoder`)**:
        *   `speech_tokens` (ID)를 임베딩 및 업샘플링: `[B, S_gen_speech_token, D_flow_embed]` → `[B, T_target_mel, D_flow_upsampled]`.
        *   `xs`도 필요시 길이/차원 변환 → `[B, T_target_mel, D_flow_condition]`.
        *   두 특징 결합 및 Conformer 통과 후 `encoder_out`: `[B, T_target_mel, D_flow_encoder_out]`.
    2.  **디코더 (`CausalConditionalCFM` - Flow Matching)**:
        *   입력: `encoder_out`, `xvector`.
        *   `xvector`는 `unsqueeze(1)` 후 `repeat` 되어 `[B, T_target_mel, D_embed]` 형태로 `encoder_out`의 각 시간 스텝에 조건으로 사용.
        *   초기 노이즈 `z`: `[B, T_target_mel, D_mel]` (목표 Mel-spectrogram과 동일 모양).
        *   Flow Matching (ODE 풀이 또는 반복적 정제)을 통해 `mel` 생성.
            *   내부 신경망에서 Conv 연산을 위해 `[B, T, D]` → `[B, D, T]` (`transpose`) 후 `unsqueeze`로 채널 차원 추가 가능.

*   **출력**:
    *   `mel`: `[B, T_target_mel, D_mel]` (예: `D_mel=80`).

### 9.4. HiFiGAN Vocoder (`cosyvoice.hifigan.generator.HiFTGenerator`)

*   **입력**:
    *   `mel`: `[B, T_target_mel, D_mel]`.
    *   (선택적) F0 컨투어 `pitch`: `[B, T_target_mel]`.

*   **처리 과정**:
    1.  `mel` 전치: `[B, D_mel, T_target_mel]` (Conv1D가 채널을 두 번째 차원으로 기대).
    2.  일련의 업샘플링 블록 (`ConvTranspose1d`) 및 잔차 블록 (`Conv1d`) 통과.
        *   `ConvTranspose1d`는 시간 차원 `T`를 증가시키고 채널 차원 `D`를 점진적으로 감소시킴.
    3.  최종 레이어에서 채널 수를 1로 줄이고 (`Conv1d` + `tanh`), 파형 생성.

*   **출력**:
    *   `waveform`: `[B, T_audio_output]`. (`T_audio_output`은 `T_target_mel`에 전체 업샘플링 비율을 곱한 값).
        *   출력이 `[B, 1, T_audio_output]`일 경우 `.squeeze(1)`으로 최종 모양 만듦.

