
3. Method
This section details the architecture and training methodology of CosyVoice 2, a unified framework for both streaming and non-streaming zero-shot text-to-speech (TTS) synthesis. The system comprises four key modules: the Text Tokenizer, Supervised Semantic Speech Tokenizer (FSQ-based), Unified Text-Speech Language Model (LM), and the Chunk-aware Causal Flow Matching module. We also describe the streaming sequence construction strategy, the instructed TTS control mechanism, and the classifier-free guidance setup.

3.1 Text Tokenizer and Supervised Semantic Speech Tokenizer
3.1.1 Text Tokenizer
The Text Tokenizer is a BPE-based tokenizer that converts raw text input into discrete tokens. Unlike conventional pipelines requiring grapheme-to-phoneme (G2P) conversion or explicit phoneme-level preprocessing, our tokenizer directly operates on raw text, enabling end-to-end learning of pronunciation within context.

To mitigate issues arising from excessive token granularity, such as Chinese BPE tokens covering multiple characters, the tokenizer masks multi-character tokens and splits them into individual sub-tokens. This strategy prevents overly long token durations and improves model robustness for languages like Chinese, while no such masking is applied for English, Japanese, or Korean.

3.1.2 Supervised Semantic Speech Tokenizer (FSQ-based)
Architecture Overview
The speech tokenizer is designed to extract semantic-level discrete speech representations from raw audio 𝑋 ∈ 𝑅 𝑇 × 𝐷 𝑟 𝑎 𝑤. The pipeline consists of:
Encoder1 → Finite Scalar Quantization (FSQ) → Encoder2 → ASR Decoder
Here, Encoder2 and the ASR Decoder are used only during training to optimize tokenization quality via auxiliary ASR loss.

Finite Scalar Quantization (FSQ) Mechanism
FSQ replaces traditional vector quantization (VQ) to overcome codebook utilization issues and information bottlenecks. The procedure is as follows:

Down-Projection: Intermediate feature 𝐻 ∈ 𝑅 𝑇 × 𝐷 ℎ is linearly projected to a lower-dimensional space:
𝐻ˉ = ROUND(Proj down(H)) ∈ Z 𝑇 × 𝐷 𝐹𝑆𝑄
where ℎˉ𝑖,𝑗 ∈ [−𝐾,𝐾] are integer-quantized values within a bounded range, and ROUND denotes element-wise rounding.

Up-Projection: The quantized features are mapped back to the original space:
𝐻^ = Proj up(𝐻ˉ)
Token Index Computation: Each semantic speech token 𝜇𝑖 is computed as:
𝜇𝑖 = ∑𝑗=0 𝐷 𝐹𝑆𝑄−1 ℎˉ𝑖,𝑗 ⋅ (2𝐾+1)𝑗

Training Strategy
The FSQ module employs the Straight-Through Estimator (STE) to approximate gradients across the non-differentiable rounding operation. Encoder1 is a 6-layer Transformer with Rotary Position Embeddings (RoPE).

The semantic speech tokenizer operates at 25Hz, producing one token every 40ms.

3.2 Unified Text-Speech Language Model (LM)
3.2.1 Design
The Unified Text-Speech LM predicts mixed sequences of text tokens and semantic speech tokens within a single auto-regressive framework. It leverages a Qwen2.5-0.5B pre-trained language model as backbone, removing both the text encoder and the speaker embedding components of prior models (e.g., CosyVoice 1). This simplification:
Eliminates potential information leakage from speaker embeddings, which may inadvertently carry language and paralinguistic information.
Fully relies on the language model's capacity to learn contextual alignment between text and speech tokens.

3.2.2 Input-Output Sequence Structure
Non-Streaming Mode:
The input sequence 𝑆 is structured as:
𝑆 = {𝑆, 𝑡1,…,𝑡𝑁, 𝑇, 𝜇1,…,𝜇𝑀, 𝐸}
where 𝑆 = start-of-sequence token, 𝑇 = turn-of-speech token, 𝐸 = end-of-sequence token.

Streaming Mode:
The sequence alternates text and speech tokens in an 𝑁:𝑀 ratio (typically 𝑁=5, 𝑀=15). A Filling Token is used as a placeholder when the next text token is not yet available during streaming inference.

3.2.3 Loss Function
Cross-entropy loss is computed over next-token prediction, with Ignore Tokens excluded from loss computation. Both streaming and non-streaming modes are trained jointly within a single model.

3.3 Chunk-aware Causal Flow Matching
3.3.1 Problem Formulation
The Flow Matching module transforms semantic speech tokens 𝜇 into Mel-spectrogram frames 𝑋 ∈ 𝑅 𝑇 × 𝐷 𝑀𝑒𝑙, conditioned on a speaker embedding 𝑣 and a reference Mel input 𝑋~1. The task is framed as learning the probabilistic density flow:
𝜔𝑡(𝜙𝑡𝑂𝑇(𝑋0,𝑋1)∣𝑋1) = 𝑋1−𝑋0
where:
𝜙𝑡𝑂𝑇(𝑋0,𝑋1) = (1−𝑡)𝑋0+𝑡𝑋1
with 𝑋0∼𝑝0=𝑁(0,𝐼) and 𝑋1∼𝑞(𝑋).

3.3.2 Model Architecture
The Flow Matching network is a causal stack of modules:
- Lookahead PreConv: A 1D convolution with right-padding (kernel size 𝑃+1, padding 𝑃) introduces future context.
- Causal Upsampling Transformer: Upsamples the semantic speech token sequence to match the Mel-spectrogram frame rate (50Hz).
- Causal Transformer Encoder: Aligns temporal structures.
- Causal Conv-Transformer UNet: A 10-block residual architecture predicting 𝑋𝑡 at each time step.

3.3.3 Training Objective
The loss is formulated as:
𝜃∗ = argmin𝜃 𝐸𝑝0,𝑞,𝑡 ∥𝜔𝑡−𝜈𝑡(𝜙𝑡𝑂𝑇(𝑋0,𝑋1)∣𝜃;𝜇,𝑋~1,𝑣)∥1

Classifier-Free Guidance (CFG) is integrated for controllable generation:
𝜈~𝑡 = (1+𝛽)𝜈𝑡(⋅∣Ψ)−𝛽𝜈𝑡(⋅)
where Ψ={𝜇,𝑋~1,𝑣} and 𝛽=0.7.

3.3.4 Masking Strategies
To balance latency and quality, four masking schemes are sampled uniformly during training:
- Non-Causal Mask: All frames (past & future) attendable; for offline high-quality generation.
- Full-Causal Mask: Past-only attention; for real-time low-latency scenarios.
- Chunk-M Mask: Past + M-future frames attendable; for first-chunk generation.
- Chunk-2M Mask: Past + 2M-future frames; for later chunks with improved quality.

3.3.5 Inference Strategy
Time steps 𝑡 are scheduled using a cosine scheduler:
𝑡 := 1−cos(𝜋/2 𝑡)
Number of Flow Matching iterations (NFE) is set to 10.
During inference, masked Mel frames are sampled from reference audio.

3.4 Streaming Sequence Construction and Instructed TTS Control
3.4.1 Streaming Sequence Strategy
Text and speech tokens are alternated at an 𝑁:𝑀 ratio (typically 𝑁=5, 𝑀=15).
The model generates 𝑀 speech tokens per inference cycle, inserting Filling Tokens as placeholders for unavailable text tokens.

3.4.2 Instructed TTS Control
CosyVoice 2 supports instruction-based synthesis by embedding explicit prompts within the text input (e.g., "Speak in a sad tone."). This enables fine-grained control over emotion, prosody, accent, and speaking style. Control is unified within the same model without separate fine-tuning or additional modules, and is compatible with both streaming and non-streaming modes.

3.5 Hyperparameters and Implementation Details
Sampling rate: 24kHz.
Mel-spectrogram frame rate: 50Hz.
Semantic speech token rate: 25Hz (upsampled 2×).
Loss functions: Cross-entropy (LM), L1 (Flow Matching).
CFG strength: 𝛽=0.7.
Flow Matching steps (NFE): 10.
Masking strategies: Uniformly sampled per batch.
