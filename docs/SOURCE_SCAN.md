# Source Scan and Implementation Relevance

This scan is used to decide what belongs in the implementation and what should remain background only.

## Primary technical sources

### 1. Towards a Joint Khmer Text Recognition and Word Segmentation (Kong et al., 2026)
Implementation relevance: **highest**.

Supported design details used in this repository:
- unified Khmer text recognition + word-boundary prediction;
- CTC decoding for parallel output;
- U+200B boundary marker in the augmented vocabulary;
- inference control flag `b=0/1` for output without/with boundaries;
- visual encoder + modality-aware feature adaptation + boundary conditioning;
- evaluation across document, scene, and handwriting modalities;
- Tier B datasets summarized in the proposal: KHOB, KhmerST, WildKhmerST, KH, GKST, KHT.

Repository mapping: `ocr/joint_ctc.py`, `configs/ocr/joint_ctc.yaml`, P2/P3 experiment design.

### 2. Toward a Low-Resource Non-Latin-Complete Baseline: An Exploration of Khmer OCR (Buoy et al., 2023)
Implementation relevance: **high**.

Supported design implications:
- preserve spatial detail for Khmer subscripts/diacritics;
- avoid Latin-only assumptions about fixed-width word images;
- maintain aspect ratio and support long text lines;
- use Khmer-aware sequence representation and normalization.

Repository mapping: Tier A renderer contract, KCC/Unicode modules, OCR data-loader backlog.

### 3. Explainable CTC-Based Scene Text Recognition (Buoy et al., 2023)
Implementation relevance: **medium/high**.

Supported design implications:
- CTC is a strong low-latency decoder;
- 2D feature representations can be reconciled with 1D CTC decoding;
- timing/latency is a legitimate first-class metric.

Repository mapping: RQ1 timing utilities and CTC-oriented model interface.

### 4. ViTSTR-Transducer (Buoy et al., 2023)
Implementation relevance: **medium**.

Supported design implications:
- efficient decoding without cross-attention can preserve competitive recognition quality;
- latency/FLOPs comparisons should be measured rather than assumed.

Repository mapping: RQ1 efficiency metrics and future model-adapter path.

## Primary downstream sources from Consensus

### 5. A Comparative Study of Language Models for Khmer Retrieval-Augmented Question Answering (Ros et al., 2026)
Implementation relevance: **high**.

Supported design implications:
- Khmer RAG should be evaluated at both retrieval and generation levels;
- retriever choice is an important bottleneck;
- BGE-M3 is a reasonable *initial* dense-retrieval candidate because it was strongest among that study's tested retrievers, but the repository keeps the model configurable.

Repository mapping: `configs/retrieval/dense.yaml`, RQ2/RQ4 metrics.

### 6. OCR Hinders RAG / When Good OCR Is Not Enough
Implementation relevance: **high for research design**.

Supported design implications:
- conventional CER/WER can fail to predict downstream RAG utility;
- OCR corruption should be analyzed by type and severity;
- retrieval-side and generation-side failures should be separated.

Repository mapping: controlled corruption, P0–P5, edit logs, frozen RAG configuration.

### 7. FlashRAG
Implementation relevance: **architectural inspiration only**.

Supported design implication:
- modular retriever/generator/evaluator interfaces improve reproducibility and comparison.

Repository mapping: separate `retrieval/`, `rag/`, `metrics/`, and experiment configs.

## Supporting/background-only uploaded sources

### Benchmarking of DIA Tasks for Palm Leaf Manuscripts
Use: historical context and prior Khmer DIA benchmark background. **Not part of the main Tier A/Tier B implementation**, because historical manuscripts were explicitly removed from the empirical scope.

### Storytelling in the Heritage Language: Khmer Language in Cambodia
Use: societal/cultural motivation only. **No model behavior, metric, or code path should be derived from it.**

### KhmerWriterID
Use: neighboring Khmer handwriting/biometric research. It is **not a recognition-to-retrieval baseline** and should not be included in the core pipeline unless a later side study specifically needs writer-disjoint handwriting analysis.

### LLM-generated text detection paper
Use: unrelated to the four current RQs. **Excluded from implementation.**

## Proposal-derived controls that Codex must preserve

- Tier A = controlled modern Khmer.
- Tier B-OCR = real modern document/scene/handwriting OCR.
- Tier B-RAG = separate real modern document corpus with passage-level context.
- P0–P5 oracle design.
- Sparse/dense/hybrid retrieval.
- 0/1/2/5/10/20% controlled error curves.
- matched error-type experiment near 5%.
- fixed downstream RAG configuration inside each upstream comparison.
- 100–150 human-validated answers where feasible.
- historical Khmer is future work, not an implied validation domain.
