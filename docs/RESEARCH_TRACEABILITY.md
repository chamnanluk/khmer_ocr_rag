# Research Traceability

The repository architecture follows the revised proposal and is grounded in the research sources reviewed for the project.

Key implementation implications:

- Khmer OCR requires treatment of stacked characters, diacritics, ligatures, non-uniform character widths, and absent obligatory word delimiters. The project therefore preserves Unicode/KCC-aware representations instead of assuming space-delimited Latin tokens.
- The joint recognizer exposes U+200B as an output boundary token and a boundary-control flag so joint and non-joint decoding can share a recognition backbone.
- Retrieval is modular (sparse, dense, hybrid), because the research question asks whether the same upstream error affects each retrieval family differently.
- Controlled perturbations are first-class data artifacts, because general OCR→RAG research indicates that downstream harm is category-dependent and conventional CER can conceal semantic/structural damage.
- RAG components are versioned and frozen inside comparisons to avoid confounding upstream OCR changes with downstream model changes.

## Project references (IEEE style)

[1] R. Buoy, M. Iwamura, S. Srun, and K. Kise, “Toward a low-resource non-Latin-complete baseline: An exploration of Khmer optical character recognition,” *IEEE Access*, vol. 11, pp. 128044–128060, 2023, doi: 10.1109/ACCESS.2023.3332361.

[2] M. Kong, R. Buoy, S. Chenda, N. Taing, M. Iwamura, and K. Kise, “Towards a joint Khmer text recognition and word segmentation,” arXiv:2608.30213, 2026.

[3] M. Kong, R. Buoy, S. Chenda, N. Taing, M. Iwamura, and K. Kise, “Towards universal Khmer text recognition,” arXiv:2603.00702, 2026, doi: 10.48550/arXiv.2603.00702.

[4] S. Ros, P. Pov, R. Chhor, K. Ly, W.-S. Cho, and S. Khoeurn, “A comparative study of language models for Khmer retrieval-augmented question answering,” arXiv:2605.22099, 2026, doi: 10.48550/arXiv.2605.22099.

[5] J. Zhang et al., “OCR hinders RAG: Evaluating the cascading impact of OCR on retrieval-augmented generation,” in *Proc. IEEE/CVF ICCV*, 2025, pp. 17443–17453, doi: 10.1109/ICCV51701.2025.01620.

[6] L. Sun et al., “When good OCR is not enough: Benchmarking OCR robustness for retrieval-augmented generation,” 2026, doi: 10.18653/v1/2026.acl-industry.60.

[7] J. Jin, Y. Zhu, X. Yang, C. Zhang, and Z. Dou, “FlashRAG: A modular toolkit for efficient retrieval-augmented generation research,” in *Companion Proc. ACM Web Conf.*, 2025, doi: 10.1145/3701716.3715313.
