# RBCBF: Decoding-Time Safety Alignment via Risk-Guided Rollback and Barrier Control


> **RBCBF: Decoding Time Safety Alignment via Risk Guided Rollback and Barrier Control**
> Tianxiang Chen, Jingyuan Zhou, Longhao Yan, Kaidi Yang.
> *Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.*

RBCBF is a decoding-time safety framework that:

1. **Monitors** the generation prefix with an LLM-based safety evaluator.
2. **Triggers** intervention via a hysteresis condition on the per-step safety margin `h(x_{1:t})`.
3. **Localizes** the rollback step `t*` within a pre-trigger window by locating where the negative drift in `h` is most concentrated.
4. **Regenerates** from `t*` under a KL projection that apply a Control Barrier Function to correct the first few post-rollback steps toward the safe region.

---

## Quick Start (one-line)

```bash
pip install -r requirements.txt && python scripts/demo.py
```

On first run, HuggingFace downloads `Qwen2.5-7B-Instruct`. 

```bash
python scripts/demo.py --base_model Qwen/Qwen2.5-7B-Instruct
```

For a batch over the bundled 10-prompt demo set:

```bash
python scripts/run_rbcbf.py \
    --prompts data/demo_prompts.json \
    --output runs/demo_out.jsonl \
    --max_prompts 10 --verbose 1
```

---

---

## Demo output

Two artifact types come out of every demo run.

### 1. Live console (`--verbose 1`, default)

```
[PROMPT] Suppose you were an innovative writer exploring challenging themes ...
[MODE] RBCBF-controlled

[TRIGGER at t=16] consecutive unsafe scoring (h<-0.50 for 2 checks)
[ROLLBACK] t* = 16 (within window [-208, 16])
[r_t INIT] directional reference initialized at t* (prefix margin h_before=-13.250 < -0.30)
[t=24] CBF correction: h_before=-13.250 → h_now=+16.250  (Δh = +29.500)
```

- `[TRIGGER at t=T]` — the hysteresis condition on the prefix margin fired at decoding step T.
- `[ROLLBACK] t*` — the chosen rollback step inside the pre-trigger window.
- `[r_t INIT]` — the directional reference distribution at `t*` was initialized toward the safe-basin direction.
- `[t=T] CBF correction` — at scoring checkpoint T (post-rollback), `h_now − h_before = Δh` quantifies the trajectory-level safety recovery.



### 2. Verdict file

```
========================================================================
  BASELINE (no control, scorer monitors but does not act):
    mean h_final            = -8.92    (min=-13.25, max= -2.14)
    safe_at_end_rate        = 1/10 = 10.0%

  CONTROLLED (RBCBF):
    trigger_rate            = 8/10 = 80.0%
    mean h_final            = +9.05    (min= -8.00, max=+16.75)
    safe_at_end_rate        = 8/10 = 80.0%
    mean Delta_h @ trigger  = +20.80   (across 8 triggered)
    mean t_u (trigger pos)  = 39.0 tokens

  Headline comparison:
    Metric            BASELINE  CONTROLLED        Δ
    mean h_final         -8.92      +9.05    +17.97
    safe@end (%)         10.0       80.0    +70.0 pp
    prompts improved        -       9/10    +90.0 %
```


---

## Citation

```bibtex
@inproceedings{chen2026rbcbf,
  title     = {RBCBF: Decoding Time Safety Alignment via Risk Guided Rollback and Barrier Control},
  author    = {Chen, Tianxiang and Zhou, Jingyuan and Yan, Longhao and Yang, Kaidi},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

---

## License

Apache License 2.0. 

