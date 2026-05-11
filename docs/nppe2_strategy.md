# NPPE-2 score improvement plan

Your current notebook is a weak baseline because it:

- uses zero-shot `whisper-medium`
- does not validate on a held-out split
- does not fine-tune on the competition transcripts
- lowercases predictions but does not properly clean common Whisper normalization errors

The biggest upgrade is not a tiny parameter tweak. It is:

1. create a validation split from `train.csv`
2. fine-tune Whisper on the 2,000 labeled examples
3. decode with stronger inference settings
4. submit only after checking local WER

Recommended order if the deadline is close:

1. Run a quick validation with `openai/whisper-large-v3-turbo`
2. Run the LoRA fine-tuning script in [`nppe2_kaggle_whisper_lora.py`](/E:/AI-Debate-Analyzer/docs/nppe2_kaggle_whisper_lora.py)
3. Compare holdout WER against your current approach
4. Submit the better one

Practical notes:

- If `whisper-small` fits and trains reliably, use that first.
- If you have more GPU headroom, try `openai/whisper-medium` with the same script.
- Keep the local validation split fixed so your comparisons are real.
- Watch for number formatting issues like `4` vs `four` and abbreviation issues like `ph.d.` vs `phd`.
- Do not trust Kaggle public score alone when choosing decoding settings.
