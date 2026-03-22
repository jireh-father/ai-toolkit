"""Bangs (fringe) focused dataset validation.

Same pipeline as validate_dataset.py, but with a prompt specifically
designed to evaluate bangs quality more strictly.

Usage:
    python dataset_validator/validate_dataset_bangs.py \
        --input-dir ./data/input \
        --reference-dir ./data/reference \
        --output-dir ./data/output \
        --model qwen3-vl-8b \
        --report-dir ./reports_bangs
"""

import core.evaluator as evaluator

# Override the evaluation prompt with bangs-focused version
# All score fields remain the same (14 fields), but the prompt emphasizes bangs evaluation
evaluator.EVALUATION_PROMPT = """You are a strict image quality assessor for hair transfer models.

Above you see three labeled images: [INPUT], [REFERENCE], and [OUTPUT].
- INPUT = original person BEFORE hair editing.
- REFERENCE = target hairstyle (may be a different person, different angle).
- OUTPUT = result AFTER applying REFERENCE's hairstyle onto the INPUT person.

TASK: Compare carefully and score each criterion on 0-10 scale. Be strict and critical. Most real-world edits have flaws — a score of 7-8 should be reserved for genuinely good results.

CRITERIA (compare very carefully — look at details, not just overall impression):

1. hair_similarity_overall: Does OUTPUT's hairstyle match REFERENCE's hairstyle? Compare style, shape, volume, parting.
2. hair_color: Does OUTPUT's hair color match REFERENCE? Check highlights, roots, gradient, tone. If colors differ, score LOW.
3. hair_length: Does OUTPUT's hair length match REFERENCE? Short vs long is an obvious difference — score accordingly.
4. hair_texture: Does OUTPUT's hair texture match REFERENCE? Straight vs wavy vs curly vs permed. Wet vs dry hair is different texture.
5. hair_shape: Does OUTPUT's overall hair shape/silhouette match REFERENCE? Compare the full hairstyle shape, volume, layering, parting, and flow direction. The hair angle should match INPUT's head pose, but the style/shape should replicate REFERENCE.
6. hair_sharpness_vs_reference: Compare OUTPUT's hair sharpness/clarity against REFERENCE's hair. Does OUTPUT's hair have similar sharpness and clarity as REFERENCE? Look for blurriness, loss of fine details, or degraded quality. Score 10 if OUTPUT hair matches REFERENCE sharpness. Score LOW if OUTPUT hair is significantly blurrier or less crisp than REFERENCE.
7. hair_detail: How well does OUTPUT express fine hair details? Evaluate strand-level detail, texture definition, highlights/shadows in individual strands, flyaway hairs, natural hair layering. Score 10 for photorealistic strand-level detail. Score LOW if hair looks flat, painted, plasticky, or lacks natural strand separation and micro-details.
8. naturalness: Does the hair edit look realistic? Check hair-face boundary, artifacts, color bleeding, lighting consistency, unnatural edges.

SCORING (be honest, not generous):
- 0-3: Major failure (wrong hairstyle, face changed, severe artifacts)
- 4-5: Poor (clearly visible problems, obvious mismatch)
- 6: Acceptable but flawed
- 7-8: Good with only minor issues
- 9-10: Near perfect (rare — reserve for truly excellent results)

Respond with ONLY a JSON object, no other text:
{"hair_similarity_overall":<0-10>,"hair_color":<0-10>,"hair_length":<0-10>,"hair_texture":<0-10>,"hair_shape":<0-10>,"hair_sharpness_vs_reference":<0-10>,"hair_detail":<0-10>,"naturalness":<0-10>,"reason":"<1-2 sentences>"}"""

# Import and run the same main() from validate_dataset
from validate_dataset import main

if __name__ == "__main__":
    main()
