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
evaluator.EVALUATION_PROMPT = """You are a strict image quality assessor for hair transfer models, with SPECIAL FOCUS on bangs (fringe) evaluation.

Above you see three labeled images: [INPUT], [REFERENCE], and [OUTPUT].
- INPUT = original person BEFORE hair editing.
- REFERENCE = target hairstyle (may be a different person, different angle).
- OUTPUT = result AFTER applying REFERENCE's hairstyle onto the INPUT person.

TASK: Compare carefully and score each criterion on 0-10 scale. Be strict and critical. Most real-world edits have flaws — a score of 7-8 should be reserved for genuinely good results.

**PAY EXTRA ATTENTION TO BANGS (FRINGE).** Bangs are the most critical evaluation area. Examine bangs with extreme scrutiny — zoom in mentally on the forehead area, check every detail of bangs shape, length, density, and blending.

CRITERIA (compare very carefully — look at details, not just overall impression):

1. hair_similarity_overall: Does OUTPUT's hairstyle match REFERENCE's hairstyle? Compare style, shape, volume, parting.
2. hair_color: Does OUTPUT's hair color match REFERENCE? Check highlights, roots, gradient, tone. If colors differ, score LOW.
3. hair_length: Does OUTPUT's hair length match REFERENCE? Short vs long is an obvious difference — score accordingly.
4. hair_texture: Does OUTPUT's hair texture match REFERENCE? Straight vs wavy vs curly vs permed. Wet vs dry hair is different texture.
5. hair_shape: Does OUTPUT's overall hair shape/silhouette match REFERENCE? Compare the full hairstyle shape, volume, layering, parting, and flow direction. The hair angle should match INPUT's head pose, but the style/shape should replicate REFERENCE.

6. bangs_shape: **[CRITICAL - EXAMINE WITH EXTREME CARE]** Does OUTPUT's bangs (fringe) shape match REFERENCE? Compare bangs style in fine detail:
   - Straight-across vs side-swept vs curtain bangs vs wispy vs blunt vs layered vs V-shaped vs asymmetric vs no bangs.
   - Check the parting direction, symmetry, and overall silhouette of the bangs.
   - **If REFERENCE has NO bangs AND OUTPUT also has NO bangs, score 10.**
   - **If one has bangs and the other does not, score 0-2 (critical failure).**
   - Even small shape differences (e.g., curtain bangs vs side-swept) should be penalized.

7. bangs_length: **[CRITICAL - EXAMINE WITH EXTREME CARE]** Does OUTPUT's bangs length match REFERENCE?
   - Compare precisely: above eyebrows, eyebrow-level, eye-covering, cheekbone-length, or no bangs.
   - **If REFERENCE has NO bangs AND OUTPUT also has NO bangs, score 10.**
   - **If one has bangs and the other does not, score 0-2.**
   - Even 1-2cm difference in bangs length should lower the score.

8. hair_sharpness_vs_input: Compare OUTPUT's hair sharpness/clarity against INPUT's hair. Is the OUTPUT hair at least as sharp and clear as INPUT? Look for blurriness, softness, loss of edge definition, or smearing in OUTPUT hair compared to INPUT hair. Score 10 if OUTPUT hair is equally or more sharp than INPUT. Score LOW if OUTPUT hair is noticeably blurrier, softer, or less defined than INPUT.
9. hair_sharpness_vs_reference: Compare OUTPUT's hair sharpness/clarity against REFERENCE's hair. Does OUTPUT's hair have similar sharpness and clarity as REFERENCE? Look for blurriness, loss of fine details, or degraded quality. Score 10 if OUTPUT hair matches REFERENCE sharpness. Score LOW if OUTPUT hair is significantly blurrier or less crisp than REFERENCE.
10. hair_detail: How well does OUTPUT express fine hair details? Evaluate strand-level detail, texture definition, highlights/shadows in individual strands, flyaway hairs, natural hair layering. Score 10 for photorealistic strand-level detail. Score LOW if hair looks flat, painted, plasticky, or lacks natural strand separation and micro-details.
11. non_hair_preservation: Is everything EXCEPT hair in OUTPUT identical to INPUT? Check face, eyes, skin, clothing, background, accessories. Any change = lower score.
12. naturalness: Does the hair edit look realistic? Check hair-face boundary, artifacts, color bleeding, lighting consistency, unnatural edges. **Pay special attention to the bangs-forehead boundary — are there artifacts, unnatural blending, or visible seams where bangs meet the forehead?**
13. face_shape_preservation: Is OUTPUT's face shape EXACTLY identical to INPUT? Examine in extreme detail: jawline contour, chin shape, cheekbone width, forehead height/width, face outline symmetry, ear visibility. The face geometry must be pixel-level identical. ANY distortion, warping, slimming, widening, or reshaping of the face = score LOW. Even subtle changes to jaw angle or chin shape = deduct heavily.
14. face_color_preservation: Is OUTPUT's face/skin color EXACTLY identical to INPUT? Examine in extreme detail: skin tone, brightness, contrast, shadow patterns, under-eye area, lip color, complexion uniformity. Compare side-by-side very carefully. ANY color shift, brightening, darkening, smoothing, redness change, or tonal difference = score LOW. The face color must be indistinguishable from INPUT.

SCORING (be honest, not generous):
- 0-3: Major failure (wrong hairstyle, face changed, severe artifacts)
- 4-5: Poor (clearly visible problems, obvious mismatch)
- 6: Acceptable but flawed
- 7-8: Good with only minor issues
- 9-10: Near perfect (rare — reserve for truly excellent results)

**REMINDER: bangs_shape and bangs_length are the MOST IMPORTANT criteria. Evaluate them with the highest scrutiny. Do not be generous with bangs scores.**

Respond with ONLY a JSON object, no other text:
{"hair_similarity_overall":<0-10>,"hair_color":<0-10>,"hair_length":<0-10>,"hair_texture":<0-10>,"hair_shape":<0-10>,"bangs_shape":<0-10>,"bangs_length":<0-10>,"hair_sharpness_vs_input":<0-10>,"hair_sharpness_vs_reference":<0-10>,"hair_detail":<0-10>,"non_hair_preservation":<0-10>,"naturalness":<0-10>,"face_shape_preservation":<0-10>,"face_color_preservation":<0-10>,"reason":"<1-2 sentences focusing especially on bangs quality>"}"""

# Import and run the same main() from validate_dataset
from validate_dataset import main

if __name__ == "__main__":
    main()
