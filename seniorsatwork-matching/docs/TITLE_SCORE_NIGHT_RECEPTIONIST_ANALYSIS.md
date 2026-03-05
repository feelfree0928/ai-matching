# Why "Job Assistant" Can Outscore "Night Receptionist" for a Night Receptionist Job

## What you observed

- Hotel recruits **night receptionist** (e.g. "Nacht Rezeptionist/in").
- Candidate with title **Night Receptionist** gets **title raw value 1.48**.
- Candidate with title **Job Assistant** gets **title raw value 1.53**.
- So the less relevant title scores higher on the title dimension.

## Root cause (short)

The **job side "title" is not just the job title**. It is **title + industry + first 200 characters of required skills**. That blended text is embedded and compared to the candidate's **aggregated title embedding** (which is only the candidate's job title(s)). So we are comparing "Nacht Rezeptionist/in + industry + requirements snippet" to "Night Receptionist" or "Job Assistant". The requirements snippet often stresses support/office tasks (e.g. "Unterstützung", "Büro", "Assistenz"), so "Job Assistant" ends up closer in embedding space than the more specific "Night Receptionist".

## Where this happens in the code

1. **Job "title" vector**  
   Built in `api/matching.py` in `_job_embeddings()`:
   - `title_text = req.title`
   - then `+ " in " + industry_snippet` (if industry exists)
   - then `+ " . Key requirements: " + skills_snippet` (first 200 chars of `required_skills`)
   So the vector is for a **long, mixed string** (title + context + requirements), not for the job title alone.

2. **Candidate title vector**  
   In `embeddings/generator.py`, `aggregated_title_embedding` is a weighted average of embeddings of **only** job titles (standardized or raw) from work experience. So it represents "what roles this person had", not the full job description.

3. **Title score**  
   In `es_layer/queries.py` and `api/score_breakdown.py`, the **title** dimension is:
   - `titleSim = cosineSimilarity(job_title_vec, aggregated_title_embedding) + 1.0`
   So the "raw value for title" (e.g. 1.48 vs 1.53) is this similarity. It compares the **blended job text** to the **candidate's title(s) only**.

## Is it because everything is in German?

**Partly.**

- **Language mix**: If the job is in German ("Nacht Rezeptionist/in", "Unterstützung", "Assistenz") and the candidate title is standardized to English ("Night Receptionist"), we compare a German blob to an English title. Multilingual embeddings handle this, but "Night Receptionist" vs "Nacht Rezeptionist" is not identical in vector space, so you can lose a bit of similarity.
- **Main effect is still blending**: Even if everything were in the same language, the fact that the job vector contains 200 chars of requirements (often full of "support", "office", "assist") would still pull the vector toward "assistant"-like roles and can make "Job Assistant" score higher than "Night Receptionist" on the title dimension.

So: German vs English can worsen the effect, but the **primary cause** is that the **job "title" vector includes requirements text**, so it no longer represents "only the job title".

## Summary

| Factor | Effect |
|--------|--------|
| Job "title" = title + industry + 200 chars skills | **Main cause.** Job vector is broad; "Job Assistant" matches that broad description better than the narrow "Night Receptionist". |
| Candidate side: only title(s) | Fair; but then we compare "short title" to "long job description", which favours generic titles that overlap with the requirements wording. |
| German job vs English standardized titles | Can reduce similarity for "Night Receptionist" vs "Nacht Rezeptionist" and amplify the wrong winner. |

## Recommended fixes

1. **Use a title-only vector for the title dimension**  
   For the **title** score, compare:
   - job: embedding of **only** `req.title` (and optionally a short industry phrase if desired),
   - candidate: `aggregated_title_embedding` (and/or `primary_role_title_embedding`).
   Keep the enriched "title + requirements" vector for something else (e.g. a separate "role fit" or "skills-in-title" signal) if you want, but do **not** use it as the main title similarity. That way "Nacht Rezeptionist" vs "Night Receptionist" / "Nacht Rezeptionist" is compared directly, and "Job Assistant" will no longer get an artificial boost from the requirements text.

2. **Optional: same language on both sides**  
   If job titles in the DB are German, prefer standardizing to German titles (or keep raw German) so the candidate title embedding is in the same language as the job title. That avoids cross-lingual noise on top of the blending issue.

3. **Optional: boost exact / close title match**  
   Add a small bonus when the candidate's primary (or aggregated) title string is very close to the job title (e.g. after normalization), so that "Nacht Rezeptionist" vs "Nacht Rezeptionist/in" or "Night Receptionist" is explicitly rewarded even if the embedding similarity is slightly lower.

Implementing (1) is the critical fix; (2) and (3) further improve accuracy for German data and for exact-title matches.

---

## Implemented solution (no re-embedding)

The following is now in place so the issue is fully addressed **without re-embedding** candidates or jobs:

1. **Title-only job vector**  
   The job “title” vector used for the title dimension is now **only** the job title (optionally normalized by an LLM to remove “/in”, “(m/w/d)”, etc.). No industry or skills text is appended. So we compare job title ↔ candidate title directly.

2. **LLM job-title normalization**  
   Before embedding the job title, we call an LLM to normalize it (e.g. “Nacht Rezeptionist/in (m/w/d)” → “Nacht Rezeptionist”). This does not depend on `standardized_titles.txt` and keeps the embedded title clean and comparable to candidate titles.

3. **LLM title-fit scoring**  
   After Elasticsearch returns the shortlist, we call an LLM once (batch) to score each candidate’s main role title vs the job title (0–10). That score is blended into the final ranking (e.g. 80% ES score + 20% LLM title fit), so “Night Receptionist” is pushed above “Job Assistant” for a night receptionist job. This does not use the standardized list and does not require any re-embedding.

Code: `api/title_match.py` (normalize + batch title-fit), `api/matching.py` (`_job_embeddings` title-only, post-ES LLM step and re-sort). The API returns `score.llm_title_fit` (0–10) per candidate when available.
