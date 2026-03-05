# Scoring overview

## Experience dimension

Experience is scored from three parts so that **total career length** is rewarded and candidates with more experience do not score lower than those with less:

1. **Primary role** – The single role with the highest weighted years. Scored with a sigmoid on primary years, gated by how well that role’s title matches the job title (and by a cap `min(1, primary_years / 3)`).
2. **Secondary roles** – All other roles combined (weighted years). Scored with a sigmoid and gated by overall title relevance, then scaled by 0.30.
3. **Total** – Total weighted relevant years across all roles. A third term `sigmoid(0.15 × total_years) × aggRel × 0.25` ensures that twice as much (weighted) experience contributes more to the score even when it is spread across many roles.

Recency weighting (years in role × recency factor by end year) has a **floor** (e.g. 0.38) for very old experience so that long careers are not over-penalized; re-index is required after changing recency in ETL.
