# Job Matching System

Elasticsearch-based job-to-candidate matching for the Seniors at Work platform. Ranks candidates by title relevance, industry, experience, skills, seniority fit, and education.

## Setup

1. **Copy environment file**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your MariaDB credentials, Elasticsearch URL, and OpenAI API key.

2. **Create virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Start Elasticsearch** (optional if using remote ES)
   ```bash
   docker compose up -d elasticsearch
   ```
   Wait until health is green/yellow: `curl http://localhost:9200/_cluster/health`

4. **Add standardized job titles**
   Place a file `standardized_titles.txt` in the project root (one title per line, ~2000 titles) for title standardization.

## Running

- **API server**
  ```bash
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  ```

- **One-time full index load** (extract from WordPress DB → transform → standardize titles → embeddings → index)
  ```bash
  python scripts/initial_load.py
  ```

- **Incremental sync** (only changed profiles since last run)
  ```bash
  python scripts/incremental_sync.py
  ```

## API Endpoints

| Method | Path | Description |
|--------|------|--------------|
| POST | `/api/match` | Match candidates for a job (JSON body: title, location, skills, etc.) |
| GET | `/api/jobs/{post_id}/matches` | Get matches for an already-indexed job |
| POST | `/api/index/candidates/sync` | Trigger delta sync from WordPress |
| POST | `/api/index/jobs/sync` | Sync job postings from WordPress |
| GET | `/api/health` | Elasticsearch and DB connectivity |
| GET | `/api/config` | Current scoring weights and threshold |
| PATCH | `/api/config` | Update weights/threshold (persisted to config.json) |

## Configuration

Scoring weights (defaults from golden dataset RE-ACC-001):

- Title relevance: 40%
- Industry relevance: 20%
- Recency-weighted experience: 15%
- Skills/tools: 10%
- Seniority fit: 8%
- Education: 7%

Default minimum score: 55/100. Candidates below this are excluded. Adjust via `PATCH /api/config` or by editing `config.json`.

## Testing

```bash
pytest tests/ -v
```

Golden dataset test: `tests/golden/test_golden_ranking.py` validates ranking for job RE-ACC-001 (Senior Real Estate Accountant) against 10 synthetic candidates.
