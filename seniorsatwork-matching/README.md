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
   source .venv/bin/activate # Linux / macOS
   pip install -r requirements.txt
   ```

3. **Install and start Elasticsearch** (optional if using a remote ES instance)

   See [Installing Elasticsearch (without Docker)](#installing-elasticsearch-without-docker) below. Then ensure the cluster is up:
   ```bash
   curl http://localhost:9200/_cluster/health
   ```
   Wait until health is `green` or `yellow`.

4. **Add standardized job titles**
   Place a file `standardized_titles.txt` in the project root (one title per line, ~2000 titles) for title standardization.

---

### Installing Elasticsearch (without Docker)

Install Elasticsearch 8.x natively (no Docker). Set `ELASTICSEARCH_URL` in `.env` to `http://localhost:9200` (or your ES URL).

**Windows**

1. Download the [Elasticsearch Windows ZIP](https://www.elastic.co/downloads/elasticsearch) (e.g. 8.12 or later).
2. Unzip to a folder (e.g. `C:\Elasticsearch`). Avoid paths with spaces.
3. Open a terminal in that folder and run:
   ```powershell
   .\bin\elasticsearch.bat
   ```
   Or run `bin\elasticsearch.bat` as a Windows service if you prefer.
4. Defaults: HTTP at `http://localhost:9200`. For 8.x, note the printed `elastic` user password or reset it with `bin\elasticsearch-reset-password -u elastic`. If you use security, set `ELASTICSEARCH_URL` to `https://localhost:9200` and configure credentials in your app.

**Linux (Debian/Ubuntu)**

```bash
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
sudo apt update && sudo apt install -y elasticsearch
sudo systemctl enable elasticsearch && sudo systemctl start elasticsearch
```

**Linux (RHEL/CentOS)**

```bash
sudo rpm --import https://artifacts.elastic.co/GPG-KEY-elasticsearch
sudo tee /etc/yum.repos.d/elasticsearch.repo << 'EOF'
[elasticsearch]
name=Elasticsearch repository for 8.x packages
baseurl=https://artifacts.elastic.co/packages/8.x/yum
gpgcheck=1
gpgkey=https://artifacts.elastic.co/GPG-KEY-elasticsearch
enabled=1
autorefresh=1
type=rpm-md
EOF
sudo yum install -y elasticsearch
sudo systemctl enable elasticsearch && sudo systemctl start elasticsearch
```

**macOS**

```bash
brew tap elastic/tap
brew install elastic/tap/elasticsearch-full
brew services start elasticsearch-full
```

**Verify**

```bash
curl http://localhost:9200
```

You should get a JSON response with `version.number`. For 8.x with security on, use HTTPS and the `elastic` user (or add `ELASTICSEARCH_URL` and auth to `.env` as needed).

---

## Running on Ubuntu

End-to-end steps to run the project on Ubuntu (no Docker).

**1. Prerequisites**

- Ubuntu 22.04 or 24.04 (or similar)
- Python 3.11+ (`sudo apt install python3.11 python3.11-venv python3-pip`)
- Access to a MariaDB/MySQL instance (WordPress DB) and an OpenAI API key

**2. Install Elasticsearch**

```bash
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
sudo apt update && sudo apt install -y elasticsearch
sudo systemctl enable elasticsearch && sudo systemctl start elasticsearch
```

Wait a few seconds, then check:

```bash
curl http://localhost:9200
```

**3. Clone and enter the project**

```bash
cd /path/to/ai-matching/seniorsatwork-matching
```

**4. Python virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**5. Environment file**

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME` (WordPress/MariaDB)
- `ELASTICSEARCH_URL=http://localhost:9200` (use `https://...` for ES 8.x with security)
- `ELASTICSEARCH_USER` and `ELASTICSEARCH_PASSWORD` (optional; required if ES 8.x has security enabled)
- `ELASTICSEARCH_VERIFY_CERTS=false` (optional; use only for local ES 8.x with self-signed cert—not for production)
- `OPENAI_API_KEY=sk-...`

**6. Standardized titles**

Ensure `standardized_titles.txt` exists in the project root (one job title per line). The repo may include a small example; add more titles for production.

**7. Load candidates (one-time)**

```bash
python scripts/initial_load.py
```

Use `--test` for a small run, or `--limit 500` to cap the number of candidates.

**8. Start the API**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/health`

**9. Optional: incremental sync**

To refresh only changed candidates (e.g. from cron):

```bash
source .venv/bin/activate
python scripts/incremental_sync.py
```

---

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

- **Full reset and re-embed** (e.g. after stopping initial_load partway, or to clear all caches)
  Removes the embedding cache, title-mapping cache, and the Elasticsearch candidates/job_postings indices so the next `initial_load.py` runs from scratch.
  ```bash
  python scripts/reset_caches_and_index.py
  python scripts/initial_load.py
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

## Troubleshooting

**"Connection error" / "Remote end closed connection" when running `initial_load.py`**

- Ensure Elasticsearch is running: `sudo systemctl status elasticsearch` (Linux) or check your ES process.
- Test from the same machine: `curl http://localhost:9200` (or the URL in `ELASTICSEARCH_URL`). You should get JSON with `version.number`.
- If using Elasticsearch 8.x with security enabled, set `ELASTICSEARCH_URL=https://localhost:9200` and set `ELASTICSEARCH_USER` and `ELASTICSEARCH_PASSWORD` in `.env` (e.g. user `elastic` and the password printed at first start or reset with `bin/elasticsearch-reset-password -u elastic`).

**"SSL: CERTIFICATE_VERIFY_FAILED" / "self-signed certificate in certificate chain"**

- Elasticsearch 8.x uses a self-signed certificate by default. For local/dev, you can skip verification by adding to `.env`:
  ```bash
  ELASTICSEARCH_VERIFY_CERTS=false
  ```
- **Do not use `ELASTICSEARCH_VERIFY_CERTS=false` in production.** For production, use proper CA certificates or `ssl_assert_fingerprint`.

## Testing

```bash
pytest tests/ -v
```

Golden dataset test: `tests/golden/test_golden_ranking.py` validates ranking for job RE-ACC-001 (Senior Real Estate Accountant) against 10 synthetic candidates.
