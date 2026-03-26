# PACE-KG Backend Deployment Guide

Complete step-by-step guide to deploy the PACE-KG backend on Vast.ai RTX 3090 GPU with Neo4j AuraDB and Groq API integration.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Neo4j AuraDB Setup](#neo4j-auradb-setup)
3. [Groq API Setup](#groq-api-setup)
4. [Vast.ai Instance Setup](#vastai-instance-setup)
5. [Backend Deployment](#backend-deployment)
6. [Testing the Pipeline](#testing-the-pipeline)
7. [Frontend Connection](#frontend-connection)
8. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
9. [Security Best Practices](#security-best-practices)

---

## Prerequisites

Before starting, you need:

1. **Groq API account** (free tier available)
2. **Neo4j AuraDB instance** (free tier available)
3. **Vast.ai account** with GPU instance
4. **Local development machine** for frontend (Windows/Linux/Mac)
5. **Git** installed locally

---

## Neo4j AuraDB Setup

### Step 1: Create AuraDB Account

1. Go to https://neo4j.com/aura
2. Click **"Start Free"**
3. Sign up with email or GitHub account
4. Verify your email

### Step 2: Create a Free Instance

1. After login, click **"Create Instance"**
2. Select **"AuraDB Free"** tier:
   - **Region**: Choose closest to your Vast.ai location (e.g., `us-east-1`)
   - **Memory**: 1GB (sufficient for small-medium PDFs)
   - **Instance name**: `pace-kg-prod` (or any name)
3. Click **"Create Instance"**

### Step 3: Save Connection Credentials

After creation, Neo4j will show a dialog with:

```
Connection URI: neo4j+s://xxxxx.databases.neo4j.io
Username: neo4j
Password: <generated-password>
```

**CRITICAL**: Copy these credentials immediately. The password is shown **only once**.

If you lose the password:
1. Go to your instance in the dashboard
2. Click **"..."** → **"Reset password"**
3. A new password will be generated

### Step 4: Configure Network Access

1. In the Neo4j dashboard, click your instance
2. Go to **"Network Access"** tab
3. Add IP allowlist:
   - If using Vast.ai: Add your Vast.ai instance IP (you'll get this later)
   - For testing: Add `0.0.0.0/0` (allows all IPs — **remove after testing!**)
4. Click **"Add IP"**

### Step 5: Verify Connection

Once your instance is running (green status):

1. Click **"Query"** or **"Open with Neo4j Browser"**
2. Test with a simple query:
   ```cypher
   MATCH (n) RETURN count(n) AS total
   ```
3. Should return `0` (empty database)

**Save these credentials** — you'll need them in the `.env` file:
```
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-generated-password>
```

---

## Groq API Setup

### Step 1: Create Groq Account

1. Go to https://console.groq.com
2. Click **"Sign Up"** (or sign in with Google/GitHub)
3. Complete email verification

### Step 2: Generate API Key

1. After login, go to **"API Keys"** in the left sidebar
2. Click **"Create API Key"**
3. Give it a name: `pace-kg-production`
4. Click **"Create"**

The API key format is: `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**CRITICAL**: Copy the API key immediately. It's shown **only once**.

If you lose it:
1. Delete the old key
2. Create a new one

### Step 3: Verify Rate Limits

Free tier limits (as of 2024):
- **Requests per minute**: ~30
- **Tokens per minute**: ~14,400
- **Models available**: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, etc.

The pipeline automatically:
- Uses `llama-3.3-70b-versatile` for Steps 4 & 6 (accuracy-critical)
- Falls back to `llama-3.1-8b-instant` on HTTP 429 errors
- Adds 2-5 second delays on rate limits

For production with large PDFs:
- Upgrade to paid tier (higher limits)
- OR: Process PDFs in batches during off-peak hours

**Save this credential**:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Vast.ai Instance Setup

### Step 1: Create Vast.ai Account

1. Go to https://vast.ai
2. Click **"Sign Up"**
3. Add $10-20 credit via **"Billing"** (Vast.ai doesn't offer free tier)

### Step 2: Search for GPU Instances

1. Go to **"Search"** tab
2. Filter by:
   - **GPU Model**: `RTX 3090` (24GB VRAM — recommended)
   - **Disk Space**: ≥ 50GB
   - **CUDA Version**: 12.1+ (for marker-pdf compatibility)
   - **Docker**: ✅ Enabled
   - **Internet Speed**: ≥ 100 Mbps (for model downloads)

3. Sort by **"$/hr"** (ascending) to find cheapest options
4. Typical cost: **$0.30-0.60/hour** for RTX 3090

### Step 3: Rent an Instance

1. Click **"RENT"** on a suitable instance
2. Configuration:
   - **Image/Template**: `pytorch/pytorch:latest` or `nvidia/cuda:12.1.0-base-ubuntu22.04`
   - **On-start script**: Leave empty (we'll install manually)
   - **Disk Space**: 50GB minimum
   - **Open ports**: `8000` (for FastAPI)

3. Click **"RENT"** and wait 1-2 minutes for startup

### Step 4: Connect via SSH

After startup, Vast.ai shows:

```
SSH Connection:
ssh root@<instance-ip> -p <port> -i <ssh-key-path>
```

**Get your instance IP**:
- It's shown in the instance details (e.g., `123.45.67.89`)
- Save this IP — you'll need it for Neo4j allowlist and frontend config

Connect:
```bash
ssh root@<instance-ip> -p <port>
```

(If using password auth, enter the password shown in Vast.ai dashboard)

### Step 5: Verify GPU

```bash
# Check GPU is accessible
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.1   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 3090  Off  | 00000000:01:00.0 Off |                  N/A |
...
```

If `nvidia-smi` fails:
```bash
# Reinstall NVIDIA drivers
apt-get update
apt-get install -y nvidia-driver-525
reboot
```

---

## Backend Deployment

### Step 1: Install Dependencies

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install essential tools
apt-get install -y \
    git \
    curl \
    wget \
    nano \
    htop

# Install Docker Compose (if not present)
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

### Step 2: Install NVIDIA Container Toolkit

This allows Docker to access the GPU.

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

# Install toolkit
apt-get update
apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
nvidia-ctk runtime configure --runtime=docker

# Restart Docker
systemctl restart docker

# Verify GPU is accessible in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If the verification command shows the GPU, you're ready.

### Step 3: Clone Repository

```bash
# Clone from GitHub (replace with your repo URL)
cd /root
git clone https://github.com/your-username/Edu-KG.git
cd Edu-KG/pace-kg
```

**If your repo is private**:
```bash
# Generate SSH key on Vast.ai instance
ssh-keygen -t ed25519 -C "vastai-deploy"
cat ~/.ssh/id_ed25519.pub
# Copy the output and add to GitHub → Settings → SSH Keys

# Then clone with SSH
git clone git@github.com:your-username/Edu-KG.git
```

### Step 4: Configure Environment Variables

```bash
cd /root/Edu-KG/pace-kg/backend

# Copy example file
cp .env.example .env

# Edit with nano (or vim)
nano .env
```

Fill in **ALL FOUR** credentials:

```env
# Groq API (from Step: Groq API Setup)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Neo4j AuraDB (from Step: Neo4j AuraDB Setup)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-generated-password
```

**Important**: No quotes around values, no spaces around `=`.

Save and exit: `Ctrl+X`, `Y`, `Enter`

**Verify file contents**:
```bash
cat .env | head -10
```

Should show your credentials (first few characters visible).

### Step 5: Update Neo4j IP Allowlist

Now that you have your Vast.ai instance IP:

1. Go back to Neo4j AuraDB dashboard
2. Select your instance → **"Network Access"**
3. **Remove** `0.0.0.0/0` if you added it earlier
4. **Add** your Vast.ai instance IP: `<instance-ip>/32` (e.g., `123.45.67.89/32`)
5. Click **"Add IP"**

This restricts Neo4j access to only your backend server.

### Step 6: Build Docker Image

```bash
# Return to pace-kg directory
cd /root/Edu-KG/pace-kg

# Build backend image (this takes 10-15 minutes first time)
docker-compose build backend
```

The build process:
1. Downloads NVIDIA CUDA base image (~2GB)
2. Installs Python 3.11 and system dependencies
3. Installs Python packages from `requirements.txt` (~4GB total)
4. Downloads spaCy model (`en_core_web_sm`)

**Watch for errors** — if build fails:
- Check `backend/requirements.txt` exists
- Check `backend/Dockerfile` exists
- Verify internet connection: `ping google.com`

### Step 7: Start Backend Container

```bash
# Start in detached mode
docker-compose up -d backend

# Check if container is running
docker-compose ps
```

Expected output:
```
NAME                COMMAND                  SERVICE   STATUS    PORTS
pace-kg-backend-1   "uvicorn main:app --…"   backend   Up        0.0.0.0:8000->8000/tcp
```

**Check logs** for startup messages:
```bash
docker-compose logs -f backend
```

Expected log output:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Press `Ctrl+C` to exit logs.

---

## Testing the Pipeline

### Step 1: Health Check

```bash
# From Vast.ai instance
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-03-26T10:30:45.123456"
}
```

### Step 2: Upload a Test PDF

**From your local machine** (not Vast.ai instance):

```bash
# Replace with your Vast.ai instance IP and PDF path
curl -X POST http://<instance-ip>:8000/upload \
  -F "file=@/path/to/lecture.pdf" \
  -H "Content-Type: multipart/form-data"
```

Example:
```bash
curl -X POST http://123.45.67.89:8000/upload \
  -F "file=@./CS1050-L03.pdf" \
  -H "Content-Type: multipart/form-data"
```

Expected response:
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Pipeline started for CS1050-L03.pdf"
}
```

**Save the `doc_id`** — you'll use it for all subsequent requests.

### Step 3: Monitor Pipeline Progress

```bash
# Check status (replace <doc_id> with your UUID)
curl http://<instance-ip>:8000/status/<doc_id>
```

Response while processing:
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "current_step": "Step 4: Extracting triples",
  "progress": 44
}
```

Response when complete:
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "current_step": "Step 9: Generating summaries",
  "progress": 100
}
```

**Monitor backend logs** in real-time:
```bash
# On Vast.ai instance
docker-compose logs -f backend
```

You'll see detailed output for each step:
```
[Step 1] Parsing PDF with Marker...
[Step 2] Preprocessing markdown...
[Step 3] Extracting keyphrases...
[Step 4] Extracting triples...
[Step 5] Weighting and pruning concepts...
[Step 6] Expanding concepts...
[Step 7] Storing in Neo4j...
[Step 8] Aggregating LM-EduKG...
[Step 9] Generating summaries...
Pipeline completed successfully!
```

### Step 4: Retrieve Results

**Get the knowledge graph**:
```bash
curl http://<instance-ip>:8000/graph/<doc_id> | jq '.' > graph.json
```

Response structure:
```json
{
  "nodes": [
    {
      "name": "queue",
      "aliases": [],
      "slide_ids": ["slide_003", "slide_004"],
      "source_type": "heading",
      "keyphrase_score": 0.85,
      "final_weight": 0.7234,
      "needs_review": false
    },
    ...
  ],
  "edges": [
    {
      "subject": "queue",
      "relation": "isDefinedAs",
      "object": "fifo data structure",
      "evidence": "A queue is a FIFO data structure...",
      "confidence": 0.92,
      "source": "extraction",
      "slide_id": "slide_003",
      "material_level": null
    },
    ...
  ]
}
```

**Get slide summaries**:
```bash
curl http://<instance-ip>:8000/summaries/<doc_id> | jq '.' > summaries.json
```

Response structure:
```json
[
  {
    "slide_id": "slide_003",
    "page_number": 3,
    "heading": "Queue Data Structure",
    "summary": "A queue is a FIFO data structure where elements are added at the rear and removed from the front. Common operations include enqueue and dequeue.",
    "key_terms": ["queue", "fifo", "enqueue", "dequeue"],
    "doc_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  ...
]
```

**Download Word document**:
```bash
curl -o summaries.docx http://<instance-ip>:8000/export/<doc_id>
```

The `.docx` file contains all slide summaries formatted for students.

### Step 5: Verify in Neo4j Browser

1. Go to Neo4j AuraDB dashboard → Open your instance in **Neo4j Browser**
2. Run visualization query:
   ```cypher
   MATCH (c:Concept {doc_id: "550e8400-e29b-41d4-a716-446655440000"})
   MATCH (c)-[r:RELATION]->(o:Concept)
   RETURN c, r, o
   LIMIT 50
   ```

3. Check node count:
   ```cypher
   MATCH (c:Concept {doc_id: "550e8400-e29b-41d4-a716-446655440000"})
   RETURN count(c) AS total_concepts
   ```

4. Check edge distribution:
   ```cypher
   MATCH ()-[r:RELATION {doc_id: "550e8400-e29b-41d4-a716-446655440000"}]->()
   RETURN r.relation_type AS relation, count(r) AS count
   ORDER BY count DESC
   ```

---

## Frontend Connection

### Step 1: Setup Local Frontend

On your **local development machine** (not Vast.ai):

```bash
# Clone repository (if not already cloned)
git clone https://github.com/your-username/Edu-KG.git
cd Edu-KG/pace-kg/frontend

# Install dependencies
npm install
```

### Step 2: Configure API URL

Create `.env` file in `frontend/` directory:

```bash
cd pace-kg/frontend
nano .env
```

Add your Vast.ai instance IP:

```env
VITE_API_URL=http://<instance-ip>:8000
```

Example:
```env
VITE_API_URL=http://123.45.67.89:8000
```

Save and exit.

### Step 3: Run Frontend Locally

```bash
npm run dev
```

Expected output:
```
  VITE v4.x.x  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.1.100:5173/
```

### Step 4: Test Upload via Frontend

1. Open browser: `http://localhost:5173`
2. Drag and drop a PDF file
3. Watch progress bar (polls `/status/<doc_id>` every 2 seconds)
4. When complete:
   - **Graph View**: Interactive D3.js visualization of concepts and relations
   - **Summary Panel**: Slide-by-slide summaries with key terms
   - **Export Button**: Download Word document

---

## Monitoring & Troubleshooting

### Monitor GPU Usage

```bash
# Watch GPU memory and utilization
watch -n 1 nvidia-smi
```

Expected during pipeline execution:
- **GPU Memory Used**: 8-12 GB (Marker + GLiNER + SBERT loaded)
- **GPU Utilization**: 40-80% (spiky during inference)

### Monitor Docker Logs

```bash
# Real-time logs
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Search for errors
docker-compose logs backend | grep -i error
```

### Monitor Disk Space

```bash
df -h
```

Pipeline writes to:
- `/root/Edu-KG/pace-kg/backend/uploads/` — Uploaded PDFs (~10-50 MB each)
- `/root/Edu-KG/pace-kg/backend/outputs/<doc_id>/` — JSON outputs (~1-5 MB per PDF)

Clean up old files:
```bash
cd /root/Edu-KG/pace-kg/backend

# Remove uploads older than 7 days
find uploads/ -type f -mtime +7 -delete

# Remove outputs older than 7 days
find outputs/ -type d -mtime +7 -exec rm -rf {} \;
```

### Common Issues

#### Issue: `Connection refused` to Neo4j

**Symptoms**:
```
neo4j.exceptions.ServiceUnavailable: Unable to retrieve routing information
```

**Fix**:
1. Check Neo4j instance is running (green status in dashboard)
2. Verify IP allowlist includes Vast.ai instance IP
3. Test connection manually:
   ```bash
   apt-get install -y cypher-shell
   cypher-shell -a <NEO4J_URI> -u neo4j -p <password>
   ```

#### Issue: `HTTP 429` rate limit from Groq

**Symptoms**:
```
Rate-limited (attempt 1) — trying fallback key ...
Still rate-limited — waiting 5s ...
```

**Fix**:
- This is expected on free tier
- The pipeline automatically retries with `llama-3.1-8b-instant`
- For frequent use: Upgrade to Groq paid tier ($0.27/1M tokens for 70B)

#### Issue: Out of GPU memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Fix**:
```bash
# Check what's using GPU
nvidia-smi

# Restart Docker to clear memory
docker-compose restart backend

# If persistent: Use a GPU with more VRAM (RTX 3090 = 24GB recommended)
```

#### Issue: Marker PDF parsing fails

**Symptoms**:
```
[Step 1] Parsing PDF failed: marker-pdf error
```

**Fix**:
- Check PDF is not corrupted: `pdfinfo /path/to/file.pdf`
- Some PDFs with complex layouts fail — fallback to OCR runs automatically
- Very large PDFs (>200 pages) may timeout — split into smaller files

#### Issue: Docker build fails

**Symptoms**:
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Fix**:
```bash
# Check internet connection
ping google.com

# Try rebuilding without cache
docker-compose build --no-cache backend

# Check disk space (needs ~15GB free)
df -h
```

### Performance Optimization

**For faster processing**:

1. **Use persistent Docker volumes** (avoid model re-downloads):
   ```yaml
   # In docker-compose.yml
   volumes:
     - model-cache:/root/.cache
   ```

2. **Pre-download models** in Dockerfile:
   ```dockerfile
   RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
   RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_large-v2.1')"
   ```

3. **Batch process multiple PDFs** (share loaded models):
   - Keep container running
   - Upload multiple PDFs sequentially
   - Models stay loaded in memory

---

## Security Best Practices

### 1. Restrict Neo4j Access

**Current setup**: IP allowlist with Vast.ai IP only ✅

**Production hardening**:
- Use Neo4j AuraDB **private endpoint** (requires VPN/VPC)
- Rotate passwords monthly
- Enable Neo4j audit logging

### 2. Secure API Access

**Current setup**: No authentication ⚠️

**Add API key auth**:

```python
# In main.py
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/upload")
async def upload(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of endpoint
```

Then set in `.env`:
```env
API_KEY=your-secret-key-here
```

### 3. Use HTTPS

**For production**, add nginx reverse proxy with SSL:

```bash
apt-get install -y nginx certbot python3-certbot-nginx

# Get SSL certificate (requires domain name)
certbot --nginx -d api.your-domain.com

# Configure nginx proxy
nano /etc/nginx/sites-available/pace-kg
```

nginx config:
```nginx
server {
    listen 443 ssl;
    server_name api.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/api.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Secrets Management

**Never commit `.env` to git**:

```bash
# Add to .gitignore
echo ".env" >> .gitignore
```

**For production**: Use secrets management:
- AWS Secrets Manager
- HashiCorp Vault
- Docker Swarm secrets
- Kubernetes secrets

### 5. Network Security

**Current setup**: Port 8000 exposed publicly ⚠️

**Harden**:
1. Use Vast.ai **firewall rules** to restrict IP access
2. OR: Use SSH tunnel for testing:
   ```bash
   # On local machine
   ssh -L 8000:localhost:8000 root@<instance-ip> -p <port>
   # Then access http://localhost:8000 locally
   ```

---

## Cost Management

### Vast.ai Pricing

- **RTX 3090**: $0.30-0.60/hour
- **Average PDF processing time**: 5-15 minutes
- **Cost per PDF**: $0.05-0.15

### Save Money

1. **Release instance when not in use**:
   ```bash
   # Stop container (saves state)
   docker-compose stop
   ```
   Then release instance via Vast.ai dashboard.

2. **Use spot instances** (interruptible but cheaper)

3. **Process in batches** during development

4. **Use smaller GPU for testing** (RTX 3060 Ti = $0.15-0.30/hour)
   - But note: Slower processing, may require CPU fallback for SBERT

### Free Tier Limits

- **Neo4j AuraDB Free**: 50k nodes + 175k relationships (sufficient for 100+ PDFs)
- **Groq Free**: 30 requests/min (sufficient for 1-2 PDFs at a time)

Upgrade when:
- Neo4j: Database grows beyond limits → $0.08/hour for Pro tier
- Groq: Rate limits become blocking → $0.27/1M tokens for pay-as-you-go

---

## Stopping & Cleanup

### Stop Backend (Keep Data)

```bash
docker-compose stop backend
```

Uploads and outputs remain in `uploads/` and `outputs/`.

### Stop Backend (Delete Data)

```bash
docker-compose down -v
rm -rf uploads outputs
```

**Warning**: This deletes all PDFs and pipeline outputs. Neo4j data persists.

### Clear Neo4j Database

In Neo4j Browser:
```cypher
// Delete all nodes and relationships for a doc_id
MATCH (n {doc_id: "550e8400-e29b-41d4-a716-446655440000"})
DETACH DELETE n

// OR delete entire database (careful!)
MATCH (n) DETACH DELETE n
```

### Release Vast.ai Instance

1. Go to Vast.ai dashboard → **Instances**
2. Click instance → **Destroy**
3. Confirm (billing stops immediately)

---

## Updating Code

When you push changes to GitHub:

```bash
# SSH into Vast.ai instance
ssh root@<instance-ip> -p <port>

# Pull latest changes
cd /root/Edu-KG
git pull origin main

# Rebuild backend
cd pace-kg
docker-compose build backend

# Restart with new code
docker-compose up -d backend

# Verify
docker-compose logs -f backend
```

---

## Support & Resources

- **PACE-KG Documentation**: See `CLAUDE.md` in repo root
- **Neo4j Cypher Docs**: https://neo4j.com/docs/cypher-manual/
- **Groq API Docs**: https://console.groq.com/docs
- **Vast.ai Docs**: https://vast.ai/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com

---

## Quick Reference Card

```bash
# ═══ Neo4j AuraDB ═══
URL: neo4j+s://xxxxx.databases.neo4j.io
User: neo4j
Pass: <your-password>

# ═══ Groq API ═══
Key: gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Models: llama-3.3-70b-versatile (70B), llama-3.1-8b-instant (8B)

# ═══ Vast.ai Instance ═══
IP: <instance-ip>
Port: <ssh-port>
SSH: ssh root@<instance-ip> -p <port>

# ═══ Backend API ═══
Health: http://<instance-ip>:8000/health
Upload: POST http://<instance-ip>:8000/upload
Status: GET http://<instance-ip>:8000/status/<doc_id>
Graph:  GET http://<instance-ip>:8000/graph/<doc_id>
Summary: GET http://<instance-ip>:8000/summaries/<doc_id>
Export: GET http://<instance-ip>:8000/export/<doc_id>

# ═══ Docker Commands ═══
Build:   docker-compose build backend
Start:   docker-compose up -d backend
Stop:    docker-compose stop backend
Logs:    docker-compose logs -f backend
Restart: docker-compose restart backend
Clean:   docker-compose down -v

# ═══ Monitoring ═══
GPU:  watch -n 1 nvidia-smi
Logs: docker-compose logs -f backend
Disk: df -h
```

Save this card for quick reference during deployment.
