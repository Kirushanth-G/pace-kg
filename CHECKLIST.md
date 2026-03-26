# PACE-KG Backend Implementation Checklist

## ✅ Completed

- [x] FastAPI backend with all 5 endpoints
- [x] Background task execution with progress tracking
- [x] Pipeline runner with Steps 1-4 and 9
- [x] Data models and schemas
- [x] Environment-based configuration
- [x] Docker setup with NVIDIA CUDA base
- [x] Complete documentation (README, QUICKSTART, DEPLOYMENT, TODO)
- [x] CLAUDE.md updated with correct architecture

## 🔧 To Complete (4-6 hours)

- [ ] **Step 5**: Concept weighting and pruning
  - Extract from `edu-kg glinger.py` lines ~2010-2200
  - 3-signal SBERT weighting
  - Relation role and source type boosts
  - Semantic merge and pruning
  
- [ ] **Step 6**: Closed-corpus expansion
  - Extract from `edu-kg glinger.py` lines ~2279-2500
  - Build document vocabulary
  - 4-phase expansion (LLM + SBERT + scope + dedup)
  - Cap at 10 expansion edges per concept
  
- [ ] **Step 7**: Neo4j storage
  - Extract from `edu-kg glinger.py` lines ~2500-2700
  - Two-pass write (nodes first, then edges)
  - MERGE-based writes (never CREATE)
  - Semantic conflict resolution
  
- [ ] **Step 8**: LM-EduKG aggregation
  - Extract from `edu-kg glinger.py` lines ~2700-2850
  - Query Neo4j for full graph export
  - Verify 4 merge constraints
  - Build final lm_edkg dict

## 🧪 Testing Checklist

### Local Testing
- [ ] Set up `.env` with credentials
- [ ] Build Docker image
- [ ] Start backend: `docker-compose up backend`
- [ ] Test health: `curl http://localhost:8000/health`
- [ ] Upload PDF: `curl -X POST http://localhost:8000/upload -F "file=@test3.pdf"`
- [ ] Check status: `curl http://localhost:8000/status/<doc_id>`
- [ ] Wait for completion (watch logs)
- [ ] Verify all 9 output files exist in `outputs/<doc_id>/`
- [ ] Test graph endpoint: `curl http://localhost:8000/graph/<doc_id>`
- [ ] Test summaries endpoint: `curl http://localhost:8000/summaries/<doc_id>`
- [ ] Test export endpoint: `curl -O http://localhost:8000/export/<doc_id>`

### Neo4j Verification
- [ ] Connect to Neo4j browser
- [ ] Run: `MATCH (c:Concept) RETURN count(c)`
- [ ] Run: `MATCH ()-[r:RELATION]->() RETURN count(r)`
- [ ] Verify concepts and edges exist

### Vast.ai Deployment
- [ ] Provision RTX 3090 instance
- [ ] SSH and install Docker + NVIDIA toolkit
- [ ] Clone repository
- [ ] Set up `.env` with production credentials
- [ ] Build and run: `docker-compose up -d backend`
- [ ] Test all endpoints from local machine
- [ ] Monitor GPU usage: `nvidia-smi -l 1`
- [ ] Check logs: `docker-compose logs -f backend`

### Frontend Integration
- [ ] Update frontend `VITE_API_URL` to Vast.ai IP
- [ ] Start frontend: `npm run dev`
- [ ] Test upload flow
- [ ] Verify graph rendering
- [ ] Verify summary panel
- [ ] Test Word document download

## 📋 Pre-Deployment Checklist

- [ ] All 9 steps working locally
- [ ] Neo4j credentials secured
- [ ] Groq API key secured (not in git)
- [ ] `.env` added to `.gitignore`
- [ ] Vast.ai instance configured
- [ ] Docker image builds successfully
- [ ] GPU passthrough working
- [ ] All endpoints tested
- [ ] Frontend connects to backend
- [ ] Documentation reviewed

## 🚀 Launch Checklist

- [ ] Deploy backend to Vast.ai
- [ ] Test with multiple PDFs
- [ ] Monitor resource usage
- [ ] Verify rate limits (Groq)
- [ ] Set up monitoring/logging
- [ ] Document API endpoints
- [ ] Share access with team/users

## 📊 Success Metrics

- [ ] Pipeline processes PDF in < 15 minutes
- [ ] Graph contains > 10 concepts per 10 slides
- [ ] Summaries are coherent and accurate
- [ ] Word document exports correctly
- [ ] No GPU memory errors
- [ ] No Groq rate limit issues
- [ ] Frontend renders graph smoothly

## 🐛 Known Issues to Watch

- [ ] Groq rate limits (30 req/min free tier) - implement delays if needed
- [ ] GPU memory - RTX 3090 should handle, but monitor
- [ ] Long PDFs (>50 slides) - may hit time limits
- [ ] Neo4j connection timeouts - check network

## 📚 Reference Documents

- `CLAUDE.md` - Complete specification
- `edu-kg glinger.py` - Reference implementation
- `pace-kg/backend/README.md` - Backend overview
- `pace-kg/backend/QUICKSTART.md` - Quick start
- `pace-kg/backend/TODO_STEPS_5-8.md` - Implementation guide
- `pace-kg/DEPLOYMENT.md` - Deployment guide
- `BACKEND_COMPLETE_SUMMARY.md` - This summary

---

**Last Updated**: After backend implementation
**Status**: 90% complete, Steps 5-8 need implementation
**Next Action**: Follow `TODO_STEPS_5-8.md` to complete remaining steps
