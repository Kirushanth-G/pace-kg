import { useEffect, useMemo, useState } from "react";
import GraphView from "./components/GraphView";
import SummaryPanel from "./components/SummaryPanel";
import {
  getExportUrl,
  getGraph,
  getStatus,
  getSummaries,
  uploadPdf,
  listJobs,
  API_BASE,
} from "./services/api";
import { demoGraph, demoSummaries } from "./mocks/demoData";

const PIPELINE_STEPS = [
  "Parsing",
  "Preprocessing",
  "Keyphrases",
  "Triples",
  "Weighting",
  "Expansion",
  "Storing",
  "Aggregating",
  "Generating Summaries",
];

const FRONTEND_ONLY = import.meta.env.VITE_FRONTEND_ONLY === "true";
const DEMO_DOC_ID = "demo-local";

function normalizeStep(step) {
  const value = (step || "").toLowerCase();
  if (value.includes("parse") || value.includes("marker")) return "Parsing";
  if (value.includes("preprocess")) return "Preprocessing";
  if (value.includes("keyphrase") || value.includes("step3")) return "Keyphrases";
  if (value.includes("triple") || value.includes("step4")) return "Triples";
  if (value.includes("weight") || value.includes("step5")) return "Weighting";
  if (value.includes("expansion") || value.includes("step6")) return "Expansion";
  if (value.includes("stor") || value.includes("neo4j") || value.includes("step7")) {
    return "Storing";
  }
  if (value.includes("aggregate") || value.includes("step8")) return "Aggregating";
  if (value.includes("summary") || value.includes("step9")) {
    return "Generating Summaries";
  }
  return "Parsing";
}

function UploadView({
  selectedFile,
  onFileSelect,
  onUpload,
  uploadDisabled,
  dragActive,
  onDragEnter,
  onDragLeave,
  onDrop,
  completedJobs,
  onSelectJob,
  loadingJobs,
}) {
  return (
    <section className="upload-shell">
      <div className="hero-copy">
        <p className="eyebrow">PACE-KG</p>
        <h1>Build a Citation-Evidenced Learning Graph from Lecture Slides</h1>
        <p>
          Upload one PDF. The pipeline extracts concepts, validates relations with evidence,
          and generates per-slide revision summaries.
        </p>
      </div>

      {/* Show completed jobs if any exist */}
      {completedJobs.length > 0 && (
        <div className="completed-jobs panel">
          <h3>Previously Completed Graphs</h3>
          <p className="muted">Select a completed job to view its knowledge graph:</p>
          <div className="job-list">
            {completedJobs.map((job) => (
              <button
                key={job.doc_id}
                type="button"
                className="secondary-btn job-btn"
                onClick={() => onSelectJob(job.doc_id)}
              >
                {job.doc_id.slice(0, 8)}...
              </button>
            ))}
          </div>
        </div>
      )}

      {loadingJobs && (
        <p className="muted">Loading available jobs...</p>
      )}

      <div
        className={`dropzone ${dragActive ? "drag-active" : ""}`}
        onDragEnter={onDragEnter}
        onDragOver={onDragEnter}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <p>Drag and drop your PDF here</p>
        <span>or</span>
        <label className="secondary-btn" htmlFor="pdf-file-input">
          Choose PDF
        </label>
        <input
          id="pdf-file-input"
          type="file"
          accept="application/pdf"
          onChange={(event) => onFileSelect(event.target.files?.[0] || null)}
          hidden
        />

        {selectedFile ? <p className="file-pill">{selectedFile.name}</p> : null}

        <button
          type="button"
          className="primary-btn"
          onClick={onUpload}
          disabled={uploadDisabled}
        >
          Start Pipeline
        </button>
      </div>
    </section>
  );
}

function ProgressView({ docId, currentStep, progress, status }) {
  const currentIndex = PIPELINE_STEPS.indexOf(currentStep);

  return (
    <section className="progress-shell panel">
      <div className="panel-header">
        <h2>Pipeline Running</h2>
        <span className={`status-badge status-${status}`}>{status}</span>
      </div>

      <p>
        <strong>Document ID:</strong> {docId}
      </p>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <p className="progress-text">{Math.round(progress * 100)}% complete</p>

      <ol className="step-list">
        {PIPELINE_STEPS.map((step, index) => {
          const cls =
            index < currentIndex
              ? "done"
              : index === currentIndex
                ? "current"
                : "pending";
          return (
            <li key={step} className={cls}>
              <span>{step}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [docId, setDocId] = useState("");
  const [status, setStatus] = useState("idle");
  const [currentStep, setCurrentStep] = useState("Parsing");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const [graph, setGraph] = useState({ nodes: [], edges: [] });
  const [summaries, setSummaries] = useState([]);
  const [selectedSlideId, setSelectedSlideId] = useState("");
  const [selectedNode, setSelectedNode] = useState("");

  // New: track completed jobs from backend
  const [completedJobs, setCompletedJobs] = useState([]);
  const [loadingJobs, setLoadingJobs] = useState(false);

  const isProcessing = status === "pending" || status === "running";

  const canUpload = useMemo(
    () => Boolean(selectedFile) && !isProcessing,
    [selectedFile, isProcessing]
  );

  // Fetch completed jobs on mount (if not in frontend-only mode)
  useEffect(() => {
    if (FRONTEND_ONLY) {
      setDocId(DEMO_DOC_ID);
      setStatus("completed");
      setCurrentStep("Generating Summaries");
      setProgress(1);
      setGraph(demoGraph);
      setSummaries(demoSummaries);
      setSelectedSlideId(demoSummaries[0]?.slide_id || "");
      return;
    }

    // Fetch available jobs from backend
    async function fetchJobs() {
      setLoadingJobs(true);
      try {
        const response = await listJobs();
        const completed = (response?.jobs || []).filter(
          (job) => job.status === "completed"
        );
        setCompletedJobs(completed);
      } catch (err) {
        console.error("Failed to fetch jobs:", err);
      } finally {
        setLoadingJobs(false);
      }
    }

    fetchJobs();
  }, []);

  // Handler to load a previously completed job
  async function handleSelectJob(selectedDocId) {
    try {
      setError("");
      setDocId(selectedDocId);
      setStatus("completed");
      setProgress(1);
      setCurrentStep("Done");
      await loadResults(selectedDocId);
    } catch (err) {
      setError(err.message || "Failed to load job.");
      setStatus("failed");
    }
  }

  async function loadResults(activeDocId) {
    const [graphPayload, summariesPayload] = await Promise.all([
      getGraph(activeDocId),
      getSummaries(activeDocId),
    ]);

    setGraph({
      nodes: graphPayload?.nodes || [],
      edges: graphPayload?.edges || [],
    });

    const summaryItems = Array.isArray(summariesPayload)
      ? summariesPayload
      : summariesPayload?.summaries || [];

    setSummaries(summaryItems);
    setSelectedSlideId(summaryItems[0]?.slide_id || "");
  }

  async function pollStatus(activeDocId) {
    const response = await getStatus(activeDocId);

    setStatus(response.status || "running");
    setCurrentStep(normalizeStep(response.current_step));
    setProgress(Number(response.progress || 0));

    if (response.status === "failed") {
      throw new Error(response.error || "Pipeline failed.");
    }

    if (response.status === "completed") {
      await loadResults(activeDocId);
      return true;
    }

    return false;
  }

  useEffect(() => {
    if (!docId || !isProcessing) {
      return undefined;
    }

    let cancelled = false;
    let intervalId;

    const tick = async () => {
      try {
        const done = await pollStatus(docId);
        if (done && !cancelled) {
          setStatus("completed");
          clearInterval(intervalId);
        }
      } catch (pollError) {
        if (!cancelled) {
          setStatus("failed");
          setError(pollError.message);
          clearInterval(intervalId);
        }
      }
    };

    tick();
    intervalId = setInterval(tick, 2200);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [docId, isProcessing]);

  async function handleUpload() {
    if (!selectedFile) {
      return;
    }

    try {
      setError("");
      setGraph({ nodes: [], edges: [] });
      setSummaries([]);
      setSelectedNode("");
      setSelectedSlideId("");
      setStatus("pending");
      setCurrentStep("Parsing");
      setProgress(0);

      const response = await uploadPdf(selectedFile);
      setDocId(response.doc_id);
      setStatus(response.status || "running");
    } catch (uploadError) {
      setStatus("failed");
      setError(uploadError.message || "Upload failed.");
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);

    const file = event.dataTransfer.files?.[0];
    if (!file) {
      return;
    }

    if (file.type !== "application/pdf") {
      setError("Please upload a PDF file.");
      return;
    }

    setSelectedFile(file);
    setError("");
  }

  function handleDragEnter(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
  }

  function handleSelectSlide(slideId) {
    setSelectedSlideId(slideId);
  }

  function handleDownload() {
    if (!docId) {
      return;
    }

    if (FRONTEND_ONLY) {
      setError("Demo mode: export requires backend.");
      return;
    }

    window.open(getExportUrl(docId), "_blank", "noopener,noreferrer");
  }

  return (
    <main className="app-root">
      <header className="app-header">
        <div>
          <h1>PACE-KG</h1>
          <p>{FRONTEND_ONLY ? "Demo mode: backend disabled" : `Backend: ${API_BASE}`}</p>
        </div>
      </header>

      {status === "idle" ? (
        <UploadView
          selectedFile={selectedFile}
          onFileSelect={(file) => {
            setSelectedFile(file);
            setError("");
          }}
          onUpload={handleUpload}
          uploadDisabled={!canUpload}
          dragActive={dragActive}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          completedJobs={completedJobs}
          onSelectJob={handleSelectJob}
          loadingJobs={loadingJobs}
        />
      ) : null}

      {isProcessing ? (
        <ProgressView
          docId={docId}
          currentStep={currentStep}
          progress={progress}
          status={status}
        />
      ) : null}

      {status === "completed" ? (
        <>
          <div className="results-header">
            <button
              type="button"
              className="secondary-btn back-btn"
              onClick={() => {
                setStatus("idle");
                setDocId("");
                setGraph({ nodes: [], edges: [] });
                setSummaries([]);
                setSelectedFile(null);
              }}
            >
              &larr; Back to Upload
            </button>
            <span className="muted">Viewing: {docId.slice(0, 8)}...</span>
          </div>
          <section className="results-grid">
            <GraphView
              nodes={graph.nodes}
              edges={graph.edges}
              selectedSlideId={selectedSlideId}
              selectedNode={selectedNode}
              onSelectNode={setSelectedNode}
            />
            <SummaryPanel
              summaries={summaries}
              selectedSlideId={selectedSlideId}
              onSelectSlide={handleSelectSlide}
              onDownload={handleDownload}
            />
          </section>
        </>
      ) : null}

      {status === "failed" ? (
        <section className="panel error-panel">
          <h2>Something went wrong</h2>
          <p>{error || "Unknown error."}</p>
        </section>
      ) : null}

      {error && status !== "failed" ? <p className="error-inline">{error}</p> : null}
    </main>
  );
}

export default App;
