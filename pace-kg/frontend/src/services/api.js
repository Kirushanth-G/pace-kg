const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function readJson(response, fallbackMessage) {
  if (!response.ok) {
    let detail = fallbackMessage;
    try {
      const payload = await response.json();
      detail = payload?.detail || payload?.message || detail;
    } catch {
      // Ignore JSON parse failures and use fallback message.
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  return readJson(response, "Upload failed.");
}

export async function getStatus(docId) {
  const response = await fetch(`${API_BASE}/status/${docId}`);
  return readJson(response, "Could not get job status.");
}

export async function getGraph(docId) {
  const response = await fetch(`${API_BASE}/graph/${docId}`);
  return readJson(response, "Could not fetch graph data.");
}

export async function getSummaries(docId) {
  const response = await fetch(`${API_BASE}/summaries/${docId}`);
  return readJson(response, "Could not fetch summaries.");
}

export function getExportUrl(docId) {
  return `${API_BASE}/export/${docId}`;
}

export async function listJobs() {
  const response = await fetch(`${API_BASE}/jobs`);
  return readJson(response, "Could not fetch jobs list.");
}

export { API_BASE };
