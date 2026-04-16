import type {
  AsyncJob,
  ArtifactRecord,
  BehaviorAssessment,
  CitationAuditRecord,
  ClarificationSession,
  ContextPack,
  CreateRunPayload,
  FinalReport,
  PassageInspectionRecord,
  PlanPreview,
  ProfileFeedback,
  ProfilePreferences,
  ProfileRecord,
  ProjectDetail,
  ProjectSummary,
  PublicRuntimeConfig,
  ResearchAssetRecord,
  ResearchInputAsset,
  RunConversationMessage,
  RunConversationReply,
  RunNoteRecord,
  RunDetail,
  RunSummary,
  RunWorkspaceSnapshot,
  StagedAssetRecord,
} from "./types";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

async function requestJson<T>(
  baseUrl: string,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "content-type": "application/json" }),
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      detail = typeof payload.detail === "string" ? payload.detail : detail;
    } catch {
      // Ignore JSON parse failures and fall back to the HTTP status text.
    }
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

async function requestEmpty(
  baseUrl: string,
  path: string,
  init?: RequestInit,
): Promise<void> {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      detail = typeof payload.detail === "string" ? payload.detail : detail;
    } catch {
      // Ignore parse errors.
    }
    throw new Error(detail || `Request failed with ${response.status}`);
  }
}

export function getStreamUrl(baseUrl: string, runId: string, lastEventId?: number): string {
  const root = normalizeBaseUrl(baseUrl);
  if (lastEventId && lastEventId > 0) {
    return `${root}/runs/${runId}/stream/${lastEventId}`;
  }
  return `${root}/runs/${runId}/stream`;
}

export function fetchPublicConfig(baseUrl: string): Promise<PublicRuntimeConfig> {
  return requestJson<PublicRuntimeConfig>(baseUrl, "/config/public", {
    headers: {},
  });
}

export function fetchRuns(baseUrl: string): Promise<RunSummary[]> {
  return requestJson<RunSummary[]>(baseUrl, "/runs", { headers: {} });
}

export function fetchProjects(baseUrl: string): Promise<ProjectSummary[]> {
  return requestJson<ProjectSummary[]>(baseUrl, "/projects", { headers: {} });
}

export function createProject(
  baseUrl: string,
  payload: { name: string; description?: string | null },
): Promise<ProjectSummary> {
  return requestJson<ProjectSummary>(baseUrl, "/projects", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchProjectDetail(baseUrl: string, projectId: string): Promise<ProjectDetail> {
  return requestJson<ProjectDetail>(baseUrl, `/projects/${projectId}`, { headers: {} });
}

export function addProjectAssets(
  baseUrl: string,
  projectId: string,
  assets: ResearchInputAsset[],
): Promise<ResearchAssetRecord[]> {
  return requestJson<ResearchAssetRecord[]>(baseUrl, `/projects/${projectId}/assets`, {
    method: "POST",
    body: JSON.stringify({ assets }),
  });
}

export function uploadProjectFiles(
  baseUrl: string,
  projectId: string,
  usage: "reference_source" | "planning_context",
  files: File[],
  description?: string | null,
): Promise<ResearchAssetRecord[]> {
  const formData = new FormData();
  formData.append("usage", usage);
  if (description) formData.append("description", description);
  files.forEach((file) => formData.append("files", file));
  return requestJson<ResearchAssetRecord[]>(baseUrl, `/projects/${projectId}/assets/upload`, {
    method: "POST",
    body: formData,
  });
}

export function deleteProjectAsset(baseUrl: string, projectId: string, assetId: string): Promise<void> {
  return requestEmpty(baseUrl, `/projects/${projectId}/assets/${assetId}`, {
    method: "DELETE",
  });
}

export function stageAssets(
  baseUrl: string,
  usage: "reference_source" | "planning_context",
  files: File[],
  description?: string | null,
): Promise<StagedAssetRecord[]> {
  const formData = new FormData();
  formData.append("usage", usage);
  if (description) formData.append("description", description);
  files.forEach((file) => formData.append("files", file));
  return requestJson<StagedAssetRecord[]>(baseUrl, "/assets/staged", {
    method: "POST",
    body: formData,
  });
}

export function fetchStagedAsset(baseUrl: string, assetId: string): Promise<StagedAssetRecord> {
  return requestJson<StagedAssetRecord>(baseUrl, `/assets/staged/${assetId}`, { headers: {} });
}

export function deleteStagedAsset(baseUrl: string, assetId: string): Promise<void> {
  return requestEmpty(baseUrl, `/assets/staged/${assetId}`, { method: "DELETE" });
}

export function fetchRunDetail(baseUrl: string, runId: string): Promise<RunDetail> {
  return requestJson<RunDetail>(baseUrl, `/runs/${runId}`, { headers: {} });
}

export function fetchRunMessages(
  baseUrl: string,
  runId: string,
): Promise<RunConversationMessage[]> {
  return requestJson<RunConversationMessage[]>(baseUrl, `/runs/${runId}/messages`, {
    headers: {},
  });
}

export function sendRunMessage(
  baseUrl: string,
  runId: string,
  message: string,
): Promise<RunConversationReply> {
  return requestJson<RunConversationReply>(baseUrl, `/runs/${runId}/messages`, {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

export function fetchRunWorkspace(baseUrl: string, runId: string): Promise<RunWorkspaceSnapshot> {
  return requestJson<RunWorkspaceSnapshot>(baseUrl, `/runs/${runId}/workspace`, { headers: {} });
}

export function fetchRunReport(baseUrl: string, runId: string): Promise<FinalReport> {
  return requestJson<FinalReport>(baseUrl, `/runs/${runId}/report`, { headers: {} });
}

export function fetchRunArtifacts(baseUrl: string, runId: string): Promise<ArtifactRecord[]> {
  return requestJson<ArtifactRecord[]>(baseUrl, `/runs/${runId}/artifacts`, { headers: {} });
}

export function fetchRunAudit(baseUrl: string, runId: string): Promise<CitationAuditRecord[]> {
  return requestJson<CitationAuditRecord[]>(baseUrl, `/runs/${runId}/audit`, { headers: {} });
}

export function fetchRunNotes(baseUrl: string, runId: string): Promise<RunNoteRecord[]> {
  return requestJson<RunNoteRecord[]>(baseUrl, `/runs/${runId}/notes`, { headers: {} });
}

export function fetchRunPassages(
  baseUrl: string,
  runId: string,
): Promise<PassageInspectionRecord[]> {
  return requestJson<PassageInspectionRecord[]>(baseUrl, `/runs/${runId}/passages`, {
    headers: {},
  });
}

export function fetchRunContextPacks(baseUrl: string, runId: string): Promise<ContextPack[]> {
  return requestJson<ContextPack[]>(baseUrl, `/runs/${runId}/context-packs`, { headers: {} });
}

export function fetchRunAssessments(
  baseUrl: string,
  runId: string,
): Promise<BehaviorAssessment[]> {
  return requestJson<BehaviorAssessment[]>(baseUrl, `/runs/${runId}/assessments`, {
    headers: {},
  });
}

export function fetchProfilePreferences(baseUrl: string, profileId: string): Promise<ProfileRecord> {
  return requestJson<ProfileRecord>(baseUrl, `/profiles/${profileId}/preferences`, { headers: {} });
}

export function updateProfilePreferences(
  baseUrl: string,
  profileId: string,
  preferences: ProfilePreferences,
): Promise<ProfileRecord> {
  return requestJson<ProfileRecord>(baseUrl, `/profiles/${profileId}/preferences`, {
    method: "PUT",
    body: JSON.stringify(preferences),
  });
}

export function postProfileFeedback(
  baseUrl: string,
  profileId: string,
  feedback: ProfileFeedback,
): Promise<BehaviorAssessment[]> {
  return requestJson<BehaviorAssessment[]>(baseUrl, `/profiles/${profileId}/feedback`, {
    method: "POST",
    body: JSON.stringify(feedback),
  });
}

export function createRun(baseUrl: string, payload: CreateRunPayload): Promise<RunSummary> {
  return requestJson<RunSummary>(baseUrl, "/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createJob(baseUrl: string, payload: CreateRunPayload): Promise<AsyncJob> {
  return requestJson<AsyncJob>(baseUrl, "/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchJob(baseUrl: string, jobId: string): Promise<AsyncJob> {
  return requestJson<AsyncJob>(baseUrl, `/jobs/${jobId}`, { headers: {} });
}

export function fetchJobWorkspace(baseUrl: string, jobId: string): Promise<RunWorkspaceSnapshot> {
  return requestJson<RunWorkspaceSnapshot>(baseUrl, `/jobs/${jobId}/workspace`, {
    headers: {},
  });
}

export function fetchClarification(
  baseUrl: string,
  runId: string,
): Promise<ClarificationSession> {
  return requestJson<ClarificationSession>(baseUrl, `/runs/${runId}/clarification`, {
    headers: {},
  });
}

export function answerClarification(
  baseUrl: string,
  runId: string,
  response: string,
): Promise<RunDetail> {
  return requestJson<RunDetail>(baseUrl, `/runs/${runId}/clarification/respond`, {
    method: "POST",
    body: JSON.stringify({ response }),
  });
}

export function fetchPlanPreview(baseUrl: string, runId: string): Promise<PlanPreview> {
  return requestJson<PlanPreview>(baseUrl, `/runs/${runId}/plan-preview`, {
    headers: {},
  });
}

export function approvePlan(
  baseUrl: string,
  runId: string,
  note?: string,
): Promise<RunDetail> {
  return requestJson<RunDetail>(baseUrl, `/runs/${runId}/plan-preview/approve`, {
    method: "POST",
    body: JSON.stringify({ note: note ?? null }),
  });
}

export function rejectPlan(
  baseUrl: string,
  runId: string,
  note?: string,
): Promise<RunDetail> {
  return requestJson<RunDetail>(baseUrl, `/runs/${runId}/plan-preview/reject`, {
    method: "POST",
    body: JSON.stringify({ note: note ?? null }),
  });
}

export function requestPlanChanges(
  baseUrl: string,
  runId: string,
  note?: string,
): Promise<RunDetail> {
  return requestJson<RunDetail>(baseUrl, `/runs/${runId}/plan-preview/request-changes`, {
    method: "POST",
    body: JSON.stringify({ note: note ?? null }),
  });
}

export function promoteRunAsset(
  baseUrl: string,
  runId: string,
  assetId: string,
  projectId?: string | null,
): Promise<ResearchAssetRecord> {
  return requestJson<ResearchAssetRecord>(baseUrl, `/runs/${runId}/assets/${assetId}/promote`, {
    method: "POST",
    body: JSON.stringify({ project_id: projectId ?? null }),
  });
}

export function cancelRun(baseUrl: string, runId: string): Promise<RunDetail> {
  return requestJson<RunDetail>(baseUrl, `/runs/${runId}/cancel`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function resumeRun(baseUrl: string, runId: string): Promise<RunSummary> {
  return requestJson<RunSummary>(baseUrl, `/runs/${runId}/resume`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function retryRun(baseUrl: string, runId: string): Promise<RunSummary> {
  return requestJson<RunSummary>(baseUrl, `/runs/${runId}/retry`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}
