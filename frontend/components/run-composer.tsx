"use client";

import type { ChangeEvent } from "react";
import { useMemo, useRef, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { CustomSelect, type CustomSelectOption } from "@/components/ui/custom-select";
import {
  addProjectAssets,
  createProject,
  deleteProjectAsset,
  deleteStagedAsset,
  fetchProjectDetail,
  fetchProjects,
  stageAssets,
  uploadProjectFiles,
} from "@/lib/api";
import { DEFAULT_AGENT_CONFIG } from "@/lib/defaults";
import type {
  ClarifierConfig,
  ExecutionMode,
  MemoryPolicyNumericField,
  ModelConfig,
  ModelConfigOverride,
  ResearchAssetRecord,
  ResearchAssetUsage,
  ResearchInputAsset,
  PublicRuntimeConfig,
  ProjectSummary,
  StagedAssetRecord,
} from "@/lib/types";
import { useResearchStore } from "@/store/use-research-store";

export type FocusedDrawerKey = "workflow" | "sources" | "budget" | "models" | "profile";

interface RunComposerProps {
  publicConfig: PublicRuntimeConfig | undefined;
  onOpenPanel?: (panel: FocusedDrawerKey) => void;
}

const budgetFields = [
  {
    key: "max_streams",
    label: "Max streams",
    description: "Named parallel investigations in the run.",
    min: 1,
    max: 30,
  },
  {
    key: "max_replans",
    label: "Max replans",
    description: "How many gap-closure loops the orchestrator can trigger.",
    min: 0,
    max: 5,
  },
  {
    key: "max_queries_per_stream",
    label: "Queries per stream",
    description: "Upper bound for search passes inside each stream.",
    min: 1,
    max: 25,
  },
  {
    key: "max_results_per_query",
    label: "Results per query",
    description: "Candidates retained from each search call before filtering.",
    min: 1,
    max: 20,
  },
  {
    key: "max_sources_per_stream",
    label: "Sources per stream",
    description: "Fetched sources allowed to survive domain de-duplication.",
    min: 1,
    max: 20,
  },
  {
    key: "per_domain_limit",
    label: "Per-domain cap",
    description: "How aggressively to avoid monoculture sourcing.",
    min: 1,
    max: 10,
  },
] as const;

const memoryNumericFields: ReadonlyArray<{
  key: MemoryPolicyNumericField;
  label: string;
  description: string;
  min: number;
  max: number;
  step?: number;
}> = [
  {
    key: "retrieval_limit",
    label: "Retrieval limit",
    description: "Maximum memories packed into a run context.",
    min: 1,
    max: 50,
  },
  {
    key: "planning_budget_tokens",
    label: "Planning tokens",
    description: "Memory token budget for planning and replan phases.",
    min: 0,
    max: 6000,
  },
  {
    key: "research_budget_tokens",
    label: "Research tokens",
    description: "Memory token budget for stream execution.",
    min: 0,
    max: 4000,
  },
  {
    key: "synthesis_budget_tokens",
    label: "Synthesis tokens",
    description: "Memory token budget for report drafting.",
    min: 0,
    max: 6000,
  },
  {
    key: "grounding_budget_tokens",
    label: "Grounding tokens",
    description: "Memory token budget for claim verification.",
    min: 0,
    max: 1000,
  },
  {
    key: "stale_penalty",
    label: "Stale penalty",
    description: "How strongly older memory is downgraded.",
    min: 0,
    max: 2,
    step: 0.1,
  },
  {
    key: "conflict_penalty",
    label: "Conflict penalty",
    description: "How strongly conflicting memory is downgraded.",
    min: 0,
    max: 2,
    step: 0.1,
  },
];

const researchProfileOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "balanced",
    label: "Balanced",
    description: "Default coverage and sourcing mix.",
  },
  {
    value: "official_first",
    label: "Official first",
    description: "Favor primary and official sources early.",
  },
  {
    value: "wide_net",
    label: "Wide net",
    description: "Broaden the search surface before narrowing.",
  },
];

const recencyPolicyOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "auto",
    label: "Auto",
    description: "Default. Planner decides how much recency matters.",
  },
  {
    value: "recent_first",
    label: "Recent first",
    description: "Bias toward the newest available evidence.",
  },
  {
    value: "evergreen",
    label: "Evergreen",
    description: "Prefer durable references over fresh updates.",
  },
];

const executionModeOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "standard",
    label: "Standard",
    description: "Default. Run immediately unless complexity triggers a gate.",
  },
  {
    value: "deep",
    label: "Deep",
    description: "Allow clarifier and approval gating for deeper work.",
  },
  {
    value: "hitl",
    label: "HITL",
    description: "Always require a human approval path before execution.",
  },
];

const answerStyleOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "analyst",
    label: "Analyst",
    description: "Default. Structured and evidence-first.",
  },
  {
    value: "executive",
    label: "Executive",
    description: "Compressed framing for decision-makers.",
  },
  {
    value: "technical",
    label: "Technical",
    description: "Higher detail and implementation depth.",
  },
];

const trustFloorOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "standard",
    label: "Standard",
    description: "Default trust threshold for mixed-source research.",
  },
  {
    value: "high",
    label: "High",
    description: "Favor better-established outlets and docs.",
  },
  {
    value: "primary",
    label: "Primary",
    description: "Lean hard toward source-of-record material.",
  },
  {
    value: "low",
    label: "Low",
    description: "Allow weaker sources when coverage is sparse.",
  },
];

const citationDisciplineOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "strict",
    label: "Strict",
    description: "Default. Minimize unsupported claims.",
  },
  {
    value: "balanced",
    label: "Balanced",
    description: "Slightly more synthesis freedom before audit.",
  },
];

const claimGranularityOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "atomic",
    label: "Atomic",
    description: "Default. Smaller, easier-to-ground claims.",
  },
  {
    value: "balanced",
    label: "Balanced",
    description: "Slightly broader claim grouping.",
  },
];

const approvalGateOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "auto",
    label: "Auto",
    description: "Default. Runtime decides from plan depth and mode.",
  },
  {
    value: "force",
    label: "Always require approval",
    description: "Pause every eligible deep run for review.",
  },
  {
    value: "skip",
    label: "Skip when allowed",
    description: "Let approved-safe runs continue without a stop.",
  },
];

const policyBiasOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "",
    label: "Follow run policy",
    description: "Default. Use the active run configuration above.",
  },
  ...answerStyleOptions.map((option) => ({
    ...option,
    description:
      option.value === "analyst"
        ? "Structured and evidence-first."
        : option.value === "executive"
          ? "Compressed framing for decision-makers."
          : "Higher detail and implementation depth.",
  })),
];

const recencyBiasOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "",
    label: "Follow run policy",
    description: "Default. Use the active run recency policy above.",
  },
  ...recencyPolicyOptions.map((option) => ({
    ...option,
    description:
      option.value === "auto"
        ? "Let the planner decide how much recency matters."
        : option.value === "recent_first"
          ? "Bias toward newer evidence."
          : "Prefer durable references.",
  })),
];

const trustBiasOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "",
    label: "Follow run policy",
    description: "Default. Use the active source trust floor above.",
  },
  ...trustFloorOptions.map((option) => ({
    ...option,
    description:
      option.value === "standard"
        ? "Mixed-source trust threshold."
        : option.value === "high"
          ? "Favor established sources."
          : option.value === "primary"
            ? "Lean toward source-of-record material."
            : "Allow weaker sources when needed.",
  })),
];

const counterevidenceBiasOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "",
    label: "Follow run policy",
    description: "Default. Use the active counterevidence setting above.",
  },
  {
    value: "true",
    label: "Prefer surfacing disagreements",
    description: "Bias toward showing conflicts and dissent.",
  },
  {
    value: "false",
    label: "Prefer cleaner synthesis",
    description: "Bias toward a tighter primary narrative.",
  },
];

const assetUsageOptions: ReadonlyArray<CustomSelectOption> = [
  {
    value: "reference_source",
    label: "Reference source",
    description: "Treat this as evidence the run should ingest and reference.",
  },
  {
    value: "planning_context",
    label: "Planning context",
    description: "Use this to shape planning and approval before execution starts.",
  },
];

function resolveEffectiveModels(
  publicConfig: PublicRuntimeConfig | undefined,
  override: ModelConfigOverride,
): ModelConfig {
  return {
    lead_model: override.lead_model || publicConfig?.models.lead_model || "default",
    planner_model: override.planner_model || publicConfig?.models.planner_model || "default",
    worker_model: override.worker_model || publicConfig?.models.worker_model || "default",
    verifier_model: override.verifier_model || publicConfig?.models.verifier_model || "default",
    embedding_model:
      override.embedding_model || publicConfig?.models.embedding_model || "default",
    reranker_model:
      override.reranker_model || publicConfig?.models.reranker_model || "default",
  };
}

function formatAssetStatus(asset: { processing_status: string; extraction_method: string; ocr_used: boolean }) {
  const base = asset.processing_status.replaceAll("_", " ");
  const extraction = asset.extraction_method.replaceAll("_", " ");
  if (asset.processing_status !== "ready") {
    return `${base} · ${extraction}`;
  }
  return asset.ocr_used ? `${base} · ${extraction} · OCR` : `${base} · ${extraction}`;
}

function describeAssetUsage(usage: ResearchAssetUsage): string {
  return usage === "planning_context" ? "Planning context" : "Reference source";
}

function summarizeAssetWarnings(warnings: string[]): string | null {
  if (!warnings.length) {
    return null;
  }
  return warnings[0];
}

export function ProjectPanel({ publicConfig }: Pick<RunComposerProps, "publicConfig">) {
  const queryClient = useQueryClient();
  const apiBaseUrl = useResearchStore((state) => state.apiBaseUrl);
  const selectedProjectId = useResearchStore((state) => state.selectedProjectId);
  const setSelectedProjectId = useResearchStore((state) => state.setSelectedProjectId);
  const runInputAssets = useResearchStore((state) => state.runInputAssets);
  const addRunInputAsset = useResearchStore((state) => state.addRunInputAsset);
  const removeRunInputAsset = useResearchStore((state) => state.removeRunInputAsset);
  const stagedRunAssets = useResearchStore((state) => state.stagedRunAssets);
  const addStagedRunAssets = useResearchStore((state) => state.addStagedRunAssets);
  const removeStagedRunAsset = useResearchStore((state) => state.removeStagedRunAsset);

  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDescription, setNewProjectDescription] = useState("");
  const [projectPlanningLabel, setProjectPlanningLabel] = useState("");
  const [projectPlanningUrl, setProjectPlanningUrl] = useState("");
  const [projectReferenceLabel, setProjectReferenceLabel] = useState("");
  const [projectReferenceUrl, setProjectReferenceUrl] = useState("");
  const [runPlanningLabel, setRunPlanningLabel] = useState("");
  const [runPlanningUrl, setRunPlanningUrl] = useState("");
  const [runReferenceLabel, setRunReferenceLabel] = useState("");
  const [runReferenceUrl, setRunReferenceUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const projectPlanningInputRef = useRef<HTMLInputElement>(null);
  const projectReferenceInputRef = useRef<HTMLInputElement>(null);
  const runPlanningInputRef = useRef<HTMLInputElement>(null);
  const runReferenceInputRef = useRef<HTMLInputElement>(null);

  const limits = publicConfig?.capabilities.asset_upload_limits;

  const projectsQuery = useQuery({
    queryKey: ["projects", apiBaseUrl],
    queryFn: () => fetchProjects(apiBaseUrl),
  });

  const projectDetailQuery = useQuery({
    queryKey: ["project-detail", apiBaseUrl, selectedProjectId],
    queryFn: () => fetchProjectDetail(apiBaseUrl, selectedProjectId ?? ""),
    enabled: Boolean(selectedProjectId),
  });

  const createProjectMutation = useMutation({
    mutationFn: () =>
      createProject(apiBaseUrl, {
        name: newProjectName.trim(),
        description: newProjectDescription.trim() || null,
      }),
    onSuccess: async (project) => {
      setNewProjectName("");
      setNewProjectDescription("");
      setShowNewProject(false);
      setSelectedProjectId(project.id);
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to create project.");
    },
  });

  const addProjectAssetsMutation = useMutation({
    mutationFn: (assets: ResearchInputAsset[]) =>
      addProjectAssets(apiBaseUrl, selectedProjectId ?? "", assets),
    onSuccess: async () => {
      setProjectPlanningLabel("");
      setProjectPlanningUrl("");
      setProjectReferenceLabel("");
      setProjectReferenceUrl("");
      await queryClient.invalidateQueries({
        queryKey: ["project-detail", apiBaseUrl, selectedProjectId],
      });
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to save asset.");
    },
  });

  const uploadProjectFilesMutation = useMutation({
    mutationFn: ({
      files,
      usage,
    }: {
      files: FileList;
      usage: ResearchAssetUsage;
    }) => uploadProjectFiles(apiBaseUrl, selectedProjectId ?? "", usage, Array.from(files)),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["project-detail", apiBaseUrl, selectedProjectId],
      });
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
      setError(null);
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to upload files.");
    },
  });

  const stageRunFilesMutation = useMutation({
    mutationFn: ({
      files,
      usage,
    }: {
      files: FileList;
      usage: ResearchAssetUsage;
    }) => stageAssets(apiBaseUrl, usage, Array.from(files)),
    onSuccess: (assets) => {
      addStagedRunAssets(assets);
      setError(null);
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to stage files.");
    },
  });

  const deleteProjectAssetMutation = useMutation({
    mutationFn: (assetId: string) => deleteProjectAsset(apiBaseUrl, selectedProjectId ?? "", assetId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["project-detail", apiBaseUrl, selectedProjectId],
      });
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
      setError(null);
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to remove project asset.");
    },
  });

  const deleteStagedAssetMutation = useMutation({
    mutationFn: (assetId: string) => deleteStagedAsset(apiBaseUrl, assetId),
    onSuccess: (_, assetId) => {
      removeStagedRunAsset(assetId);
      setError(null);
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to remove staged asset.");
    },
  });

  const handleProjectFiles = async (files: FileList | null, usage: ResearchAssetUsage) => {
    if (!files?.length || !selectedProjectId) return;
    uploadProjectFilesMutation.mutate({ files, usage });
  };

  const handleRunFiles = async (files: FileList | null, usage: ResearchAssetUsage) => {
    if (!files?.length) return;
    stageRunFilesMutation.mutate({ files, usage });
  };

  const submitProjectUrl = (usage: ResearchAssetUsage) => {
    const label = usage === "planning_context" ? projectPlanningLabel : projectReferenceLabel;
    const value = usage === "planning_context" ? projectPlanningUrl : projectReferenceUrl;
    if (!label.trim() || !value.trim()) {
      setError("Both a label and URL are required.");
      return;
    }
    addProjectAssetsMutation.mutate([
      {
        source_type: "url",
        usage,
        label: label.trim(),
        url: value.trim(),
        description: null,
      },
    ]);
    setError(null);
  };

  const submitRunUrl = (usage: ResearchAssetUsage) => {
    const label = usage === "planning_context" ? runPlanningLabel : runReferenceLabel;
    const value = usage === "planning_context" ? runPlanningUrl : runReferenceUrl;
    if (!label.trim() || !value.trim()) {
      setError("Both a label and URL are required.");
      return;
    }
    addRunInputAsset({
      source_type: "url",
      usage,
      label: label.trim(),
      url: value.trim(),
      description: null,
    });
    if (usage === "planning_context") {
      setRunPlanningLabel("");
      setRunPlanningUrl("");
    } else {
      setRunReferenceLabel("");
      setRunReferenceUrl("");
    }
    setError(null);
  };

  const projects = projectsQuery.data ?? [];
  const currentProject = projectDetailQuery.data;
  const projectPlanningAssets =
    currentProject?.assets.filter((asset) => asset.usage === "planning_context") ?? [];
  const projectReferenceAssets =
    currentProject?.assets.filter((asset) => asset.usage === "reference_source") ?? [];
  const runPlanningAssets = runInputAssets.filter((asset) => asset.usage === "planning_context");
  const runReferenceAssets = runInputAssets.filter((asset) => asset.usage === "reference_source");
  const stagedPlanningAssets = stagedRunAssets.filter(
    (asset) => asset.usage === "planning_context",
  );
  const stagedReferenceAssets = stagedRunAssets.filter(
    (asset) => asset.usage === "reference_source",
  );

  const renderResearchAssetCard = (
    asset: ResearchAssetRecord,
    options: { removable?: boolean; onRemove?: () => void; projectScoped?: boolean } = {},
  ) => (
    <li key={asset.id} className="project-asset-card">
      <div className="project-asset-card-head">
        <div className="project-asset-title-group">
          <strong>{asset.label}</strong>
          <span className="project-asset-status">{formatAssetStatus(asset)}</span>
        </div>
        <div className="inline-pills">
          <span className="pill muted">{describeAssetUsage(asset.usage)}</span>
          <span className="pill muted">{asset.source_type === "url" ? "URL" : "File"}</span>
          {options.projectScoped ? <span className="pill muted">Project</span> : null}
        </div>
      </div>
      {asset.url ? <code className="project-asset-location">{asset.url}</code> : null}
      {asset.preview_excerpt ? <p className="project-asset-preview">{asset.preview_excerpt}</p> : null}
      {asset.processing_error ? (
        <p className="project-asset-warning danger-text">{asset.processing_error}</p>
      ) : null}
      {summarizeAssetWarnings(asset.warnings) ? (
        <p className="project-asset-warning">{summarizeAssetWarnings(asset.warnings)}</p>
      ) : null}
      <div className="project-asset-meta">
        <span>{asset.file_name ?? asset.content_type ?? "No file metadata"}</span>
        {asset.file_size_bytes ? (
          <span>{(asset.file_size_bytes / 1024).toFixed(1)} KB</span>
        ) : null}
        {asset.page_count ? <span>{asset.page_count} pages</span> : null}
      </div>
      {options.removable && options.onRemove ? (
        <div className="project-asset-actions">
          <button className="inline-action danger-text" onClick={options.onRemove} type="button">
            Remove
          </button>
        </div>
      ) : null}
    </li>
  );

  const renderRunUrlAsset = (asset: ResearchInputAsset, index: number) => (
    <li key={`${asset.label}-${index}`} className="project-asset-card">
      <div className="project-asset-card-head">
        <div className="project-asset-title-group">
          <strong>{asset.label}</strong>
          <span className="project-asset-status">queued URL</span>
        </div>
        <div className="inline-pills">
          <span className="pill muted">{describeAssetUsage(asset.usage)}</span>
          <span className="pill muted">Run</span>
        </div>
      </div>
      {asset.url ? <code className="project-asset-location">{asset.url}</code> : null}
      <div className="project-asset-actions">
        <button className="inline-action danger-text" onClick={() => removeRunInputAsset(index)} type="button">
          Remove
        </button>
      </div>
    </li>
  );

  const renderStagedAssetCard = (asset: StagedAssetRecord) => (
    <li key={asset.id} className="project-asset-card">
      <div className="project-asset-card-head">
        <div className="project-asset-title-group">
          <strong>{asset.label}</strong>
          <span className="project-asset-status">{formatAssetStatus(asset)}</span>
        </div>
        <div className="inline-pills">
          <span className="pill muted">{describeAssetUsage(asset.usage)}</span>
          <span className="pill muted">Run file</span>
        </div>
      </div>
      {asset.preview_excerpt ? <p className="project-asset-preview">{asset.preview_excerpt}</p> : null}
      {asset.processing_error ? (
        <p className="project-asset-warning danger-text">{asset.processing_error}</p>
      ) : null}
      {summarizeAssetWarnings(asset.warnings) ? (
        <p className="project-asset-warning">{summarizeAssetWarnings(asset.warnings)}</p>
      ) : null}
      <div className="project-asset-meta">
        <span>{asset.file_name ?? asset.content_type ?? "Uploaded file"}</span>
        {asset.file_size_bytes ? <span>{(asset.file_size_bytes / 1024).toFixed(1)} KB</span> : null}
        {asset.page_count ? <span>{asset.page_count} pages</span> : null}
      </div>
      <div className="project-asset-actions">
        <button
          className="inline-action danger-text"
          onClick={() => deleteStagedAssetMutation.mutate(asset.id)}
          type="button"
        >
          Remove
        </button>
      </div>
    </li>
  );

  const renderUrlComposer = (
    title: string,
    usage: ResearchAssetUsage,
    scope: "project" | "run",
    label: string,
    value: string,
    setLabel: (value: string) => void,
    setValue: (value: string) => void,
  ) => (
    <div className="project-inline-form">
      <span className="project-inline-title">{title}</span>
      <input
        className="text-input"
        value={label}
        onChange={(event) => setLabel(event.target.value)}
        placeholder="Label"
      />
      <input
        className="text-input"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder="https://..."
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            if (scope === "project") {
              submitProjectUrl(usage);
            } else {
              submitRunUrl(usage);
            }
          }
        }}
      />
      <button
        className="secondary-button"
        onClick={() => {
          if (scope === "project") {
            submitProjectUrl(usage);
          } else {
            submitRunUrl(usage);
          }
        }}
        type="button"
      >
        Add URL
      </button>
    </div>
  );

  return (
    <div className="project-panel">
      <div className="project-panel-header">
        <div>
          <span className="eyebrow">Project</span>
          <p className="project-panel-copy">
            Fresh runs stay isolated. Project runs inherit the project corpus only.
          </p>
        </div>
        <button
          className="project-new-btn"
          onClick={() => { setShowNewProject((v) => !v); setError(null); }}
          type="button"
        >
          {showNewProject ? "Cancel" : "+ New"}
        </button>
      </div>

      {showNewProject ? (
        <div className="project-create-form">
          <input
            className="text-input"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="Project name"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter" && newProjectName.trim()) createProjectMutation.mutate();
            }}
          />
          <input
            className="text-input"
            value={newProjectDescription}
            onChange={(e) => setNewProjectDescription(e.target.value)}
            placeholder="Description (optional)"
          />
          <button
            className="primary-button"
            disabled={!newProjectName.trim() || createProjectMutation.isPending}
            onClick={() => createProjectMutation.mutate()}
            type="button"
          >
            {createProjectMutation.isPending ? "Creating..." : "Create project"}
          </button>
        </div>
      ) : null}

      <div className="project-list">
        <button
          className={`project-item ${!selectedProjectId ? "active" : ""}`}
          onClick={() => setSelectedProjectId(null)}
          type="button"
        >
          <span className="project-item-name">No project</span>
          <span className="project-item-hint">Fresh run, no persistent corpus</span>
        </button>
        {projects.map((project: ProjectSummary) => (
          <button
            key={project.id}
            className={`project-item ${selectedProjectId === project.id ? "active" : ""}`}
            onClick={() => setSelectedProjectId(project.id)}
            type="button"
          >
            <span className="project-item-name">{project.name}</span>
            {project.description ? (
              <span className="project-item-hint">{project.description}</span>
            ) : null}
          </button>
        ))}
      </div>

      <div className="project-summary-card">
        <div className="project-summary-row">
          <span>Project</span>
          <strong>{currentProject?.name ?? "No project"}</strong>
        </div>
        <div className="project-summary-row">
          <span>Persistent corpus</span>
          <strong>{currentProject?.assets.length ?? 0} assets</strong>
        </div>
        <div className="project-summary-row">
          <span>Run-only attachments</span>
          <strong>{runInputAssets.length + stagedRunAssets.length}</strong>
        </div>
        {limits ? (
          <p className="project-summary-hint">
            Upload limit: {limits.max_files_per_batch} files /{" "}
            {(limits.max_file_size_bytes / (1024 * 1024)).toFixed(0)} MB each.
          </p>
        ) : null}
      </div>

      <input
        ref={projectPlanningInputRef}
        className="sr-only-input"
        type="file"
        multiple
        onChange={(event) => void handleProjectFiles(event.target.files, "planning_context")}
      />
      <input
        ref={projectReferenceInputRef}
        className="sr-only-input"
        type="file"
        multiple
        onChange={(event) => void handleProjectFiles(event.target.files, "reference_source")}
      />
      <input
        ref={runPlanningInputRef}
        className="sr-only-input"
        type="file"
        multiple
        onChange={(event) => void handleRunFiles(event.target.files, "planning_context")}
      />
      <input
        ref={runReferenceInputRef}
        className="sr-only-input"
        type="file"
        multiple
        onChange={(event) => void handleRunFiles(event.target.files, "reference_source")}
      />

      <div className="project-corpus-section">
        <div className="project-section-header">
          <span className="eyebrow">Planning Context</span>
          <span className="project-section-copy">
            Read before planning and approval. Shapes the proposed deep research plan.
          </span>
        </div>

        {selectedProjectId ? (
          <div className="project-scope-card">
            <div className="project-scope-head">
              <strong>Project corpus</strong>
              <div className="project-section-actions">
                <button
                  className="project-icon-btn"
                  onClick={() => projectPlanningInputRef.current?.click()}
                  type="button"
                  title="Upload planning files to project"
                >
                  Upload
                </button>
              </div>
            </div>
            {renderUrlComposer(
              "Add persistent planning URL",
              "planning_context",
              "project",
              projectPlanningLabel,
              projectPlanningUrl,
              setProjectPlanningLabel,
              setProjectPlanningUrl,
            )}
            {projectPlanningAssets.length > 0 ? (
              <ul className="project-asset-list detailed">
                {projectPlanningAssets.map((asset) =>
                  renderResearchAssetCard(asset, {
                    removable: true,
                    projectScoped: true,
                    onRemove: () => deleteProjectAssetMutation.mutate(asset.id),
                  }),
                )}
              </ul>
            ) : (
              <p className="project-empty-hint">No persistent planning context in this project.</p>
            )}
          </div>
        ) : null}

        <div className="project-scope-card">
          <div className="project-scope-head">
            <strong>This run only</strong>
            <div className="project-section-actions">
              <button
                className="project-icon-btn"
                onClick={() => runPlanningInputRef.current?.click()}
                type="button"
                title="Upload planning files for this run"
              >
                Upload
              </button>
            </div>
          </div>
          {renderUrlComposer(
            "Add run-only planning URL",
            "planning_context",
            "run",
            runPlanningLabel,
            runPlanningUrl,
            setRunPlanningLabel,
            setRunPlanningUrl,
          )}
          {runPlanningAssets.length > 0 || stagedPlanningAssets.length > 0 ? (
            <ul className="project-asset-list detailed">
              {stagedPlanningAssets.map((asset) => renderStagedAssetCard(asset))}
              {runPlanningAssets.map((asset, index) =>
                renderRunUrlAsset(asset, runInputAssets.indexOf(asset, index)),
              )}
            </ul>
          ) : (
            <p className="project-empty-hint">No run-only planning context attached.</p>
          )}
        </div>
      </div>

      <div className="project-run-section">
        <div className="project-section-header">
          <span className="eyebrow">Reference Sources</span>
          <span className="project-section-copy">
            Ingest as evidence. They inform retrieval and grounding, but are only cited if they
            truly support final claims.
          </span>
        </div>

        {selectedProjectId ? (
          <div className="project-scope-card">
            <div className="project-scope-head">
              <strong>Project corpus</strong>
              <div className="project-section-actions">
                <button
                  className="project-icon-btn"
                  onClick={() => projectReferenceInputRef.current?.click()}
                  type="button"
                  title="Upload reference files to project"
                >
                  Upload
                </button>
              </div>
            </div>
            {renderUrlComposer(
              "Add persistent reference URL",
              "reference_source",
              "project",
              projectReferenceLabel,
              projectReferenceUrl,
              setProjectReferenceLabel,
              setProjectReferenceUrl,
            )}
            {projectReferenceAssets.length > 0 ? (
              <ul className="project-asset-list detailed">
                {projectReferenceAssets.map((asset) =>
                  renderResearchAssetCard(asset, {
                    removable: true,
                    projectScoped: true,
                    onRemove: () => deleteProjectAssetMutation.mutate(asset.id),
                  }),
                )}
              </ul>
            ) : (
              <p className="project-empty-hint">No persistent reference sources in this project.</p>
            )}
          </div>
        ) : null}

        <div className="project-scope-card">
          <div className="project-scope-head">
            <strong>This run only</strong>
            <div className="project-section-actions">
              <button
                className="project-icon-btn"
                onClick={() => runReferenceInputRef.current?.click()}
                type="button"
                title="Upload reference files for this run"
              >
                Upload
              </button>
            </div>
          </div>
          {renderUrlComposer(
            "Add run-only reference URL",
            "reference_source",
            "run",
            runReferenceLabel,
            runReferenceUrl,
            setRunReferenceLabel,
            setRunReferenceUrl,
          )}
          {runReferenceAssets.length > 0 || stagedReferenceAssets.length > 0 ? (
            <ul className="project-asset-list detailed">
              {stagedReferenceAssets.map((asset) => renderStagedAssetCard(asset))}
              {runReferenceAssets.map((asset, index) =>
                renderRunUrlAsset(asset, runInputAssets.indexOf(asset, index)),
              )}
            </ul>
          ) : (
            <p className="project-empty-hint">No run-only reference sources attached.</p>
          )}
        </div>
      </div>

      {error ? <p className="error-text project-error">{error}</p> : null}
    </div>
  );
}

function CommonSettingsFields({ publicConfig, onOpenPanel }: RunComposerProps) {
  const apiBaseUrl = useResearchStore((state) => state.apiBaseUrl);
  const setApiBaseUrl = useResearchStore((state) => state.setApiBaseUrl);
  const profileId = useResearchStore((state) => state.profileId);
  const setProfileId = useResearchStore((state) => state.setProfileId);
  const budget = useResearchStore((state) => state.budget);
  const memoryPolicyOverride = useResearchStore((state) => state.memoryPolicyOverride);
  const executionMode = useResearchStore((state) => state.executionMode);
  const setExecutionMode = useResearchStore((state) => state.setExecutionMode);
  const sourceSelection = useResearchStore((state) => state.sourceSelection);
  const selectedProjectId = useResearchStore((state) => state.selectedProjectId);
  const runInputAssets = useResearchStore((state) => state.runInputAssets);
  const stagedRunAssets = useResearchStore((state) => state.stagedRunAssets);
  const modelConfigOverride = useResearchStore((state) => state.modelConfigOverride);
  const effectiveModels = resolveEffectiveModels(publicConfig, modelConfigOverride);

  return (
    <>
      <label className="field">
        <span className="field-label">API endpoint</span>
        <input
          className="text-input"
          value={apiBaseUrl}
          onChange={(event) => setApiBaseUrl(event.target.value)}
          placeholder="http://127.0.0.1:8000"
        />
      </label>

      <label className="field">
        <span className="field-label">Profile ID</span>
        <input
          className="text-input"
          value={profileId}
          onChange={(event) => setProfileId(event.target.value || "default")}
          placeholder="default"
        />
      </label>

      <label className="field">
        <span className="field-label">Execution mode</span>
        <CustomSelect
          ariaLabel="Execution mode"
          value={executionMode}
          onChange={(value) => setExecutionMode(value as ExecutionMode)}
          options={executionModeOptions}
        />
      </label>

      <details className="settings-summary-details">
        <summary className="settings-summary-trigger">Config summary</summary>
        <div className="config-callout settings-summary-card">
          <dl className="settings-summary-grid">
            <div className="settings-summary-row">
              <dt>Budget</dt>
              <dd>
                {budget.max_streams} streams / {budget.max_queries_per_stream} queries /{" "}
                {budget.max_sources_per_stream} sources
              </dd>
            </div>
            <div className="settings-summary-row">
              <dt>Memory</dt>
              <dd>
                {memoryPolicyOverride.enabled ? "Enabled" : "Disabled"},{" "}
                {memoryPolicyOverride.retrieval_limit} retrieval slots
              </dd>
            </div>
            <div className="settings-summary-row">
              <dt>Workflow</dt>
              <dd>{executionMode}</dd>
            </div>
            <div className="settings-summary-row">
              <dt>Project</dt>
              <dd>{selectedProjectId ? "Attached" : "Fresh run"}</dd>
            </div>
            <div className="settings-summary-row">
              <dt>Sources</dt>
              <dd>
                {sourceSelection.length > 0
                  ? `${sourceSelection.length} selected`
                  : "Deployment default"}
              </dd>
            </div>
            <div className="settings-summary-row">
              <dt>Inputs</dt>
              <dd>{runInputAssets.length + stagedRunAssets.length} run-only items</dd>
            </div>
            <div className="settings-summary-row">
              <dt>Models</dt>
              <dd>
                {effectiveModels.lead_model} / {effectiveModels.planner_model} /{" "}
                {effectiveModels.worker_model} / {effectiveModels.verifier_model}
              </dd>
            </div>
            {publicConfig ? (
              <div className="settings-summary-row settings-summary-row-wide">
                <dt>Runtime</dt>
                <dd>
                  {publicConfig.backends.llm} LLM, {publicConfig.backends.search} search,{" "}
                  {publicConfig.backends.fetch} fetch, lead {publicConfig.models.lead_model}
                </dd>
              </div>
            ) : null}
          </dl>
        </div>
      </details>

      {onOpenPanel ? (
        <div className="button-row composer-actions">
          <button
            className="secondary-button"
            onClick={() => onOpenPanel("workflow")}
            type="button"
          >
            Workflow
          </button>
          <button
            className="secondary-button"
            onClick={() => onOpenPanel("models")}
            type="button"
          >
            Models
          </button>
          <button
            className="secondary-button"
            onClick={() => onOpenPanel("profile")}
            type="button"
          >
            Profile
          </button>
        </div>
      ) : null}
    </>
  );
}

export function RunComposer({ publicConfig, onOpenPanel }: RunComposerProps) {
  return (
    <>
      <ProjectPanel publicConfig={publicConfig} />
      <section className="panel composer-panel">
        <div className="panel-header">
          <p className="eyebrow">Quick settings</p>
        </div>
        <CommonSettingsFields publicConfig={publicConfig} onOpenPanel={onOpenPanel} />
      </section>
    </>
  );
}

function BehaviorSettingsFields() {
  const agentConfig = useResearchStore((state) => state.agentConfig);
  const updateAgentConfig = useResearchStore((state) => state.updateAgentConfig);

  return (
    <div className="settings-grid">
      <label className="field">
        <span className="field-label">Research profile</span>
        <CustomSelect
          ariaLabel="Research profile"
          value={agentConfig.research_profile}
          onChange={(value) => updateAgentConfig({ research_profile: value as never })}
          options={researchProfileOptions}
        />
        <span className="field-hint">Coverage and sourcing strategy.</span>
      </label>

      <label className="field">
        <span className="field-label">Recency policy</span>
        <CustomSelect
          ariaLabel="Recency policy"
          value={agentConfig.recency_policy}
          onChange={(value) => updateAgentConfig({ recency_policy: value as never })}
          options={recencyPolicyOptions}
        />
        <span className="field-hint">How much freshness matters.</span>
      </label>

      <label className="field">
        <span className="field-label">Answer style</span>
        <CustomSelect
          ariaLabel="Answer style"
          value={agentConfig.answer_style}
          onChange={(value) => updateAgentConfig({ answer_style: value as never })}
          options={answerStyleOptions}
        />
        <span className="field-hint">Report framing and depth.</span>
      </label>

      <label className="field">
        <span className="field-label">Source trust floor</span>
        <CustomSelect
          ariaLabel="Source trust floor"
          value={agentConfig.source_trust_floor}
          onChange={(value) => updateAgentConfig({ source_trust_floor: value as never })}
          options={trustFloorOptions}
        />
        <span className="field-hint">Minimum acceptable source quality.</span>
      </label>
    </div>
  );
}

export function SettingsDrawerPanel({ publicConfig, onOpenPanel }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <BehaviorSettingsFields />
      <div className="drawer-divider" />
      <p className="drawer-section-label">Connection</p>
      <CommonSettingsFields publicConfig={publicConfig} onOpenPanel={onOpenPanel} />
    </div>
  );
}

function AgentPolicyFields() {
  const agentConfig = useResearchStore((state) => state.agentConfig);
  const updateAgentConfig = useResearchStore((state) => state.updateAgentConfig);

  return (
    <>
      <div className="settings-grid">
        <label className="field">
          <span className="field-label">Citation discipline</span>
          <CustomSelect
            ariaLabel="Citation discipline"
            value={agentConfig.citation_discipline}
            onChange={(value) => updateAgentConfig({ citation_discipline: value as never })}
            options={citationDisciplineOptions}
          />
          <span className="field-hint">
            Default: {DEFAULT_AGENT_CONFIG.citation_discipline}. Keeps unsupported claims out of the
            final body.
          </span>
        </label>

        <label className="field">
          <span className="field-label">Claim granularity</span>
          <CustomSelect
            ariaLabel="Claim granularity"
            value={agentConfig.claim_granularity}
            onChange={(value) => updateAgentConfig({ claim_granularity: value as never })}
            options={claimGranularityOptions}
          />
          <span className="field-hint">
            Default: {DEFAULT_AGENT_CONFIG.claim_granularity}. Smaller claims are easier to verify.
          </span>
        </label>
      </div>

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={agentConfig.include_counterevidence}
          onChange={(event) =>
            updateAgentConfig({ include_counterevidence: event.target.checked })
          }
        />
        <span>
          Surface counterevidence and disagreements instead of optimizing for a single story.
        </span>
      </label>
    </>
  );
}

function ProfilePreferenceFields() {
  const profileId = useResearchStore((state) => state.profileId);
  const agentConfig = useResearchStore((state) => state.agentConfig);
  const profilePreferences = useResearchStore((state) => state.profilePreferences);
  const updateProfilePreferences = useResearchStore(
    (state) => state.updateProfilePreferences,
  );

  return (
    <>
      <div className="config-callout">
        <p>
          These preferences are saved under profile <code>{profileId}</code> before each run
          starts.
        </p>
        <p>
          “Follow run policy” means: answer style {agentConfig.answer_style}, recency{" "}
          {agentConfig.recency_policy}, trust floor {agentConfig.source_trust_floor}, and
          counterevidence {agentConfig.include_counterevidence ? "enabled" : "disabled"}.
        </p>
      </div>

      <div className="settings-grid">
        <label className="field settings-grid-wide">
          <span className="field-label">Preferred sources</span>
          <input
            className="text-input"
            value={profilePreferences.preferred_source_patterns.join(", ")}
            onChange={(event) =>
              updateProfilePreferences({
                preferred_source_patterns: event.target.value
                  .split(",")
                  .map((value) => value.trim())
                  .filter(Boolean),
              })
            }
            placeholder="docs, official, arxiv, sec.gov"
          />
        </label>

        <label className="field settings-grid-wide">
          <span className="field-label">Avoided sources</span>
          <input
            className="text-input"
            value={profilePreferences.avoided_source_patterns.join(", ")}
            onChange={(event) =>
              updateProfilePreferences({
                avoided_source_patterns: event.target.value
                  .split(",")
                  .map((value) => value.trim())
                  .filter(Boolean),
              })
            }
            placeholder="content farms, shorteners"
          />
        </label>

        <label className="field">
          <span className="field-label">Style bias</span>
          <CustomSelect
            ariaLabel="Style bias"
            value={profilePreferences.answer_style_bias ?? ""}
            onChange={(value) =>
              updateProfilePreferences({
                answer_style_bias: value ? (value as never) : null,
              })
            }
            options={policyBiasOptions}
          />
        </label>

        <label className="field">
          <span className="field-label">Recency bias</span>
          <CustomSelect
            ariaLabel="Recency bias"
            value={profilePreferences.recency_bias ?? ""}
            onChange={(value) =>
              updateProfilePreferences({
                recency_bias: value ? (value as never) : null,
              })
            }
            options={recencyBiasOptions}
          />
        </label>

        <label className="field">
          <span className="field-label">Trust floor bias</span>
          <CustomSelect
            ariaLabel="Trust floor bias"
            value={profilePreferences.source_trust_floor_bias ?? ""}
            onChange={(value) =>
              updateProfilePreferences({
                source_trust_floor_bias: value ? (value as never) : null,
              })
            }
            options={trustBiasOptions}
          />
        </label>

        <label className="field">
          <span className="field-label">Counterevidence bias</span>
          <CustomSelect
            ariaLabel="Counterevidence bias"
            value={
              profilePreferences.include_counterevidence_bias === null
                ? ""
                : profilePreferences.include_counterevidence_bias
                  ? "true"
                  : "false"
            }
            onChange={(value) =>
              updateProfilePreferences({
                include_counterevidence_bias:
                  value === "" ? null : value === "true",
              })
            }
            options={counterevidenceBiasOptions}
          />
        </label>
      </div>
    </>
  );
}

function ModelSelectionFields({ publicConfig }: RunComposerProps) {
  const modelConfigOverride = useResearchStore((state) => state.modelConfigOverride);
  const updateModelConfigOverride = useResearchStore(
    (state) => state.updateModelConfigOverride,
  );
  const resetModelConfigOverride = useResearchStore(
    (state) => state.resetModelConfigOverride,
  );
  const effectiveModels = resolveEffectiveModels(publicConfig, modelConfigOverride);

  const handleModelChange =
    (key: keyof ModelConfig) => (event: ChangeEvent<HTMLInputElement>) => {
      const next = event.target.value.trim();
      const defaultValue = publicConfig?.models[key];
      updateModelConfigOverride({
        [key]: !next || next === defaultValue ? null : next,
      });
    };

  return (
    <>
      <div className="config-callout">
        <p>Model changes apply to the next run immediately. No backend restart is required.</p>
        <p>Leaving a model override blank falls back to the server default shown in the hint.</p>
      </div>

      <div className="settings-grid">
        <label className="field settings-grid-wide">
          <span className="field-label">Lead model</span>
          <input
            className="text-input"
            value={effectiveModels.lead_model}
            onChange={handleModelChange("lead_model")}
            placeholder={publicConfig?.models.lead_model ?? "gpt-5.4"}
          />
          <span className="field-hint">
            Report synthesis and final high-level drafting. Default:{" "}
            {publicConfig?.models.lead_model ?? "unknown"}.
          </span>
        </label>

        <label className="field">
          <span className="field-label">Planner model</span>
          <input
            className="text-input"
            value={effectiveModels.planner_model}
            onChange={handleModelChange("planner_model")}
            placeholder={publicConfig?.models.planner_model ?? "gpt-5.4"}
          />
          <span className="field-hint">
            Preview planning, post-approval execution planning, and replans.
          </span>
        </label>

        <label className="field">
          <span className="field-label">Worker model</span>
          <input
            className="text-input"
            value={effectiveModels.worker_model}
            onChange={handleModelChange("worker_model")}
            placeholder={publicConfig?.models.worker_model ?? "gpt-5.4-mini"}
          />
          <span className="field-hint">Streams and note writing.</span>
        </label>

        <label className="field">
          <span className="field-label">Verifier model</span>
          <input
            className="text-input"
            value={effectiveModels.verifier_model}
            onChange={handleModelChange("verifier_model")}
            placeholder={publicConfig?.models.verifier_model ?? "gpt-5.4-mini"}
          />
          <span className="field-hint">Claim verification and repair.</span>
        </label>

        <label className="field">
          <span className="field-label">Embedding model</span>
          <input
            className="text-input"
            value={effectiveModels.embedding_model}
            onChange={handleModelChange("embedding_model")}
            placeholder={publicConfig?.models.embedding_model ?? "text-embedding-3-large"}
          />
          <span className="field-hint">Only used when embeddings are enabled.</span>
        </label>

        <label className="field">
          <span className="field-label">Reranker model</span>
          <input
            className="text-input"
            value={effectiveModels.reranker_model}
            onChange={handleModelChange("reranker_model")}
            placeholder={publicConfig?.models.reranker_model ?? "BAAI/bge-reranker-v2-m3"}
          />
          <span className="field-hint">Only used when reranking is enabled.</span>
        </label>
      </div>

      <div className="button-row">
        <button className="secondary-button" onClick={resetModelConfigOverride} type="button">
          Reset models
        </button>
      </div>
    </>
  );
}

function ExecutionWorkflowFields({ publicConfig }: RunComposerProps) {
  const requirePlanApproval = useResearchStore((state) => state.requirePlanApproval);
  const setRequirePlanApproval = useResearchStore((state) => state.setRequirePlanApproval);
  const clarifierConfig = useResearchStore((state) => state.clarifierConfig);
  const updateClarifierConfig = useResearchStore((state) => state.updateClarifierConfig);
  const asyncSubmit = useResearchStore((state) => state.asyncSubmit);
  const setAsyncSubmit = useResearchStore((state) => state.setAsyncSubmit);

  return (
    <>
      <div className="config-callout">
        <p>
          This drawer controls approval, clarification, and async behavior. Execution mode stays on
          the main rail so it is always visible before you start a run.
        </p>
        <p>Approval gate defaults to Auto, which lets the runtime decide from mode and plan depth.</p>
      </div>

      <div className="settings-grid">
        <label className="field">
          <span className="field-label">Approval gate</span>
          <CustomSelect
            ariaLabel="Approval gate"
            value={
              requirePlanApproval === null ? "auto" : requirePlanApproval ? "force" : "skip"
            }
            onChange={(value) =>
              setRequirePlanApproval(
                value === "auto" ? null : value === "force",
              )
            }
            options={approvalGateOptions}
          />
          <span className="field-hint">
            Default: Auto. The harness evaluates execution mode and plan complexity.
          </span>
        </label>

        <label className="field">
          <span className="field-label">Clarifier max questions</span>
          <input
            className="text-input"
            type="number"
            min={0}
            max={5}
            value={clarifierConfig.max_questions}
            onChange={(event) =>
              updateClarifierConfig({ max_questions: Number(event.target.value) })
            }
          />
        </label>

        <label className="field">
          <span className="field-label">Ambiguity threshold</span>
          <input
            className="text-input"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={clarifierConfig.ambiguity_threshold}
            onChange={(event) =>
              updateClarifierConfig({ ambiguity_threshold: Number(event.target.value) })
            }
          />
        </label>
      </div>

      <div className="toggle-grid">
        <label className="toggle-card">
          <input
            type="checkbox"
            checked={clarifierConfig.enabled}
            onChange={(event) => updateClarifierConfig({ enabled: event.target.checked })}
          />
          <span>Enable clarifier</span>
        </label>

        <label className="toggle-card">
          <input
            type="checkbox"
            checked={clarifierConfig.require_response_for_deep}
            onChange={(event) =>
              updateClarifierConfig({
                require_response_for_deep: event.target.checked,
              })
            }
          />
          <span>Always ask before deep runs</span>
        </label>

        <label className="toggle-card">
          <input
            type="checkbox"
            checked={asyncSubmit}
            onChange={(event) => setAsyncSubmit(event.target.checked)}
          />
          <span>Submit as async job</span>
        </label>
      </div>

      {publicConfig?.capabilities.supports_deep_approval === false ? (
        <div className="config-callout">
          <p>The backend currently has deep approval disabled. These controls will be ignored.</p>
        </div>
      ) : null}
    </>
  );
}

function SourceSelectionFields({ publicConfig }: RunComposerProps) {
  const sourceSelection = useResearchStore((state) => state.sourceSelection);
  const toggleSourceSelection = useResearchStore((state) => state.toggleSourceSelection);
  const setSourceSelection = useResearchStore((state) => state.setSourceSelection);

  const availableSources = publicConfig?.available_sources ?? [];

  return (
    <>
      <div className="config-callout">
        <p>
          Source selection constrains which search and fetch backends the planner and worker may
          use for the next run.
        </p>
        <p>
          Deployment default means the backend’s default source set, not every provider in the
          catalog.
        </p>
      </div>

      <div className="toggle-grid">
        {availableSources.map((source) => (
          <label className="toggle-card" key={source.id}>
            <input
              type="checkbox"
              checked={sourceSelection.includes(source.id)}
              onChange={() => toggleSourceSelection(source.id)}
            />
            <span>{source.name}</span>
            <span className="muted-text">
              {source.backend_kind} · {source.configured ? "configured" : "missing"}
            </span>
          </label>
        ))}
      </div>

      <div className="button-row">
        <button
          className="secondary-button"
          onClick={() => setSourceSelection(publicConfig?.default_source_selection ?? [])}
          type="button"
        >
          Reset to deployment defaults
        </button>
      </div>
    </>
  );
}

function MemoryHarnessFields({ publicConfig }: RunComposerProps) {
  const memoryPolicyOverride = useResearchStore((state) => state.memoryPolicyOverride);
  const updateMemoryPolicyOverride = useResearchStore(
    (state) => state.updateMemoryPolicyOverride,
  );
  const supportsMemoryHarness =
    publicConfig?.capabilities.supports_memory_harness ?? true;
  const memoryLimits = publicConfig?.capabilities.memory_policy_limits;

  const handleNumericChange =
    (key: MemoryPolicyNumericField) => (event: ChangeEvent<HTMLInputElement>) => {
      updateMemoryPolicyOverride({ [key]: Number(event.target.value) });
    };

  return (
    <>
      {!supportsMemoryHarness ? (
        <div className="config-callout">
          <p>The backend has profile memory disabled. These controls will not affect runs.</p>
        </div>
      ) : null}

      <div className="config-callout">
        <p>These settings apply as a per-run override. They do not rewrite the saved profile.</p>
      </div>

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={memoryPolicyOverride.enabled}
          onChange={(event) =>
            updateMemoryPolicyOverride({ enabled: event.target.checked })
          }
        />
        <span>Enable profile memory retrieval and context packing for this run.</span>
      </label>

      <div className="settings-grid">
        {memoryNumericFields.map((field) => (
          <label className="field" key={field.key}>
            <span className="field-label">{field.label}</span>
            <input
              className="text-input"
              type="number"
              min={memoryLimits?.[field.key]?.min ?? field.min}
              max={memoryLimits?.[field.key]?.max ?? field.max}
              step={field.step ?? 1}
              value={memoryPolicyOverride[field.key]}
              onChange={handleNumericChange(field.key)}
              disabled={!supportsMemoryHarness}
            />
            <span className="field-hint">
              {field.description} Range: {memoryLimits?.[field.key]?.min ?? field.min}-
              {memoryLimits?.[field.key]?.max ?? field.max}.
            </span>
          </label>
        ))}
      </div>

      <div className="toggle-grid">
        <label className="toggle-card">
          <input
            type="checkbox"
            checked={memoryPolicyOverride.allow_preference_in_planning}
            onChange={(event) =>
              updateMemoryPolicyOverride({
                allow_preference_in_planning: event.target.checked,
              })
            }
            disabled={!supportsMemoryHarness}
          />
          <span>Allow preferences in planning</span>
        </label>

        <label className="toggle-card">
          <input
            type="checkbox"
            checked={memoryPolicyOverride.allow_preference_in_research}
            onChange={(event) =>
              updateMemoryPolicyOverride({
                allow_preference_in_research: event.target.checked,
              })
            }
            disabled={!supportsMemoryHarness}
          />
          <span>Allow preferences in research</span>
        </label>

        <label className="toggle-card">
          <input
            type="checkbox"
            checked={memoryPolicyOverride.allow_preference_in_synthesis}
            onChange={(event) =>
              updateMemoryPolicyOverride({
                allow_preference_in_synthesis: event.target.checked,
              })
            }
            disabled={!supportsMemoryHarness}
          />
          <span>Allow preferences in synthesis</span>
        </label>

        <label className="toggle-card">
          <input
            type="checkbox"
            checked={memoryPolicyOverride.allow_preference_in_grounding}
            onChange={(event) =>
              updateMemoryPolicyOverride({
                allow_preference_in_grounding: event.target.checked,
              })
            }
            disabled={!supportsMemoryHarness}
          />
          <span>Allow preferences in grounding</span>
        </label>
      </div>
    </>
  );
}

function BudgetFields({ publicConfig }: RunComposerProps) {
  const budget = useResearchStore((state) => state.budget);
  const updateBudget = useResearchStore((state) => state.updateBudget);
  const budgetLimits = publicConfig?.capabilities.budget_limits;

  const handleBudgetChange =
    (key: keyof typeof budget) => (event: ChangeEvent<HTMLInputElement>) => {
      updateBudget({ [key]: Number(event.target.value) });
    };

  return (
    <div className="settings-grid">
      {budgetFields.map((field) => (
        <label className="field" key={field.key}>
          <span className="field-label">{field.label}</span>
          <input
            className="text-input"
            type="number"
            min={budgetLimits?.[field.key]?.min ?? field.min}
            max={budgetLimits?.[field.key]?.max ?? field.max}
            value={budget[field.key]}
            onChange={handleBudgetChange(field.key)}
          />
          <span className="field-hint">
            {field.description} Range: {budgetLimits?.[field.key]?.min ?? field.min}-
            {budgetLimits?.[field.key]?.max ?? field.max}.
          </span>
        </label>
      ))}
    </div>
  );
}

export function WorkflowDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <ExecutionWorkflowFields publicConfig={publicConfig} />
      <div className="drawer-divider" />
      <p className="drawer-section-label">Output policy</p>
      <AgentPolicyFields />
    </div>
  );
}

export function SourcesDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <SourceSelectionFields publicConfig={publicConfig} />
    </div>
  );
}

export function BudgetDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <BudgetFields publicConfig={publicConfig} />
    </div>
  );
}

export function ModelsDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <ModelSelectionFields publicConfig={publicConfig} />
    </div>
  );
}

export function ProfileDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <ProfilePreferenceFields />
      <div className="drawer-divider" />
      <p className="drawer-section-label">Memory harness</p>
      <MemoryHarnessFields publicConfig={publicConfig} />
    </div>
  );
}
