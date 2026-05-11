"use client";

import type { ChangeEvent } from "react";
import { useMemo, useRef, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { CustomSelect, type CustomSelectOption } from "@/components/ui/custom-select";
import {
  createProject,
  deleteProjectAsset,
  deleteStagedAsset,
  fetchProjectDetail,
  fetchProjects,
  stageAssets,
  uploadProjectFiles,
} from "@/lib/api";
import {
  DEFAULT_AGENT_CONFIG,
  DEFAULT_BUDGET,
  DEFAULT_REPORT_OUTPUT_CONFIG,
} from "@/lib/defaults";
import type {
  AgentConfig,
  BudgetPolicy,
  ClarifierConfig,
  MemoryPolicyNumericField,
  ModelConfig,
  ModelConfigOverride,
  ResearchAssetRecord,
  ResearchAssetUsage,
  PublicRuntimeConfig,
  ProjectSummary,
  ReportOutputConfig,
  StagedAssetRecord,
} from "@/lib/types";
import { useResearchStore } from "@/store/use-research-store";

export type FocusedDrawerKey = "workflow" | "sources" | "budget" | "models" | "profile";

interface RunComposerProps {
  publicConfig: PublicRuntimeConfig | undefined;
  onOpenPanel?: (panel: FocusedDrawerKey) => void;
}

interface RunConfigPreset {
  key: "recommended" | "quick" | "audit";
  label: string;
  eyebrow: string;
  description: string;
  budget: BudgetPolicy;
  reportOutputConfig: ReportOutputConfig;
  agentConfig: AgentConfig;
  requirePlanApproval: boolean;
  clarifierConfig: ClarifierConfig;
}

const budgetFields = [
  {
    key: "max_streams",
    label: "Research streams",
    description: "Planner lanes.",
    min: 1,
    max: 30,
  },
  {
    key: "max_replans",
    label: "Replans",
    description: "Gap-closure loops.",
    min: 0,
    max: 5,
  },
  {
    key: "max_queries_per_stream",
    label: "Queries per stream",
    description: "Researcher queries.",
    min: 1,
    max: 25,
  },
  {
    key: "max_results_per_query",
    label: "Results per query",
    description: "Results retained.",
    min: 1,
    max: 20,
  },
  {
    key: "max_sources_per_stream",
    label: "Sources per stream",
    description: "Fetched sources kept.",
    min: 1,
    max: 20,
  },
  {
    key: "per_domain_limit",
    label: "Per-domain cap",
    description: "Per-domain source limit.",
    min: 1,
    max: 10,
  },
] as const;

const runConfigPresets: ReadonlyArray<RunConfigPreset> = [
  {
    key: "recommended",
    label: "Recommended",
    eyebrow: "Default",
    description: "Use deployment defaults with balanced evidence handling.",
    budget: DEFAULT_BUDGET,
    reportOutputConfig: DEFAULT_REPORT_OUTPUT_CONFIG,
    agentConfig: DEFAULT_AGENT_CONFIG,
    requirePlanApproval: false,
    clarifierConfig: {
      enabled: true,
      max_questions: 2,
      ambiguity_threshold: 0.45,
      require_response_for_deep: false,
    },
  },
  {
    key: "quick",
    label: "Quick scan",
    eyebrow: "Fast",
    description: "Smaller search surface for exploratory prompts.",
    budget: {
      max_streams: 8,
      max_replans: 0,
      max_queries_per_stream: 8,
      max_results_per_query: 5,
      max_sources_per_stream: 2,
      per_domain_limit: 2,
    },
    reportOutputConfig: {
      min_words: 500,
      max_words: 1200,
    },
    agentConfig: {
      ...DEFAULT_AGENT_CONFIG,
      answer_style: "executive",
      claim_granularity: "balanced",
    },
    requirePlanApproval: false,
    clarifierConfig: {
      enabled: true,
      max_questions: 1,
      ambiguity_threshold: 0.5,
      require_response_for_deep: false,
    },
  },
  {
    key: "audit",
    label: "Evidence audit",
    eyebrow: "Deep",
    description: "Higher coverage, stricter sourcing, and plan review.",
    budget: {
      max_streams: 30,
      max_replans: 2,
      max_queries_per_stream: 25,
      max_results_per_query: 8,
      max_sources_per_stream: 5,
      per_domain_limit: 3,
    },
    reportOutputConfig: {
      min_words: 1800,
      max_words: 4200,
    },
    agentConfig: {
      ...DEFAULT_AGENT_CONFIG,
      research_profile: "official_first",
      answer_style: "technical",
      source_trust_floor: "high",
      citation_discipline: "strict",
      claim_granularity: "atomic",
      include_counterevidence: true,
    },
    requirePlanApproval: true,
    clarifierConfig: {
      enabled: true,
      max_questions: 3,
      ambiguity_threshold: 0.35,
      require_response_for_deep: true,
    },
  },
];

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

function uniqueModelOptions(values: Array<string | null | undefined>): string[] {
  return Array.from(
    new Set(
      values
        .map((value) => value?.trim())
        .filter((value): value is string => Boolean(value)),
    ),
  );
}

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

function clampPresetNumber(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function resolvePresetBudget(
  preset: RunConfigPreset,
  publicConfig: PublicRuntimeConfig | undefined,
): BudgetPolicy {
  const serverBudget =
    preset.key === "recommended" && publicConfig?.default_budget
      ? publicConfig.default_budget
      : preset.budget;
  const limits = publicConfig?.capabilities.budget_limits;
  if (!limits) return serverBudget;

  return {
    max_streams: clampPresetNumber(
      serverBudget.max_streams,
      limits.max_streams?.min ?? 1,
      limits.max_streams?.max ?? 30,
    ),
    max_replans: clampPresetNumber(
      serverBudget.max_replans,
      limits.max_replans?.min ?? 0,
      limits.max_replans?.max ?? 5,
    ),
    max_queries_per_stream: clampPresetNumber(
      serverBudget.max_queries_per_stream,
      limits.max_queries_per_stream?.min ?? 1,
      limits.max_queries_per_stream?.max ?? 25,
    ),
    max_results_per_query: clampPresetNumber(
      serverBudget.max_results_per_query,
      limits.max_results_per_query?.min ?? 1,
      limits.max_results_per_query?.max ?? 20,
    ),
    max_sources_per_stream: clampPresetNumber(
      serverBudget.max_sources_per_stream,
      limits.max_sources_per_stream?.min ?? 1,
      limits.max_sources_per_stream?.max ?? 20,
    ),
    per_domain_limit: clampPresetNumber(
      serverBudget.per_domain_limit,
      limits.per_domain_limit?.min ?? 1,
      limits.per_domain_limit?.max ?? 10,
    ),
  };
}

function PresetConfigurationCards({ publicConfig }: RunComposerProps) {
  const [appliedPreset, setAppliedPreset] = useState<string | null>(null);
  const updateBudget = useResearchStore((state) => state.updateBudget);
  const updateReportOutputConfig = useResearchStore(
    (state) => state.updateReportOutputConfig,
  );
  const updateAgentConfig = useResearchStore((state) => state.updateAgentConfig);
  const setRequirePlanApproval = useResearchStore((state) => state.setRequirePlanApproval);
  const updateClarifierConfig = useResearchStore((state) => state.updateClarifierConfig);
  const setSourceSelection = useResearchStore((state) => state.setSourceSelection);
  const resetModelConfigOverride = useResearchStore(
    (state) => state.resetModelConfigOverride,
  );

  const applyPreset = (preset: RunConfigPreset) => {
    const nextBudget = resolvePresetBudget(preset, publicConfig);
    const nextAgentConfig =
      preset.key === "recommended" && publicConfig?.default_agent_config
        ? publicConfig.default_agent_config
        : preset.agentConfig;

    updateBudget(nextBudget);
    updateReportOutputConfig(preset.reportOutputConfig);
    updateAgentConfig(nextAgentConfig);
    setRequirePlanApproval(preset.requirePlanApproval);
    updateClarifierConfig(preset.clarifierConfig);
    setSourceSelection(publicConfig?.default_source_selection ?? []);
    resetModelConfigOverride();
    setAppliedPreset(preset.key);
  };

  return (
    <div className="preset-card-grid" aria-label="Configuration presets">
      {runConfigPresets.map((preset) => {
        const previewBudget = resolvePresetBudget(preset, publicConfig);
        const isApplied = appliedPreset === preset.key;
        return (
          <button
            aria-pressed={isApplied}
            className={`preset-card ${isApplied ? "active" : ""}`}
            key={preset.key}
            onClick={() => applyPreset(preset)}
            type="button"
          >
            <span className="preset-card-eyebrow">{preset.eyebrow}</span>
            <strong>{preset.label}</strong>
            <span>{preset.description}</span>
            <small>
              {previewBudget.max_streams} streams ·{" "}
              {previewBudget.max_queries_per_stream} queries ·{" "}
              {preset.reportOutputConfig.min_words}-{preset.reportOutputConfig.max_words} words
            </small>
          </button>
        );
      })}
    </div>
  );
}

function PanelShortcutGrid({ onOpenPanel }: { onOpenPanel: (panel: FocusedDrawerKey) => void }) {
  const shortcuts: ReadonlyArray<{
    key: FocusedDrawerKey;
    label: string;
    description: string;
  }> = [
    {
      key: "workflow",
      label: "Workflow",
      description: "Approval, clarifier, async job behavior.",
    },
    {
      key: "sources",
      label: "Sources",
      description: "Search providers, enabled tools, source defaults.",
    },
    {
      key: "budget",
      label: "Config",
      description: "Depth, query limits, report length.",
    },
    {
      key: "models",
      label: "Models",
      description: "Deployment defaults and temporary overrides.",
    },
    {
      key: "profile",
      label: "Profile",
      description: "Memory, source preferences, response bias.",
    },
  ];

  return (
    <div className="panel-shortcut-grid" aria-label="Advanced configuration panels">
      {shortcuts.map((shortcut) => (
        <button
          className="panel-shortcut-card"
          key={shortcut.key}
          onClick={() => onOpenPanel(shortcut.key)}
          type="button"
        >
          <strong>{shortcut.label}</strong>
          <span>{shortcut.description}</span>
        </button>
      ))}
    </div>
  );
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

interface ProjectPanelProps extends Pick<RunComposerProps, "publicConfig"> {
  onProjectChange?: (projectId: string | null) => void;
}

export function ProjectPanel({ publicConfig, onProjectChange }: ProjectPanelProps) {
  const queryClient = useQueryClient();
  const apiBaseUrl = useResearchStore((state) => state.apiBaseUrl);
  const selectedProjectId = useResearchStore((state) => state.selectedProjectId);
  const setSelectedProjectId = useResearchStore((state) => state.setSelectedProjectId);
  const stagedRunAssets = useResearchStore((state) => state.stagedRunAssets);
  const addStagedRunAssets = useResearchStore((state) => state.addStagedRunAssets);
  const removeStagedRunAsset = useResearchStore((state) => state.removeStagedRunAsset);

  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDescription, setNewProjectDescription] = useState("");
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
      onProjectChange?.(project.id);
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to create project.");
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

  const projects = projectsQuery.data ?? [];
  const currentProject = projectDetailQuery.data;
  const projectPlanningAssets =
    currentProject?.assets.filter(
      (asset) => asset.usage === "planning_context" && asset.source_type === "file",
    ) ?? [];
  const projectReferenceAssets =
    currentProject?.assets.filter(
      (asset) => asset.usage === "reference_source" && asset.source_type === "file",
    ) ?? [];
  const stagedPlanningAssets = stagedRunAssets.filter(
    (asset) => asset.usage === "planning_context",
  );
  const stagedReferenceAssets = stagedRunAssets.filter(
    (asset) => asset.usage === "reference_source",
  );
  const projectAssetCount = projectPlanningAssets.length + projectReferenceAssets.length;

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
          <span className="pill muted">File</span>
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

  return (
    <div className="project-panel">
      <div className="project-panel-header">
        <div>
          <span className="eyebrow">Project</span>
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
          onClick={() => {
            setSelectedProjectId(null);
            onProjectChange?.(null);
          }}
          type="button"
        >
          <span className="project-item-name">No project</span>
        </button>
        {projects.map((project: ProjectSummary) => (
          <button
            key={project.id}
            className={`project-item ${selectedProjectId === project.id ? "active" : ""}`}
            onClick={() => {
              setSelectedProjectId(project.id);
              onProjectChange?.(project.id);
            }}
            type="button"
          >
            <span className="project-item-name">{project.name}</span>
          </button>
        ))}
      </div>

      <div className="project-summary-pills">
        <span className="pill muted">
          {currentProject?.name ?? "No project"}
        </span>
        {selectedProjectId ? (
          <span className="pill muted">
            {projectAssetCount} corpus files
          </span>
        ) : null}
        <span className="pill muted">
          {stagedRunAssets.length} run files
        </span>
        {limits ? (
          <span className="pill muted">
            {limits.max_files_per_batch} files /{" "}
            {(limits.max_file_size_bytes / (1024 * 1024)).toFixed(0)} MB
          </span>
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
          <span className="eyebrow">Planning</span>
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
            ) : null}
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
          {stagedPlanningAssets.length > 0 ? (
            <ul className="project-asset-list detailed">
              {stagedPlanningAssets.map((asset) => renderStagedAssetCard(asset))}
            </ul>
          ) : null}
        </div>
      </div>

      <div className="project-run-section">
        <div className="project-section-header">
          <span className="eyebrow">References</span>
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
            ) : null}
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
          {stagedReferenceAssets.length > 0 ? (
            <ul className="project-asset-list detailed">
              {stagedReferenceAssets.map((asset) => renderStagedAssetCard(asset))}
            </ul>
          ) : null}
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
  const reportOutputConfig = useResearchStore((state) => state.reportOutputConfig);
  const sourceSelection = useResearchStore((state) => state.sourceSelection);
  const requirePlanApproval = useResearchStore((state) => state.requirePlanApproval);
  const selectedProjectId = useResearchStore((state) => state.selectedProjectId);
  const runInputAssets = useResearchStore((state) => state.runInputAssets);
  const stagedRunAssets = useResearchStore((state) => state.stagedRunAssets);
  const modelConfigOverride = useResearchStore((state) => state.modelConfigOverride);
  const effectiveModels = resolveEffectiveModels(publicConfig, modelConfigOverride);

  return (
    <>
      <div className="config-callout settings-summary-card">
        <dl className="settings-summary-grid">
          <div className="settings-summary-row">
            <dt>Budget</dt>
            <dd>
              {budget.max_streams} streams / {budget.max_queries_per_stream} queries
            </dd>
          </div>
          <div className="settings-summary-row">
            <dt>Report</dt>
            <dd>
              {reportOutputConfig.min_words}-{reportOutputConfig.max_words} words
            </dd>
          </div>
          <div className="settings-summary-row">
            <dt>Workflow</dt>
            <dd>{requirePlanApproval ? "Approval-first" : "Direct run"}</dd>
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
            <dd>{runInputAssets.length + stagedRunAssets.length} attached</dd>
          </div>
          <div className="settings-summary-row settings-summary-row-wide">
            <dt>Models</dt>
            <dd>
              {effectiveModels.lead_model} / {effectiveModels.planner_model} /{" "}
              {effectiveModels.worker_model} / {effectiveModels.verifier_model}
            </dd>
          </div>
        </dl>
      </div>

      <div className="settings-grid">
        <label className="field">
          <span className="field-label">API endpoint</span>
          <input
            className="text-input"
            value={apiBaseUrl}
            onChange={(event) => setApiBaseUrl(event.target.value)}
            placeholder="http://127.0.0.1:8010"
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
      </div>

      {onOpenPanel ? (
        <PanelShortcutGrid onOpenPanel={onOpenPanel} />
      ) : null}
    </>
  );
}

export function RunComposer({ publicConfig, onOpenPanel }: RunComposerProps) {
  return (
    <>
      <ProjectPanel publicConfig={publicConfig} />
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
      </label>

      <label className="field">
        <span className="field-label">Recency policy</span>
        <CustomSelect
          ariaLabel="Recency policy"
          value={agentConfig.recency_policy}
          onChange={(value) => updateAgentConfig({ recency_policy: value as never })}
          options={recencyPolicyOptions}
        />
      </label>

      <label className="field">
        <span className="field-label">Answer style</span>
        <CustomSelect
          ariaLabel="Answer style"
          value={agentConfig.answer_style}
          onChange={(value) => updateAgentConfig({ answer_style: value as never })}
          options={answerStyleOptions}
        />
      </label>

      <label className="field">
        <span className="field-label">Source trust floor</span>
        <CustomSelect
          ariaLabel="Source trust floor"
          value={agentConfig.source_trust_floor}
          onChange={(value) => updateAgentConfig({ source_trust_floor: value as never })}
          options={trustFloorOptions}
        />
      </label>
    </div>
  );
}

export function SettingsDrawerPanel({ publicConfig, onOpenPanel }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <section className="config-modal-section featured">
        <div className="config-section-head">
          <div>
            <p className="drawer-section-label">Recommended</p>
            <h3>Choose a starting point</h3>
          </div>
          <span>One click</span>
        </div>
        <PresetConfigurationCards publicConfig={publicConfig} />
      </section>

      <section className="config-modal-section">
        <div className="config-section-head">
          <div>
            <p className="drawer-section-label">Current setup</p>
            <h3>Active run profile</h3>
          </div>
        </div>
        <CommonSettingsFields publicConfig={publicConfig} onOpenPanel={onOpenPanel} />
      </section>

      <details className="config-modal-disclosure">
        <summary>
          <strong>Research behavior</strong>
          <span>Profiles, recency, answer style, and source trust.</span>
        </summary>
        <BehaviorSettingsFields />
      </details>
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
        </label>

        <label className="field">
          <span className="field-label">Claim granularity</span>
          <CustomSelect
            ariaLabel="Claim granularity"
            value={agentConfig.claim_granularity}
            onChange={(value) => updateAgentConfig({ claim_granularity: value as never })}
            options={claimGranularityOptions}
          />
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
        <span>Surface counterevidence and disagreements.</span>
      </label>
    </>
  );
}

function ProfilePreferenceFields() {
  const profilePreferences = useResearchStore((state) => state.profilePreferences);
  const updateProfilePreferences = useResearchStore(
    (state) => state.updateProfilePreferences,
  );

  return (
    <>
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
  const textModelOptions = useMemo(
    () =>
      uniqueModelOptions([
        publicConfig?.models.lead_model,
        publicConfig?.models.planner_model,
        publicConfig?.models.worker_model,
        publicConfig?.models.verifier_model,
        effectiveModels.lead_model,
        effectiveModels.planner_model,
        effectiveModels.worker_model,
        effectiveModels.verifier_model,
      ]),
    [effectiveModels, publicConfig],
  );
  const embeddingModelOptions = useMemo(
    () =>
      uniqueModelOptions([
        publicConfig?.models.embedding_model,
        effectiveModels.embedding_model,
      ]),
    [effectiveModels.embedding_model, publicConfig?.models.embedding_model],
  );
  const rerankerModelOptions = useMemo(
    () =>
      uniqueModelOptions([
        publicConfig?.models.reranker_model,
        effectiveModels.reranker_model,
      ]),
    [effectiveModels.reranker_model, publicConfig?.models.reranker_model],
  );

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
      <div className="settings-grid">
        <label className="field settings-grid-wide">
          <span className="field-label">Lead model</span>
          <input
            className="text-input"
            list="text-model-options"
            value={effectiveModels.lead_model}
            onChange={handleModelChange("lead_model")}
            placeholder={publicConfig?.models.lead_model ?? "gpt-5.5"}
          />
        </label>

        <label className="field">
          <span className="field-label">Planner model</span>
          <input
            className="text-input"
            list="text-model-options"
            value={effectiveModels.planner_model}
            onChange={handleModelChange("planner_model")}
            placeholder={publicConfig?.models.planner_model ?? "gpt-5.5"}
          />
        </label>

        <label className="field">
          <span className="field-label">Worker model</span>
          <input
            className="text-input"
            list="text-model-options"
            value={effectiveModels.worker_model}
            onChange={handleModelChange("worker_model")}
            placeholder={publicConfig?.models.worker_model ?? "gpt-5.5"}
          />
        </label>

        <label className="field">
          <span className="field-label">Verifier model</span>
          <input
            className="text-input"
            list="text-model-options"
            value={effectiveModels.verifier_model}
            onChange={handleModelChange("verifier_model")}
            placeholder={publicConfig?.models.verifier_model ?? "gpt-5.5"}
          />
        </label>

        <label className="field">
          <span className="field-label">Embedding model</span>
          <input
            className="text-input"
            list="embedding-model-options"
            value={effectiveModels.embedding_model}
            onChange={handleModelChange("embedding_model")}
            placeholder={publicConfig?.models.embedding_model ?? "text-embedding-3-large"}
          />
          <span className="field-hint">Vectors for semantic retrieval.</span>
        </label>

        <label className="field">
          <span className="field-label">Reranker model</span>
          <input
            className="text-input"
            list="reranker-model-options"
            value={effectiveModels.reranker_model}
            onChange={handleModelChange("reranker_model")}
            placeholder={publicConfig?.models.reranker_model ?? "BAAI/bge-reranker-v2-m3"}
          />
          <span className="field-hint">Reorders passages before grounding.</span>
        </label>
      </div>
      <datalist id="text-model-options">
        {textModelOptions.map((model) => (
          <option key={model} value={model} />
        ))}
      </datalist>
      <datalist id="embedding-model-options">
        {embeddingModelOptions.map((model) => (
          <option key={model} value={model} />
        ))}
      </datalist>
      <datalist id="reranker-model-options">
        {rerankerModelOptions.map((model) => (
          <option key={model} value={model} />
        ))}
      </datalist>

      <div className="button-row">
        <button className="secondary-button" onClick={resetModelConfigOverride} type="button">
          Reset models
        </button>
      </div>
    </>
  );
}

function ExecutionWorkflowFields({ publicConfig }: RunComposerProps) {
  const clarifierConfig = useResearchStore((state) => state.clarifierConfig);
  const updateClarifierConfig = useResearchStore((state) => state.updateClarifierConfig);
  const requirePlanApproval = useResearchStore((state) => state.requirePlanApproval);
  const setRequirePlanApproval = useResearchStore((state) => state.setRequirePlanApproval);
  const asyncSubmit = useResearchStore((state) => state.asyncSubmit);
  const setAsyncSubmit = useResearchStore((state) => state.setAsyncSubmit);

  return (
    <>
      <div className="settings-grid">
        <div className="field static-field">
          <span className="field-label">Workflow</span>
          <span className="field-static-value">
            {requirePlanApproval ? "Approval-first" : "Direct run"}
          </span>
        </div>

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
            checked={requirePlanApproval}
            onChange={(event) => setRequirePlanApproval(event.target.checked)}
          />
          <span>Require plan approval</span>
        </label>

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
          <span>Always clarify before planning</span>
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
        <p className="muted-text">Plan approval is disabled on this backend.</p>
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

function ToolCatalogFields({ publicConfig }: RunComposerProps) {
  const tools = publicConfig?.tool_catalog ?? [];
  if (!tools.length) {
    return null;
  }

  return (
    <div className="tool-catalog-grid">
      {tools.map((tool) => (
        <article className={`tool-catalog-card ${tool.enabled ? "enabled" : "disabled"}`} key={tool.name}>
          <div>
            <strong>{tool.display_name}</strong>
            <span>{tool.category} · {tool.owner}</span>
          </div>
          <div className="ops-badges">
            <span className="pill muted">{tool.enabled ? "enabled" : "disabled"}</span>
            {tool.provider ? <span className="pill muted">{tool.provider}</span> : null}
            {tool.per_run_limit ? <span className="pill muted">{tool.per_run_limit}/run</span> : null}
          </div>
        </article>
      ))}
    </div>
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
        <p className="muted-text">Profile memory is disabled on this backend.</p>
      ) : null}

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
              {memoryLimits?.[field.key]?.min ?? field.min}-
              {memoryLimits?.[field.key]?.max ?? field.max}
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
            {field.description} {budgetLimits?.[field.key]?.min ?? field.min}-
            {budgetLimits?.[field.key]?.max ?? field.max}.
          </span>
        </label>
      ))}
    </div>
  );
}

function ReportOutputFields({ publicConfig }: RunComposerProps) {
  const reportOutputConfig = useResearchStore((state) => state.reportOutputConfig);
  const updateReportOutputConfig = useResearchStore(
    (state) => state.updateReportOutputConfig,
  );
  const completionGate =
    publicConfig?.capabilities.custom_responses_contract?.completion_gate;

  return (
    <>
      <div className="settings-grid">
        <label className="field">
          <span className="field-label">Minimum words</span>
          <input
            className="text-input"
            type="number"
            min={100}
            max={8000}
            step={100}
            value={reportOutputConfig.min_words}
            onChange={(event) =>
              updateReportOutputConfig({ min_words: Number(event.target.value) })
            }
          />
          <span className="field-hint">Target floor for the final report draft.</span>
        </label>

        <label className="field">
          <span className="field-label">Maximum words</span>
          <input
            className="text-input"
            type="number"
            min={reportOutputConfig.min_words}
            max={12000}
            step={100}
            value={reportOutputConfig.max_words}
            onChange={(event) =>
              updateReportOutputConfig({ max_words: Number(event.target.value) })
            }
          />
          <span className="field-hint">Target ceiling before citations and source notes.</span>
        </label>
      </div>
      {completionGate ? (
        <p className="muted-text">
          Completion gate: {completionGate.min_chars} chars, {completionGate.min_headings} headings,
          {` ${completionGate.max_attempts}`} attempts.
        </p>
      ) : null}
    </>
  );
}

export function WorkflowDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <details className="config-modal-disclosure" open>
        <summary>
          <strong>Workflow</strong>
          <span>Approval, clarifier, and async execution behavior.</span>
        </summary>
        <ExecutionWorkflowFields publicConfig={publicConfig} />
      </details>
      <details className="config-modal-disclosure">
        <summary>
          <strong>Output policy</strong>
          <span>Citation discipline, claim granularity, and disagreements.</span>
        </summary>
        <AgentPolicyFields />
      </details>
    </div>
  );
}

export function SourcesDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <section className="config-modal-section">
        <div className="config-section-head">
          <div>
            <p className="drawer-section-label">Source defaults</p>
            <h3>Search surfaces</h3>
          </div>
        </div>
        <SourceSelectionFields publicConfig={publicConfig} />
      </section>
      <details className="config-modal-disclosure">
        <summary>
          <strong>Tool registry</strong>
          <span>Configured source tools and per-run limits.</span>
        </summary>
        <ToolCatalogFields publicConfig={publicConfig} />
      </details>
    </div>
  );
}

export function BudgetDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <section className="config-modal-section featured">
        <div className="config-section-head">
          <div>
            <p className="drawer-section-label">Presets</p>
            <h3>Run depth without tuning</h3>
          </div>
          <span>Recommended first</span>
        </div>
        <PresetConfigurationCards publicConfig={publicConfig} />
      </section>
      <details className="config-modal-disclosure" open>
        <summary>
          <strong>Research depth</strong>
          <span>Streams, replans, search results, and source caps.</span>
        </summary>
        <BudgetFields publicConfig={publicConfig} />
      </details>
      <details className="config-modal-disclosure" open>
        <summary>
          <strong>Report length</strong>
          <span>Target floor and ceiling for the final report.</span>
        </summary>
        <ReportOutputFields publicConfig={publicConfig} />
      </details>
    </div>
  );
}

export function ModelsDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <section className="config-modal-section">
        <div className="config-section-head">
          <div>
            <p className="drawer-section-label">Model routing</p>
            <h3>Use deployment defaults unless needed</h3>
          </div>
        </div>
        <ModelSelectionFields publicConfig={publicConfig} />
      </section>
    </div>
  );
}

export function ProfileDrawerPanel({ publicConfig }: RunComposerProps) {
  return (
    <div className="drawer-section-stack">
      <details className="config-modal-disclosure" open>
        <summary>
          <strong>Preferences</strong>
          <span>Preferred sources, avoided sources, and style bias.</span>
        </summary>
        <ProfilePreferenceFields />
      </details>
      <details className="config-modal-disclosure">
        <summary>
          <strong>Memory harness</strong>
          <span>Context packing limits and where memory can influence a run.</span>
        </summary>
        <MemoryHarnessFields publicConfig={publicConfig} />
      </details>
    </div>
  );
}
