"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  answerClarification,
  approvePlan,
  cancelRun,
  createJob,
  createRun,
  fetchProfilePreferences,
  fetchProjects,
  fetchPublicConfig,
  fetchRunDetail,
  fetchRunMessages,
  fetchRunWorkspace,
  fetchRuns,
  rejectPlan,
  requestPlanChanges,
  resumeRun,
  retryRun,
  sendRunMessage,
  updateProfilePreferences,
} from "@/lib/api";
import type { RunConversationMessage, RunDetail, RunEvent } from "@/lib/types";
import { useRunStream } from "@/hooks/use-run-stream";
import { useResearchStore } from "@/store/use-research-store";

import {
  type FocusedDrawerKey,
  BudgetDrawerPanel,
  ModelsDrawerPanel,
  ProjectPanel,
  ProfileDrawerPanel,
  SettingsDrawerPanel,
  SourcesDrawerPanel,
  WorkflowDrawerPanel,
} from "./run-composer";
import { RunHistory } from "./run-history";
import { RunWorkspace } from "./run-workspace";

type DrawerKey = "project" | "settings" | FocusedDrawerKey;

const DRAWER_META: Record<
  DrawerKey,
  { eyebrow: string; title: string; description: string }
> = {
  project: {
    eyebrow: "Context",
    title: "Project context",
    description:
      "Attach a project corpus, manage its planning and reference assets, and stage inputs for this run.",
  },
  settings: {
    eyebrow: "Settings",
    title: "Agent behavior",
    description:
      "Tune behavior, policy, and connection plumbing in one place. Deep panels are one click away.",
  },
  workflow: {
    eyebrow: "Workflow",
    title: "Execution and policy",
    description:
      "Pick the execution mode, toggle human-in-the-loop approval, and shape the output policy.",
  },
  sources: {
    eyebrow: "Sources",
    title: "Source registry",
    description:
      "Choose which source presets the agent may use. Unselected presets fall back to the deployment default.",
  },
  budget: {
    eyebrow: "Budget",
    title: "Budget controls",
    description:
      "Cap streams, queries, and provider spend. The planner respects these as hard ceilings.",
  },
  models: {
    eyebrow: "Models",
    title: "Model selection",
    description:
      "Override the lead, planner, and reviewer models for this run. Leave blank to use the deployment defaults.",
  },
  profile: {
    eyebrow: "Profile",
    title: "Profile and memory",
    description:
      "Shape how prior runs and preferences condition the agent's planning and grounding behavior.",
  },
};

function formatLabel(value: string | null | undefined): string {
  if (!value) return "Not configured";
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function isActive(status: RunDetail["status"] | undefined): boolean {
  return (
    status === "queued" ||
    status === "clarifying" ||
    status === "awaiting_plan_approval" ||
    status === "planning" ||
    status === "researching" ||
    status === "grounding"
  );
}

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">("light");

  useEffect(() => {
    const stored = localStorage.getItem("or-theme");
    const resolved = stored === "dark" || stored === "light" ? stored : "light";
    setTheme(resolved);
    document.documentElement.setAttribute("data-theme", resolved);
    localStorage.setItem("or-theme", resolved);
  }, []);

  const toggle = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      localStorage.setItem("or-theme", next);
      document.documentElement.setAttribute("data-theme", next);
      return next;
    });
  }, []);

  return { theme, toggle };
}

export function ResearchDashboard() {
  const queryClient = useQueryClient();
  const { theme, toggle: toggleTheme } = useTheme();
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [activeDrawer, setActiveDrawer] = useState<DrawerKey | null>(null);
  const centerScrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pendingQuestionRef = useRef<string>("");

  const autoResizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);
  const apiBaseUrl = useResearchStore((state) => state.apiBaseUrl);
  const selectedRunId = useResearchStore((state) => state.selectedRunId);
  const setSelectedRunId = useResearchStore((state) => state.setSelectedRunId);
  const profileId = useResearchStore((state) => state.profileId);
  const profilePreferences = useResearchStore((state) => state.profilePreferences);
  const memoryPolicyOverride = useResearchStore((state) => state.memoryPolicyOverride);
  const executionMode = useResearchStore((state) => state.executionMode);
  const requirePlanApproval = useResearchStore((state) => state.requirePlanApproval);
  const clarifierConfig = useResearchStore((state) => state.clarifierConfig);
  const sourceSelection = useResearchStore((state) => state.sourceSelection);
  const selectedProjectId = useResearchStore((state) => state.selectedProjectId);
  const setSelectedProjectId = useResearchStore((state) => state.setSelectedProjectId);
  const runInputAssets = useResearchStore((state) => state.runInputAssets);
  const stagedRunAssets = useResearchStore((state) => state.stagedRunAssets);
  const clearRunInputAssets = useResearchStore((state) => state.clearRunInputAssets);
  const clearStagedRunAssets = useResearchStore((state) => state.clearStagedRunAssets);
  const asyncSubmit = useResearchStore((state) => state.asyncSubmit);
  const updateProfilePreferencesInStore = useResearchStore((state) => state.updateProfilePreferences);
  const questionDraft = useResearchStore((state) => state.questionDraft);
  const setQuestionDraft = useResearchStore((state) => state.setQuestionDraft);
  const budget = useResearchStore((state) => state.budget);
  const agentConfig = useResearchStore((state) => state.agentConfig);
  const modelConfigOverride = useResearchStore((state) => state.modelConfigOverride);
  const hydrateFromServerConfig = useResearchStore((state) => state.hydrateFromServerConfig);
  const replaceRunEvents = useResearchStore((state) => state.replaceRunEvents);
  const setRunWorkspace = useResearchStore((state) => state.setRunWorkspace);
  const cachedWorkspace = useResearchStore((state) =>
    selectedRunId ? state.runWorkspaces[selectedRunId] : undefined,
  );
  const streamState = useResearchStore((state) =>
    selectedRunId ? state.runStreams[selectedRunId] : undefined,
  );

  const publicConfigQuery = useQuery({
    queryKey: ["public-config", apiBaseUrl],
    queryFn: () => fetchPublicConfig(apiBaseUrl),
  });

  const profileQuery = useQuery({
    queryKey: ["profile-preferences", apiBaseUrl, profileId],
    queryFn: () => fetchProfilePreferences(apiBaseUrl, profileId),
    enabled: Boolean(profileId),
  });

  const runsQuery = useQuery({
    queryKey: ["runs", apiBaseUrl],
    queryFn: () => fetchRuns(apiBaseUrl),
    refetchInterval: 5_000,
  });

  const projectsQuery = useQuery({
    queryKey: ["projects", apiBaseUrl],
    queryFn: () => fetchProjects(apiBaseUrl),
  });

  const detailQuery = useQuery({
    queryKey: ["run-detail", apiBaseUrl, selectedRunId],
    queryFn: () => fetchRunDetail(apiBaseUrl, selectedRunId as string),
    enabled: Boolean(selectedRunId),
    refetchInterval: (query) => (isActive(query.state.data?.status) ? 2_500 : false),
  });

  const workspaceQuery = useQuery({
    queryKey: ["run-workspace", apiBaseUrl, selectedRunId],
    queryFn: () => fetchRunWorkspace(apiBaseUrl, selectedRunId as string),
    enabled: Boolean(selectedRunId),
    refetchInterval: (query) => (isActive(query.state.data?.status) ? 2_500 : false),
  });

  const messagesQuery = useQuery({
    queryKey: ["run-messages", apiBaseUrl, selectedRunId],
    queryFn: () => fetchRunMessages(apiBaseUrl, selectedRunId as string),
    enabled: Boolean(selectedRunId),
  });

  useEffect(() => {
    if (publicConfigQuery.data) {
      hydrateFromServerConfig(publicConfigQuery.data);
    }
  }, [hydrateFromServerConfig, publicConfigQuery.data]);

  useEffect(() => {
    if (!activeDrawer) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setActiveDrawer(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeDrawer]);

  useEffect(() => {
    if (profileQuery.data) {
      updateProfilePreferencesInStore(profileQuery.data.preferences);
    }
  }, [profileQuery.data, updateProfilePreferencesInStore]);

  useEffect(() => {
    if (selectedRunId && detailQuery.data?.events?.length) {
      replaceRunEvents(selectedRunId, detailQuery.data.events);
    }
  }, [detailQuery.data?.events, replaceRunEvents, selectedRunId]);

  useEffect(() => {
    if (selectedRunId && workspaceQuery.data) {
      setRunWorkspace(selectedRunId, workspaceQuery.data);
    }
  }, [selectedRunId, setRunWorkspace, workspaceQuery.data]);

  useRunStream({
    runId: selectedRunId,
    apiBaseUrl,
  });

  const createRunMutation = useMutation({
    mutationFn: async (question: string) => {
      await updateProfilePreferences(apiBaseUrl, profileId, profilePreferences);
      const payload = {
        question,
        budget,
        agent_config: agentConfig,
        model_config_override:
          Object.keys(modelConfigOverride).length > 0 ? modelConfigOverride : null,
        profile_id: profileId,
        project_id: selectedProjectId,
        memory_policy_override: memoryPolicyOverride,
        execution_mode: executionMode,
        require_plan_approval: requirePlanApproval,
        clarifier_config: clarifierConfig,
        source_selection: sourceSelection,
        input_assets: runInputAssets,
        staged_asset_ids: stagedRunAssets.map((asset) => asset.id),
        async_submit: asyncSubmit,
        metadata: {
          client: {
            surface: "nextjs-dashboard",
            build: "2026-04-12.2",
          },
        },
      };
      if (asyncSubmit) {
        const job = await createJob(apiBaseUrl, payload);
        return { id: job.run_id };
      }
      return createRun(apiBaseUrl, payload);
    },
    onMutate: async (question) => {
      pendingQuestionRef.current = question;
      setSubmitError(null);
      setQuestionDraft("");
      if (textareaRef.current) textareaRef.current.style.height = "";
    },
    onSuccess: (run) => {
      setSelectedRunId(run.id);
      clearRunInputAssets();
      clearStagedRunAssets();
      pendingQuestionRef.current = "";
      void queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      void queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, run.id] });
      void queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, run.id] });
    },
    onError: (error) => {
      setSubmitError(error instanceof Error ? error.message : "Failed to start run.");
      setQuestionDraft(pendingQuestionRef.current);
      pendingQuestionRef.current = "";
      requestAnimationFrame(() => autoResizeTextarea());
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (runId: string) => cancelRun(apiBaseUrl, runId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      await queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, selectedRunId] });
      await queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, selectedRunId] });
    },
  });

  const resumeMutation = useMutation({
    mutationFn: (runId: string) => resumeRun(apiBaseUrl, runId),
    onSuccess: async (run) => {
      setSelectedRunId(run.id);
      await queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      await queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, run.id] });
      await queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, run.id] });
    },
  });

  const retryMutation = useMutation({
    mutationFn: (runId: string) => retryRun(apiBaseUrl, runId),
    onSuccess: async (run) => {
      setSelectedRunId(run.id);
      await queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      await queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, run.id] });
    },
  });

  const clarificationMutation = useMutation({
    mutationFn: ({ runId, response }: { runId: string; response: string }) =>
      answerClarification(apiBaseUrl, runId, response),
    onSuccess: async (detail) => {
      setSelectedRunId(detail.id);
      await queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      await queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, detail.id] });
      await queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, detail.id] });
    },
  });

  const approveMutation = useMutation({
    mutationFn: ({ runId, note }: { runId: string; note?: string }) =>
      approvePlan(apiBaseUrl, runId, note),
    onSuccess: async (detail) => {
      setSelectedRunId(detail.id);
      await queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      await queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, detail.id] });
      await queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, detail.id] });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: ({ runId, note }: { runId: string; note?: string }) =>
      rejectPlan(apiBaseUrl, runId, note),
    onSuccess: async (detail) => {
      setSelectedRunId(detail.id);
      await queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      await queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, detail.id] });
      await queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, detail.id] });
    },
  });

  const requestChangesMutation = useMutation({
    mutationFn: ({ runId, note }: { runId: string; note?: string }) =>
      requestPlanChanges(apiBaseUrl, runId, note),
    onSuccess: async (detail) => {
      setSelectedRunId(detail.id);
      await queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      await queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, detail.id] });
      await queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, detail.id] });
    },
  });

  const chatMutation = useMutation({
    mutationFn: ({ runId, message }: { runId: string; message: string }) =>
      sendRunMessage(apiBaseUrl, runId, message),
    onSuccess: async (reply) => {
      queryClient.setQueryData<RunConversationMessage[]>(
        ["run-messages", apiBaseUrl, reply.user_message.run_id],
        (current) => [
          ...(current ?? []),
          reply.user_message,
          reply.assistant_message,
        ],
      );
      await queryClient.invalidateQueries({
        queryKey: ["run-detail", apiBaseUrl, reply.user_message.run_id],
      });
    },
  });

  const activeDetail = detailQuery.data;
  const activeWorkspace = workspaceQuery.data ?? cachedWorkspace;
  const visibleRuns = useMemo(
    () =>
      selectedProjectId
        ? (runsQuery.data ?? []).filter((run) => run.project_id === selectedProjectId)
        : (runsQuery.data ?? []),
    [runsQuery.data, selectedProjectId],
  );
  const eventFeed = useMemo<RunEvent[]>(
    () => streamState?.events ?? activeDetail?.events ?? [],
    [activeDetail?.events, streamState?.events],
  );
  const conversationMessages = messagesQuery.data ?? activeDetail?.conversation_messages ?? [];
  const displayConnectionState =
    activeWorkspace &&
    (activeWorkspace.status === "completed" ||
      activeWorkspace.status === "failed" ||
      activeWorkspace.status === "cancelled") &&
    (streamState?.connectionState ?? "idle") === "error"
      ? "terminal"
      : streamState?.connectionState ?? "idle";
  const leadModel =
    modelConfigOverride.lead_model || publicConfigQuery.data?.models.lead_model || "Default";
  const plannerModel =
    modelConfigOverride.planner_model ||
    publicConfigQuery.data?.models.planner_model ||
    "Default";
  const setupSummary = [
    {
      label: "Project",
      value: selectedProjectId ? "Attached" : "Standalone run",
    },
    {
      label: "Assets",
      value: `${runInputAssets.length + stagedRunAssets.length} attached`,
    },
    {
      label: "Workflow",
      value: "Approval-first research",
    },
    {
      label: "Sources",
      value:
        sourceSelection.length > 0
          ? `${sourceSelection.length} selected`
          : "Deployment default",
    },
    {
      label: "Budget",
      value: `${budget.max_streams} streams`,
    },
    {
      label: "Models",
      value: `${leadModel} / ${plannerModel}`,
    },
  ];
  const runStatusLabel = selectedRunId
    ? formatLabel(activeDetail?.status ?? displayConnectionState)
    : "Ready";

  const handleSubmit = () => {
    const trimmedQuestion = questionDraft.trim();
    if (trimmedQuestion.length >= 12 && !createRunMutation.isPending) {
      createRunMutation.mutate(trimmedQuestion);
    }
  };

  const handleNewChat = () => {
    setSelectedRunId(null);
    setSubmitError(null);
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
      centerScrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    });
  };

  const handleSelectProject = (projectId: string | null) => {
    setSelectedProjectId(projectId);
    setSelectedRunId(null);
  };

  const projects = projectsQuery.data ?? [];
  const activeProject = projects.find((project) => project.id === selectedProjectId) ?? null;
  const isLanding = !selectedRunId;
  const suggestionChips = [
    "Compare the economic impact of universal basic income pilots since 2020.",
    "Survey recent advances in liquid neural networks and their applications.",
    "Draft a grounded literature review on microplastics in freshwater systems.",
    "Summarize the state of direct air capture economics in 2025–2026.",
  ];

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <main className="dashboard-page">
      <header className="hero">
        <div className="hero-brand">
          <div>
            <p className="eyebrow">Open Research</p>
            <h1 className="hero-title">Console</h1>
          </div>
        </div>
        <div className="hero-badges">
          <span
            className={`pill ${
              activeDetail?.status
                ? `status-${activeDetail.status}`
                : "muted"
            }`}
          >
            <span
              className={`status-dot status-${
                activeDetail?.status ?? (selectedRunId ? "queued" : "completed")
              }`}
            />
            {runStatusLabel}
          </span>
          <span className="pill muted">
            {publicConfigQuery.data?.backends.workflow ?? "local"}
          </span>
          <span className="pill muted">
            {visibleRuns.length} run{visibleRuns.length === 1 ? "" : "s"}
          </span>
        </div>
        <div className="hero-actions">
          <button
            className={`config-action ${activeDrawer === "project" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "project" ? null : "project"))
            }
            type="button"
          >
            Project
          </button>
          <button
            className={`config-action ${activeDrawer === "settings" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "settings" ? null : "settings"))
            }
            type="button"
          >
            Customize
          </button>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            type="button"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </header>

      <div className="dashboard-grid">
        <aside className="left-column">
          <div className="left-column-shell">
            <div className="rail-new-chat">
              <button
                className="new-chat-button"
                onClick={handleNewChat}
                type="button"
                disabled={isLanding}
              >
                <span className="new-chat-plus" aria-hidden>+</span>
                New chat
              </button>
            </div>

            <section className="rail-projects">
              <div className="rail-section-head">
                <p className="eyebrow">Projects</p>
                <button
                  className="rail-section-action"
                  onClick={() => setActiveDrawer("project")}
                  type="button"
                >
                  Manage
                </button>
              </div>
              <div className="rail-project-list">
                <button
                  className={`rail-project-chip ${activeProject === null ? "active" : ""}`}
                  onClick={() => handleSelectProject(null)}
                  type="button"
                >
                  <span className="rail-project-name">All runs</span>
                  <span className="rail-project-count">
                    {(runsQuery.data ?? []).length}
                  </span>
                </button>
                {projects.slice(0, 6).map((project) => {
                  const count = (runsQuery.data ?? []).filter(
                    (run) => run.project_id === project.id,
                  ).length;
                  return (
                    <button
                      className={`rail-project-chip ${
                        activeProject?.id === project.id ? "active" : ""
                      }`}
                      key={project.id}
                      onClick={() => handleSelectProject(project.id)}
                      type="button"
                      title={project.description ?? undefined}
                    >
                      <span className="rail-project-name">{project.name}</span>
                      <span className="rail-project-count">{count}</span>
                    </button>
                  );
                })}
              </div>
            </section>

            <RunHistory
              runs={visibleRuns}
              selectedRunId={selectedRunId}
              onSelect={setSelectedRunId}
            />

            <div className="rail-setup-actions">
              <p className="eyebrow">Setup</p>
              <div className="rail-setup-buttons">
                <button
                  className="rail-setup-button"
                  onClick={() => setActiveDrawer("workflow")}
                  type="button"
                >
                  Workflow
                </button>
                <button
                  className="rail-setup-button"
                  onClick={() => setActiveDrawer("sources")}
                  type="button"
                >
                  Sources
                </button>
                <button
                  className="rail-setup-button"
                  onClick={() => setActiveDrawer("budget")}
                  type="button"
                >
                  Budget
                </button>
                <button
                  className="rail-setup-button"
                  onClick={() => setActiveDrawer("models")}
                  type="button"
                >
                  Models
                </button>
                <button
                  className="rail-setup-button"
                  onClick={() => setActiveDrawer("profile")}
                  type="button"
                >
                  Profile
                </button>
              </div>
            </div>
          </div>
        </aside>

        <section
          className={`center-column workspace-column ${
            isLanding ? "is-landing" : ""
          }`}
        >
          {isLanding ? null : (
            <div className="center-scroll" ref={centerScrollRef}>
              <RunWorkspace
                workspace={activeWorkspace}
                rawEvents={eventFeed}
                conversationMessages={conversationMessages}
                connectionState={displayConnectionState}
                connectionMode={streamState?.lastMode}
                onCancel={(runId) => cancelMutation.mutate(runId)}
                onResume={(runId) => resumeMutation.mutate(runId)}
                onRetry={(runId) => retryMutation.mutate(runId)}
                onAnswerClarification={(runId, response) =>
                  clarificationMutation.mutate({ runId, response })
                }
                onApprove={(runId, note) => approveMutation.mutate({ runId, note })}
                onReject={(runId, note) => rejectMutation.mutate({ runId, note })}
                onRequestChanges={(runId, note) =>
                  requestChangesMutation.mutate({ runId, note })
                }
                onSendMessage={(runId, message) => chatMutation.mutate({ runId, message })}
                pending={{
                  cancel: cancelMutation.isPending,
                  resume: resumeMutation.isPending,
                  retry: retryMutation.isPending,
                  clarification: clarificationMutation.isPending,
                  approve: approveMutation.isPending,
                  reject: rejectMutation.isPending,
                  requestChanges: requestChangesMutation.isPending,
                  chat: chatMutation.isPending,
                }}
              />
            </div>
          )}

          {isLanding ? (
            <div className="landing-hero">
              <div className="landing-hero-inner">
                <p className="eyebrow">Open Research</p>
                <h2 className="landing-title">
                  What should we dig into?
                </h2>
                <p className="landing-copy">
                  Ask a research question. The agent plans, runs parallel search
                  streams, grounds every claim, and delivers a cited report.
                  {activeProject
                    ? ` Attached to project ${activeProject.name}.`
                    : ""}
                </p>
                <div className="landing-setup">
                  {setupSummary.slice(0, 4).map((item) => (
                    <article className="landing-setup-item" key={item.label}>
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </article>
                  ))}
                </div>
                <div className="landing-suggestions">
                  <p className="landing-suggestions-head">Try a starter</p>
                  <div className="landing-suggestion-grid">
                    {suggestionChips.map((chip) => (
                      <button
                        className="landing-suggestion"
                        key={chip}
                        onClick={() => {
                          setQuestionDraft(chip);
                          requestAnimationFrame(() => {
                            autoResizeTextarea();
                            textareaRef.current?.focus();
                          });
                        }}
                        type="button"
                      >
                        {chip}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          <div className="center-input">
            <div className="center-input-shell">
              <div className="center-input-meta">
                <span>{selectedProjectId ? "Project attached" : "Fresh run"}</span>
                <span>
                  {runInputAssets.length + stagedRunAssets.length} input asset
                  {runInputAssets.length + stagedRunAssets.length === 1 ? "" : "s"}
                </span>
                <span>
                  {sourceSelection.length > 0
                    ? `${sourceSelection.length} source presets`
                    : "Default source registry"}
                </span>
              </div>
              <div className="center-input-row">
                <textarea
                  className="textarea-input"
                  ref={textareaRef}
                  value={questionDraft}
                  onChange={(e) => {
                    setQuestionDraft(e.target.value);
                    autoResizeTextarea();
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a research question... (Enter to submit, Shift+Enter for newline)"
                  rows={1}
                />
                <button
                  className="primary-button"
                  disabled={
                    questionDraft.trim().length < 12 || createRunMutation.isPending
                  }
                  onClick={handleSubmit}
                  type="button"
                >
                  {createRunMutation.isPending ? "..." : "Run"}
                </button>
              </div>
            </div>
          </div>
          {submitError ? (
            <div style={{ padding: "0 16px 6px" }}>
              <p className="error-text">{submitError}</p>
            </div>
          ) : null}
        </section>
      </div>

      {activeDrawer ? (
        <>
          <div
            aria-hidden
            className="drawer-backdrop"
            onClick={() => setActiveDrawer(null)}
          />
          <aside className="drawer" role="dialog" aria-label={DRAWER_META[activeDrawer].title}>
            <div className="drawer-header">
              <div>
                <p className="eyebrow">{DRAWER_META[activeDrawer].eyebrow}</p>
                <h2 className="panel-title">{DRAWER_META[activeDrawer].title}</h2>
                <p className="drawer-lead">{DRAWER_META[activeDrawer].description}</p>
              </div>
              <button
                className="drawer-close"
                onClick={() => setActiveDrawer(null)}
                type="button"
                aria-label="Close drawer"
              >
                Close
              </button>
            </div>
            <div className="drawer-body">
              {activeDrawer === "project" ? (
                <ProjectPanel publicConfig={publicConfigQuery.data} />
              ) : null}
              {activeDrawer === "settings" ? (
                <SettingsDrawerPanel
                  publicConfig={publicConfigQuery.data}
                  onOpenPanel={(panel) => setActiveDrawer(panel)}
                />
              ) : null}
              {activeDrawer === "workflow" ? (
                <WorkflowDrawerPanel publicConfig={publicConfigQuery.data} />
              ) : null}
              {activeDrawer === "sources" ? (
                <SourcesDrawerPanel publicConfig={publicConfigQuery.data} />
              ) : null}
              {activeDrawer === "budget" ? (
                <BudgetDrawerPanel publicConfig={publicConfigQuery.data} />
              ) : null}
              {activeDrawer === "models" ? (
                <ModelsDrawerPanel publicConfig={publicConfigQuery.data} />
              ) : null}
              {activeDrawer === "profile" ? (
                <ProfileDrawerPanel publicConfig={publicConfigQuery.data} />
              ) : null}
            </div>
          </aside>
        </>
      ) : null}
    </main>
  );
}
