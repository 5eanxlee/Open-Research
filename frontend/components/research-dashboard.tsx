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
    description: "Projects and run inputs.",
  },
  settings: {
    eyebrow: "Settings",
    title: "Agent behavior",
    description: "Behavior and connection.",
  },
  workflow: {
    eyebrow: "Workflow",
    title: "Execution and policy",
    description: "Execution mode and approval.",
  },
  sources: {
    eyebrow: "Sources",
    title: "Source registry",
    description: "Search, fetch, and tools.",
  },
  budget: {
    eyebrow: "Configuration",
    title: "Run configuration",
    description: "Depth, query, source, and report targets.",
  },
  models: {
    eyebrow: "Models",
    title: "Model selection",
    description: "Model overrides.",
  },
  profile: {
    eyebrow: "Profile",
    title: "Profile and memory",
    description: "Memory and preferences.",
  },
};

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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
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
  const reportOutputConfig = useResearchStore((state) => state.reportOutputConfig);
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
    const stored = localStorage.getItem("or-sidebar-collapsed");
    if (stored === "true" || stored === "false") {
      setSidebarCollapsed(stored === "true");
      return;
    }
    setSidebarCollapsed(window.innerWidth < 900);
  }, []);

  useEffect(() => {
    localStorage.setItem("or-sidebar-collapsed", String(sidebarCollapsed));
  }, [sidebarCollapsed]);

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
          output_contract: {
            report_min_words: reportOutputConfig.min_words,
            report_max_words: reportOutputConfig.max_words,
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
    onMutate: async ({ message }) => {
      pendingQuestionRef.current = message;
      setSubmitError(null);
      setQuestionDraft("");
      if (textareaRef.current) textareaRef.current.style.height = "";
    },
    onSuccess: async (reply) => {
      pendingQuestionRef.current = "";
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
    onError: (error) => {
      setSubmitError(error instanceof Error ? error.message : "Failed to send message.");
      setQuestionDraft(pendingQuestionRef.current);
      pendingQuestionRef.current = "";
      requestAnimationFrame(() => autoResizeTextarea());
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
  const activeRunStatus = activeWorkspace?.status ?? activeDetail?.status;
  const hasCompletedReport = Boolean(
    activeWorkspace?.final_report_markdown ?? activeDetail?.final_report,
  );
  const composerMode =
    selectedRunId && activeRunStatus === "completed" && hasCompletedReport
      ? "message"
      : "research";
  const displayConnectionState =
    activeWorkspace &&
    (activeWorkspace.status === "completed" ||
      activeWorkspace.status === "failed" ||
      activeWorkspace.status === "cancelled") &&
    (streamState?.connectionState ?? "idle") === "error"
      ? "terminal"
      : streamState?.connectionState ?? "idle";
  const handleSubmit = () => {
    const trimmedInput = questionDraft.trim();
    if (composerMode === "message") {
      if (trimmedInput && selectedRunId && !chatMutation.isPending) {
        chatMutation.mutate({ runId: selectedRunId, message: trimmedInput });
      }
      return;
    }
    if (trimmedInput.length >= 12 && !createRunMutation.isPending) {
      createRunMutation.mutate(trimmedInput);
    }
  };

  const handleNewChat = () => {
    setSelectedRunId(null);
    setSubmitError(null);
    setQuestionDraft("");
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
  const composerDisabled =
    composerMode === "message"
      ? !questionDraft.trim() || chatMutation.isPending
      : questionDraft.trim().length < 12 || createRunMutation.isPending;
  const composerPlaceholder = composerMode === "message" ? "Message" : "Research topic";
  const composerButtonLabel =
    composerMode === "message"
      ? chatMutation.isPending
        ? "..."
        : "Send"
      : createRunMutation.isPending
        ? "..."
        : "Run";

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <main className="dashboard-page">
      <header className="hero">
        <button
          className="sidebar-toggle"
          type="button"
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!sidebarCollapsed}
          onClick={() => setSidebarCollapsed((value) => !value)}
        >
          <span aria-hidden>☰</span>
        </button>
        <div className="hero-brand">
          <h1 className="hero-title">Open Research</h1>
        </div>
        <button className="top-new-chat-button" onClick={handleNewChat} type="button">
          <span aria-hidden>+</span>
          New Chat
        </button>
        <div className="hero-spacer" />
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
            className={`config-action ${activeDrawer === "sources" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "sources" ? null : "sources"))
            }
            type="button"
          >
            Sources
          </button>
          <button
            className={`config-action ${activeDrawer === "budget" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "budget" ? null : "budget"))
            }
            type="button"
          >
            Config
          </button>
          <button
            className={`config-action ${activeDrawer === "models" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "models" ? null : "models"))
            }
            type="button"
          >
            Models
          </button>
          <button
            className={`config-action ${activeDrawer === "settings" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "settings" ? null : "settings"))
            }
            type="button"
          >
            Settings
          </button>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            type="button"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            Theme
          </button>
        </div>
      </header>

      <div className={`dashboard-grid ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <aside className="left-column" aria-hidden={sidebarCollapsed}>
          {!sidebarCollapsed ? (
            <div className="left-column-shell">
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
                    className={`rail-project-chip ${
                      activeProject === null ? "active" : ""
                    }`}
                    onClick={() => handleSelectProject(null)}
                    type="button"
                  >
                    <span className="rail-project-name">All runs</span>
                  </button>
                  {projects.slice(0, 6).map((project) => {
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
            </div>
          ) : null}
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
                pending={{
                  cancel: cancelMutation.isPending,
                  resume: resumeMutation.isPending,
                  retry: retryMutation.isPending,
                  clarification: clarificationMutation.isPending,
                  approve: approveMutation.isPending,
                  reject: rejectMutation.isPending,
                  requestChanges: requestChangesMutation.isPending,
                }}
              />
            </div>
          )}

          {isLanding ? (
            <div className="landing-hero">
              <div className="landing-hero-inner">
                <h2 className="landing-title">Open Research</h2>
              </div>
            </div>
          ) : null}

          <div className="center-input">
            <div className="center-input-shell">
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
                  placeholder={composerPlaceholder}
                  rows={1}
                />
                <button
                  className="primary-button"
                  disabled={composerDisabled}
                  onClick={handleSubmit}
                  type="button"
                >
                  {composerButtonLabel}
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
