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
  ProfileDrawerPanel,
  RunComposer,
  SettingsDrawerPanel,
  SourcesDrawerPanel,
  WorkflowDrawerPanel,
} from "./run-composer";
import { RunHistory } from "./run-history";
import { RunWorkspace } from "./run-workspace";

type DrawerKey = "settings" | FocusedDrawerKey;

const DRAWER_META: Record<DrawerKey, { eyebrow: string; title: string }> = {
  settings: { eyebrow: "Settings", title: "Agent behavior" },
  workflow: { eyebrow: "Workflow", title: "Execution and policy" },
  sources: { eyebrow: "Sources", title: "Source registry" },
  budget: { eyebrow: "Budget", title: "Budget controls" },
  models: { eyebrow: "Models", title: "Model selection" },
  profile: { eyebrow: "Profile", title: "Profile and memory" },
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
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const stored = localStorage.getItem("or-theme");
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
      document.documentElement.setAttribute("data-theme", stored);
    }
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
    mutationFn: async () => {
      await updateProfilePreferences(apiBaseUrl, profileId, profilePreferences);
      const payload = {
        question: questionDraft.trim(),
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
    onSuccess: (run) => {
      setSubmitError(null);
      setSelectedRunId(run.id);
      clearRunInputAssets();
      clearStagedRunAssets();
      if (textareaRef.current) textareaRef.current.style.height = "";
      void queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
      void queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, run.id] });
      void queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, run.id] });
    },
    onError: (error) => {
      setSubmitError(error instanceof Error ? error.message : "Failed to start run.");
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

  const handleSubmit = () => {
    if (questionDraft.trim().length >= 12 && !createRunMutation.isPending) {
      createRunMutation.mutate();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <main className="dashboard-page">
      {/* ── Header bar ── */}
      <header className="hero">
        <p className="eyebrow">Open Research Console</p>
        <div className="hero-actions">
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
            className={`config-action ${activeDrawer === "workflow" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "workflow" ? null : "workflow"))
            }
            type="button"
          >
            Workflow
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
            Budget
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
            className={`config-action ${activeDrawer === "profile" ? "active" : ""}`}
            onClick={() =>
              setActiveDrawer((prev) => (prev === "profile" ? null : "profile"))
            }
            type="button"
          >
            Profile
          </button>
        </div>
        <div className="hero-badges">
          <span className="pill">
            {publicConfigQuery.data?.backends.workflow ?? "\u2014"}
          </span>
          <span className="pill muted">
            {streamState?.connectionState ?? "idle"}
          </span>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            type="button"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </header>

      {/* ── 3-column grid ── */}
      <div className="dashboard-grid">
        {/* Left sidebar: settings */}
        <aside className="left-column">
          <RunComposer
            publicConfig={publicConfigQuery.data}
          />
          <RunHistory
            runs={visibleRuns}
            selectedRunId={selectedRunId}
            onSelect={setSelectedRunId}
          />
        </aside>

        {/* Workspace */}
        <section className="center-column workspace-column">
          <div className="center-scroll" ref={centerScrollRef}>
            <RunWorkspace
              workspace={activeWorkspace}
              rawEvents={eventFeed}
              conversationMessages={conversationMessages}
              connectionState={streamState?.connectionState ?? "idle"}
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

          {/* ── Pinned input bar at bottom ── */}
          <div className="center-input">
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
              disabled={questionDraft.trim().length < 12 || createRunMutation.isPending}
              onClick={handleSubmit}
              type="button"
            >
              {createRunMutation.isPending ? "..." : "Run"}
            </button>
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
              </div>
              <button
                className="drawer-close"
                onClick={() => setActiveDrawer(null)}
                type="button"
              >
                Close
              </button>
            </div>
            <div className="drawer-body">
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
