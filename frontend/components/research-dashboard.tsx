"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BrainCircuit,
  FileText,
  Folder,
  FolderKanban,
  FolderPlus,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  SearchCheck,
  Send,
  Settings2,
  SlidersHorizontal,
  Sun,
  Trash2,
  Upload,
  X,
  type LucideIcon,
} from "lucide-react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

import {
  answerClarification,
  approvePlan,
  cancelRun,
  createProject,
  createJob,
  createRun,
  deleteProjectAsset,
  fetchProjectDetail,
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
  uploadProjectFiles,
  updateProfilePreferences,
} from "@/lib/api";
import type {
  ProjectSummary,
  ResearchAssetRecord,
  ResearchAssetUsage,
  RunConversationMessage,
  RunDetail,
  RunEvent,
  RunSummary,
} from "@/lib/types";
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

const RunWorkspace = dynamic(
  () => import("./run-workspace").then((module) => module.RunWorkspace),
  {
    loading: () => <WorkspaceModuleFallback />,
    ssr: false,
  },
);

type DrawerKey = "project" | "settings" | FocusedDrawerKey;
type ProjectHomeTab = "chats" | "sources";

interface ResearchDashboardProps {
  initialProjectId?: string | null;
}

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

const HEADER_DRAWERS: Array<{
  key: DrawerKey;
  label: string;
  icon: LucideIcon;
}> = [
  { key: "project", label: "Project", icon: FolderKanban },
  { key: "sources", label: "Sources", icon: SearchCheck },
  { key: "budget", label: "Config", icon: SlidersHorizontal },
  { key: "models", label: "Models", icon: BrainCircuit },
  { key: "settings", label: "Settings", icon: Settings2 },
];

function sortRunsByUpdatedAt(runs: RunSummary[]): RunSummary[] {
  return [...runs].sort(
    (left, right) =>
      new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime(),
  );
}

function describeProjectAssetUsage(usage: ResearchAssetUsage): string {
  return usage === "planning_context" ? "Planning context" : "Reference source";
}

function formatProjectAssetStatus(asset: ResearchAssetRecord): string {
  return asset.processing_status.replaceAll("_", " ");
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

function WorkspaceModuleFallback() {
  return (
    <section className="panel workspace-panel workspace-loading-panel" aria-busy="true">
      <div className="workspace-loading-grid">
        <div className="workspace-loading-main">
          <span className="skeleton skeleton-kicker" />
          <span className="skeleton skeleton-title" />
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line short" />
        </div>
        <div className="workspace-loading-side">
          {Array.from({ length: 5 }).map((_, index) => (
            <span className="skeleton skeleton-card" key={index} />
          ))}
        </div>
      </div>
    </section>
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

export function ResearchDashboard({ initialProjectId }: ResearchDashboardProps) {
  const queryClient = useQueryClient();
  const router = useRouter();
  const { theme, toggle: toggleTheme } = useTheme();
  const composerErrorId = useId();
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [activeDrawer, setActiveDrawer] = useState<DrawerKey | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showSidebarProjectForm, setShowSidebarProjectForm] = useState(false);
  const [sidebarProjectName, setSidebarProjectName] = useState("");
  const [sidebarProjectDescription, setSidebarProjectDescription] = useState("");
  const [projectHomeTab, setProjectHomeTab] = useState<ProjectHomeTab>("chats");
  const [projectSourceUsage, setProjectSourceUsage] =
    useState<ResearchAssetUsage>("reference_source");
  const [projectSourceError, setProjectSourceError] = useState<string | null>(null);
  const centerScrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pendingQuestionRef = useRef<string>("");
  const routedProjectRef = useRef<string | null | undefined>(undefined);
  const projectSourceFileInputRef = useRef<HTMLInputElement>(null);

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
  const effectiveProjectId =
    initialProjectId === undefined ? selectedProjectId : initialProjectId;
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

  const projectDetailQuery = useQuery({
    queryKey: ["project-detail", apiBaseUrl, effectiveProjectId],
    queryFn: () => fetchProjectDetail(apiBaseUrl, effectiveProjectId as string),
    enabled: Boolean(effectiveProjectId),
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
    if (initialProjectId === undefined) {
      return;
    }
    if (routedProjectRef.current === initialProjectId) {
      return;
    }
    const routedProjectId = initialProjectId;
    routedProjectRef.current = initialProjectId;
    setSelectedProjectId(routedProjectId);
    setSelectedRunId(null);
  }, [initialProjectId, setSelectedProjectId, setSelectedRunId]);

  useEffect(() => {
    if (window.innerWidth < 820) {
      setSidebarCollapsed(true);
      return;
    }
    const stored = localStorage.getItem("or-sidebar-collapsed");
    if (stored === "true" || stored === "false") {
      setSidebarCollapsed(stored === "true");
      return;
    }
    setSidebarCollapsed(window.innerWidth < 900);
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 820) {
        setSidebarCollapsed(true);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
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

  const createSidebarProjectMutation = useMutation({
    mutationFn: () =>
      createProject(apiBaseUrl, {
        name: sidebarProjectName.trim(),
        description: sidebarProjectDescription.trim() || null,
      }),
    onSuccess: async (project) => {
      setSidebarProjectName("");
      setSidebarProjectDescription("");
      setShowSidebarProjectForm(false);
      setSelectedProjectId(project.id);
      setSelectedRunId(null);
      setProjectHomeTab("chats");
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
      await queryClient.invalidateQueries({
        queryKey: ["project-detail", apiBaseUrl, project.id],
      });
      router.push(`/projects?projectId=${encodeURIComponent(project.id)}`);
    },
    onError: (error) => {
      setSubmitError(error instanceof Error ? error.message : "Failed to create project.");
    },
  });

  const uploadProjectSourceMutation = useMutation({
    mutationFn: ({ files, usage }: { files: FileList; usage: ResearchAssetUsage }) =>
      uploadProjectFiles(apiBaseUrl, effectiveProjectId ?? "", usage, Array.from(files)),
    onSuccess: async () => {
      setProjectSourceError(null);
      await queryClient.invalidateQueries({
        queryKey: ["project-detail", apiBaseUrl, effectiveProjectId],
      });
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
    },
    onError: (error) => {
      setProjectSourceError(error instanceof Error ? error.message : "Failed to upload source.");
    },
  });

  const deleteProjectSourceMutation = useMutation({
    mutationFn: (assetId: string) =>
      deleteProjectAsset(apiBaseUrl, effectiveProjectId ?? "", assetId),
    onSuccess: async () => {
      setProjectSourceError(null);
      await queryClient.invalidateQueries({
        queryKey: ["project-detail", apiBaseUrl, effectiveProjectId],
      });
      await queryClient.invalidateQueries({ queryKey: ["projects", apiBaseUrl] });
    },
    onError: (error) => {
      setProjectSourceError(error instanceof Error ? error.message : "Failed to remove source.");
    },
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
        project_id: effectiveProjectId,
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
      if (effectiveProjectId) {
        void queryClient.invalidateQueries({
          queryKey: ["project-detail", apiBaseUrl, effectiveProjectId],
        });
      }
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
  const workspaceLoading = Boolean(selectedRunId) && workspaceQuery.isLoading && !activeWorkspace;
  const workspaceErrorMessage =
    workspaceQuery.error instanceof Error ? workspaceQuery.error.message : null;
  const projects = projectsQuery.data ?? [];
  const activeProject =
    projectDetailQuery.data ??
    projects.find((project) => project.id === effectiveProjectId) ??
    null;
  const projectHomeProject: ProjectSummary | null =
    activeProject ??
    (effectiveProjectId
      ? {
          id: effectiveProjectId,
          name: "Project",
          description: null,
          created_at: "",
          updated_at: "",
        }
      : null);
  const activeProjectAssets = useMemo(
    () =>
      (projectDetailQuery.data?.assets ?? []).filter(
        (asset) => asset.source_type === "file",
      ),
    [projectDetailQuery.data?.assets],
  );
  const globalRuns = runsQuery.data ?? [];
  const projectRuns = useMemo(
    () => globalRuns.filter((run) => run.project_id === effectiveProjectId),
    [globalRuns, effectiveProjectId],
  );
  const sortedProjectRuns = useMemo(() => sortRunsByUpdatedAt(projectRuns), [projectRuns]);
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
  const isLanding = !selectedRunId;
  const isProjectHome = isLanding && Boolean(projectHomeProject);
  const composerDisabled =
    composerMode === "message"
      ? !questionDraft.trim() || chatMutation.isPending
      : questionDraft.trim().length < 12 || createRunMutation.isPending;
  const composerPlaceholder =
    composerMode === "message"
      ? "Message"
      : activeProject
        ? `Research in ${activeProject.name}`
        : effectiveProjectId
          ? "Research in this project"
        : "Research topic";
  const composerButtonLabel =
    composerMode === "message"
      ? chatMutation.isPending
        ? "..."
        : "Send"
      : createRunMutation.isPending
        ? "..."
        : "Run";

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

  const handleNewChat = useCallback(() => {
    setSelectedRunId(null);
    setSubmitError(null);
    setQuestionDraft("");
    setProjectHomeTab("chats");
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
      centerScrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    });
  }, [setQuestionDraft, setSelectedRunId]);

  const handleSelectProject = useCallback(
    (projectId: string | null) => {
      setSelectedProjectId(projectId);
      setSelectedRunId(null);
      setSubmitError(null);
      setProjectHomeTab("chats");
      if (projectId) {
        router.push(`/projects?projectId=${encodeURIComponent(projectId)}`);
      } else {
        router.push("/");
      }
    },
    [router, setSelectedProjectId, setSelectedRunId],
  );

  const handleSidebarProjectSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (sidebarProjectName.trim() && !createSidebarProjectMutation.isPending) {
      createSidebarProjectMutation.mutate();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleProjectSourceFiles = (files: FileList | null) => {
    if (!files?.length || !effectiveProjectId) {
      return;
    }
    uploadProjectSourceMutation.mutate({ files, usage: projectSourceUsage });
  };

  const renderComposer = (className = "center-input") => (
    <form
      className={className}
      onSubmit={(event) => {
        event.preventDefault();
        handleSubmit();
      }}
    >
      <div className="center-input-shell">
        <div className="center-input-row">
          <textarea
            aria-describedby={submitError ? composerErrorId : undefined}
            aria-invalid={Boolean(submitError)}
            aria-label={
              composerMode === "message"
                ? "Message this completed research run"
                : activeProject
                  ? `Research in ${activeProject.name}`
                  : effectiveProjectId
                    ? "Research in this project"
                  : "Research topic"
            }
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
            aria-label={composerMode === "message" ? "Send message" : "Run research"}
            className="primary-button"
            disabled={composerDisabled}
            type="submit"
          >
            {composerMode === "message" ? (
              <Send aria-hidden size={14} strokeWidth={2} />
            ) : (
              <SearchCheck aria-hidden size={14} strokeWidth={2} />
            )}
            <span>{composerButtonLabel}</span>
          </button>
        </div>
      </div>
    </form>
  );

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
          {sidebarCollapsed ? (
            <PanelLeftOpen aria-hidden size={17} strokeWidth={1.9} />
          ) : (
            <PanelLeftClose aria-hidden size={17} strokeWidth={1.9} />
          )}
        </button>
        <div className="hero-brand">
          <h1 className="hero-title">Open Research</h1>
        </div>
        <div className="hero-spacer" />
        <nav className="hero-actions" aria-label="Run configuration">
          {HEADER_DRAWERS.map(({ key, label, icon: Icon }) => (
            <button
              aria-label={`Open ${label.toLowerCase()} drawer`}
              aria-pressed={activeDrawer === key}
              className={`config-action ${activeDrawer === key ? "active" : ""}`}
              key={key}
              onClick={() =>
                setActiveDrawer((prev) => (prev === key ? null : key))
              }
              type="button"
            >
              <Icon aria-hidden size={14} strokeWidth={1.9} />
              <span className="config-action-label">{label}</span>
            </button>
          ))}
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            type="button"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <Sun aria-hidden size={14} strokeWidth={1.9} />
            ) : (
              <Moon aria-hidden size={14} strokeWidth={1.9} />
            )}
            <span className="config-action-label">Theme</span>
          </button>
        </nav>
      </header>

      <div className={`dashboard-grid ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        {!sidebarCollapsed ? (
          <button
            aria-label="Close sidebar"
            className="mobile-rail-backdrop"
            onClick={() => setSidebarCollapsed(true)}
            type="button"
          />
        ) : null}
        <aside className="left-column" aria-hidden={sidebarCollapsed}>
          {!sidebarCollapsed ? (
            <div className="left-column-shell">
              <section className="rail-projects">
                <div className="rail-section-head">
                  <p className="eyebrow">Projects</p>
                  <button
                    className="rail-section-action"
                    aria-label="Create project"
                    onClick={() => {
                      setShowSidebarProjectForm((value) => !value);
                      setSubmitError(null);
                    }}
                    title="Create project"
                    type="button"
                  >
                    <FolderPlus aria-hidden size={14} strokeWidth={1.9} />
                  </button>
                </div>
                {showSidebarProjectForm ? (
                  <form className="rail-project-create" onSubmit={handleSidebarProjectSubmit}>
                    <label className="sr-only" htmlFor="sidebar-project-name">
                      Project name
                    </label>
                    <input
                      autoFocus
                      className="text-input"
                      id="sidebar-project-name"
                      onChange={(event) => setSidebarProjectName(event.target.value)}
                      placeholder="Project name"
                      value={sidebarProjectName}
                    />
                    <label className="sr-only" htmlFor="sidebar-project-description">
                      Project description
                    </label>
                    <input
                      className="text-input"
                      id="sidebar-project-description"
                      onChange={(event) => setSidebarProjectDescription(event.target.value)}
                      placeholder="Description"
                      value={sidebarProjectDescription}
                    />
                    <div className="rail-project-create-actions">
                      <button
                        className="secondary-button"
                        onClick={() => {
                          setShowSidebarProjectForm(false);
                          setSidebarProjectName("");
                          setSidebarProjectDescription("");
                        }}
                        type="button"
                      >
                        Cancel
                      </button>
                      <button
                        className="primary-button"
                        disabled={
                          !sidebarProjectName.trim() ||
                          createSidebarProjectMutation.isPending
                        }
                        type="submit"
                      >
                        {createSidebarProjectMutation.isPending ? "Creating..." : "Create"}
                      </button>
                    </div>
                  </form>
                ) : null}
                <div className="rail-project-list">
                  <button
                    className={`rail-project-chip ${
                      effectiveProjectId === null ? "active" : ""
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
                          effectiveProjectId === project.id ? "active" : ""
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
                runs={globalRuns}
                isLoading={runsQuery.isLoading}
                errorMessage={
                  runsQuery.error instanceof Error ? runsQuery.error.message : null
                }
                selectedRunId={selectedRunId}
                onSelect={setSelectedRunId}
                onNewChat={handleNewChat}
                newChatLabel={
                  activeProject ? `New chat in ${activeProject.name}` : "New chat"
                }
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
                isLoading={workspaceLoading}
                errorMessage={workspaceErrorMessage}
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

          {isProjectHome && projectHomeProject ? (
            <div className="project-home">
              <div className="project-home-inner">
                <header className="project-home-header">
                  <span className="project-home-icon" aria-hidden="true">
                    <Folder size={26} strokeWidth={1.75} />
                  </span>
                  <div>
                    <p className="eyebrow">Project</p>
                    <h2>{projectHomeProject.name}</h2>
                    {projectHomeProject.description ? (
                      <p>{projectHomeProject.description}</p>
                    ) : null}
                  </div>
                </header>

                {renderComposer("project-home-composer")}

                <div className="project-home-tabs" role="tablist" aria-label="Project sections">
                  <button
                    aria-selected={projectHomeTab === "chats"}
                    className={projectHomeTab === "chats" ? "active" : ""}
                    onClick={() => setProjectHomeTab("chats")}
                    role="tab"
                    type="button"
                  >
                    Chats
                    <span>{sortedProjectRuns.length}</span>
                  </button>
                  <button
                    aria-selected={projectHomeTab === "sources"}
                    className={projectHomeTab === "sources" ? "active" : ""}
                    onClick={() => setProjectHomeTab("sources")}
                    role="tab"
                    type="button"
                  >
                    Sources
                    <span>{activeProjectAssets.length}</span>
                  </button>
                </div>

                {projectHomeTab === "chats" ? (
                  <section className="project-home-panel" role="tabpanel">
                    {runsQuery.isLoading ? (
                      <div className="project-home-list" aria-label="Loading project chats">
                        {Array.from({ length: 4 }).map((_, index) => (
                          <div className="project-home-skeleton" key={index} />
                        ))}
                      </div>
                    ) : sortedProjectRuns.length > 0 ? (
                      <div className="project-home-list">
                        {sortedProjectRuns.map((run) => (
                          <button
                            className="project-chat-row"
                            key={run.id}
                            onClick={() => setSelectedRunId(run.id)}
                            type="button"
                          >
                            <span>{run.conversation_topic || run.report_title || run.question}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="project-home-empty">
                        <strong>No chats yet</strong>
                        <span>Research runs in {projectHomeProject.name} will live here.</span>
                      </div>
                    )}
                  </section>
                ) : (
                  <section className="project-home-panel" role="tabpanel">
                    <input
                      ref={projectSourceFileInputRef}
                      className="sr-only-input"
                      type="file"
                      multiple
                      onChange={(event) => {
                        handleProjectSourceFiles(event.target.files);
                        event.currentTarget.value = "";
                      }}
                    />
                    <div className="project-source-form">
                      <label className="sr-only" htmlFor="project-source-usage">
                        Source role
                      </label>
                      <select
                        className="text-input"
                        id="project-source-usage"
                        onChange={(event) =>
                          setProjectSourceUsage(event.target.value as ResearchAssetUsage)
                        }
                        value={projectSourceUsage}
                      >
                        <option value="reference_source">Reference source</option>
                        <option value="planning_context">Planning context</option>
                      </select>
                      <button
                        className="secondary-button"
                        disabled={uploadProjectSourceMutation.isPending}
                        onClick={() => projectSourceFileInputRef.current?.click()}
                        type="button"
                      >
                        <Upload aria-hidden size={14} strokeWidth={1.9} />
                        <span>
                          {uploadProjectSourceMutation.isPending ? "Uploading..." : "Upload files"}
                        </span>
                      </button>
                    </div>

                    {projectSourceError ? (
                      <p className="error-text project-source-error" role="alert">
                        {projectSourceError}
                      </p>
                    ) : null}

                    {projectDetailQuery.isLoading ? (
                      <div className="project-home-list" aria-label="Loading project sources">
                        {Array.from({ length: 3 }).map((_, index) => (
                          <div className="project-home-skeleton" key={index} />
                        ))}
                      </div>
                    ) : activeProjectAssets.length > 0 ? (
                      <ul className="project-source-list">
                        {activeProjectAssets.map((asset) => (
                          <li className="project-source-row" key={asset.id}>
                            <FileText aria-hidden size={16} strokeWidth={1.8} />
                            <div>
                              <strong>{asset.label}</strong>
                              <span>
                                {describeProjectAssetUsage(asset.usage)} · File ·{" "}
                                {formatProjectAssetStatus(asset)}
                              </span>
                              {asset.preview_excerpt ? <code>{asset.preview_excerpt}</code> : null}
                            </div>
                            <button
                              aria-label={`Remove ${asset.label}`}
                              className="project-source-remove"
                              disabled={deleteProjectSourceMutation.isPending}
                              onClick={() => deleteProjectSourceMutation.mutate(asset.id)}
                              type="button"
                            >
                              <Trash2 aria-hidden size={14} strokeWidth={1.9} />
                            </button>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="project-home-empty">
                        <strong>No sources yet</strong>
                        <span>Upload files once and every run in this project can use them.</span>
                      </div>
                    )}
                  </section>
                )}
              </div>
            </div>
          ) : null}

          {isLanding && !isProjectHome ? (
            <div className="landing-hero">
              <div className="landing-hero-inner">
                <h2 className="landing-title">What would you like to research next?</h2>
                <button
                  className="landing-config-button"
                  onClick={() => setActiveDrawer("budget")}
                  type="button"
                >
                  <SlidersHorizontal aria-hidden="true" size={15} strokeWidth={1.9} />
                  <span>Adjust config</span>
                </button>
              </div>
            </div>
          ) : null}

          {!isProjectHome ? renderComposer("center-input") : null}
          {submitError ? (
            <div className="composer-error-region">
              <p className="error-text" id={composerErrorId} role="alert">
                {submitError}
              </p>
            </div>
          ) : null}
        </section>
      </div>

      <Dialog.Root
        open={Boolean(activeDrawer)}
        onOpenChange={(open) => {
          if (!open) setActiveDrawer(null);
        }}
      >
        {activeDrawer ? (
          <Dialog.Portal>
          <Dialog.Overlay className="drawer-backdrop" />
          <Dialog.Content className="drawer">
            <div className="drawer-header">
              <div>
                <p className="eyebrow">{DRAWER_META[activeDrawer].eyebrow}</p>
                <Dialog.Title className="panel-title">
                  {DRAWER_META[activeDrawer].title}
                </Dialog.Title>
                <Dialog.Description className="drawer-lead">
                  {DRAWER_META[activeDrawer].description}
                </Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button
                  className="drawer-close"
                  type="button"
                  aria-label="Close drawer"
                >
                  <X aria-hidden size={14} strokeWidth={2} />
                  <span>Close</span>
                </button>
              </Dialog.Close>
            </div>
            <div className="drawer-body">
              {activeDrawer === "project" ? (
                <ProjectPanel
                  publicConfig={publicConfigQuery.data}
                  onProjectChange={handleSelectProject}
                />
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
          </Dialog.Content>
          </Dialog.Portal>
        ) : null}
      </Dialog.Root>
    </main>
  );
}
