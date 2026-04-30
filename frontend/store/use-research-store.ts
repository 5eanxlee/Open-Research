"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

import {
  DEFAULT_AGENT_CONFIG,
  DEFAULT_API_BASE_URL,
  DEFAULT_BUDGET,
  DEFAULT_MEMORY_POLICY,
  DEFAULT_PROFILE_PREFERENCES,
  DEFAULT_REPORT_OUTPUT_CONFIG,
} from "@/lib/defaults";
import type {
  AgentConfig,
  BudgetPolicy,
  ClarifierConfig,
  ExecutionMode,
  MemoryInfluencePolicy,
  ModelConfigOverride,
  ProfilePreferences,
  PublicRuntimeConfig,
  ReportOutputConfig,
  ResearchInputAsset,
  RunEvent,
  RunWorkspaceSnapshot,
  StreamConnectionState,
  StagedAssetRecord,
} from "@/lib/types";

interface RunStreamState {
  events: RunEvent[];
  lastEventId: number;
  connectionState: StreamConnectionState;
  lastMode: string | null;
}

interface ResearchStoreState {
  apiBaseUrl: string;
  selectedRunId: string | null;
  selectedProjectId: string | null;
  profileId: string;
  profilePreferences: ProfilePreferences;
  memoryPolicyOverride: MemoryInfluencePolicy;
  executionMode: ExecutionMode;
  requirePlanApproval: boolean;
  clarifierConfig: ClarifierConfig;
  sourceSelection: string[];
  runInputAssets: ResearchInputAsset[];
  stagedRunAssets: StagedAssetRecord[];
  asyncSubmit: boolean;
  questionDraft: string;
  budget: BudgetPolicy;
  reportOutputConfig: ReportOutputConfig;
  agentConfig: AgentConfig;
  modelConfigOverride: ModelConfigOverride;
  serverDefaultsApplied: boolean;
  runStreams: Record<string, RunStreamState>;
  runWorkspaces: Record<string, RunWorkspaceSnapshot>;
  setApiBaseUrl: (value: string) => void;
  setSelectedRunId: (runId: string | null) => void;
  setSelectedProjectId: (projectId: string | null) => void;
  setProfileId: (value: string) => void;
  updateProfilePreferences: (patch: Partial<ProfilePreferences>) => void;
  updateMemoryPolicyOverride: (patch: Partial<MemoryInfluencePolicy>) => void;
  setExecutionMode: (value: ExecutionMode) => void;
  setRequirePlanApproval: (value: boolean) => void;
  updateClarifierConfig: (patch: Partial<ClarifierConfig>) => void;
  setSourceSelection: (value: string[]) => void;
  toggleSourceSelection: (value: string) => void;
  addRunInputAsset: (asset: ResearchInputAsset) => void;
  removeRunInputAsset: (index: number) => void;
  replaceRunInputAssets: (assets: ResearchInputAsset[]) => void;
  clearRunInputAssets: () => void;
  addStagedRunAssets: (assets: StagedAssetRecord[]) => void;
  removeStagedRunAsset: (assetId: string) => void;
  clearStagedRunAssets: () => void;
  setAsyncSubmit: (value: boolean) => void;
  setQuestionDraft: (value: string) => void;
  updateBudget: (patch: Partial<BudgetPolicy>) => void;
  updateReportOutputConfig: (patch: Partial<ReportOutputConfig>) => void;
  updateAgentConfig: (patch: Partial<AgentConfig>) => void;
  updateModelConfigOverride: (patch: Partial<ModelConfigOverride>) => void;
  resetModelConfigOverride: () => void;
  hydrateFromServerConfig: (config: PublicRuntimeConfig) => void;
  resetComposer: () => void;
  appendEvent: (runId: string, event: RunEvent) => void;
  replaceRunEvents: (runId: string, events: RunEvent[]) => void;
  clearRunEvents: (runId: string) => void;
  setConnectionState: (runId: string, value: StreamConnectionState, mode?: string | null) => void;
  setRunWorkspace: (runId: string, workspace: RunWorkspaceSnapshot) => void;
  clearRunWorkspace: (runId: string) => void;
}

function ensureRunStreamState(
  runStreams: Record<string, RunStreamState>,
  runId: string,
): RunStreamState {
  return (
    runStreams[runId] ?? {
      events: [],
      lastEventId: 0,
      connectionState: "idle",
      lastMode: null,
    }
  );
}

export const useResearchStore = create<ResearchStoreState>()(
  persist(
    (set, get) => ({
      apiBaseUrl: DEFAULT_API_BASE_URL,
      selectedRunId: null,
      selectedProjectId: null,
      profileId: "default",
      profilePreferences: DEFAULT_PROFILE_PREFERENCES,
      memoryPolicyOverride: DEFAULT_MEMORY_POLICY,
      executionMode: "deep",
      requirePlanApproval: false,
      clarifierConfig: {
        enabled: true,
        max_questions: 2,
        ambiguity_threshold: 0.45,
        require_response_for_deep: false,
      },
      sourceSelection: [],
      runInputAssets: [],
      stagedRunAssets: [],
      asyncSubmit: false,
      questionDraft: "",
      budget: DEFAULT_BUDGET,
      reportOutputConfig: DEFAULT_REPORT_OUTPUT_CONFIG,
      agentConfig: DEFAULT_AGENT_CONFIG,
      modelConfigOverride: {},
      serverDefaultsApplied: false,
      runStreams: {},
      runWorkspaces: {},
      setApiBaseUrl: (value) => set({ apiBaseUrl: value }),
      setSelectedRunId: (runId) => set({ selectedRunId: runId }),
      setSelectedProjectId: (projectId) => set({ selectedProjectId: projectId }),
      setProfileId: (value) => set({ profileId: value }),
      updateProfilePreferences: (patch) =>
        set((state) => ({
          profilePreferences: {
            ...state.profilePreferences,
            ...patch,
          },
        })),
      updateMemoryPolicyOverride: (patch) =>
        set((state) => ({
          memoryPolicyOverride: {
            ...state.memoryPolicyOverride,
            ...patch,
          },
        })),
      setExecutionMode: (value) => set({ executionMode: value }),
      setRequirePlanApproval: (value) => set({ requirePlanApproval: value }),
      updateClarifierConfig: (patch) =>
        set((state) => ({
          clarifierConfig: {
            ...state.clarifierConfig,
            ...patch,
          },
        })),
      setSourceSelection: (value) => set({ sourceSelection: value }),
      toggleSourceSelection: (value) =>
        set((state) => ({
          sourceSelection: state.sourceSelection.includes(value)
            ? state.sourceSelection.filter((entry) => entry !== value)
            : [...state.sourceSelection, value],
        })),
      addRunInputAsset: (asset) =>
        set((state) => ({
          runInputAssets: [...state.runInputAssets, asset],
        })),
      removeRunInputAsset: (index) =>
        set((state) => ({
          runInputAssets: state.runInputAssets.filter((_, assetIndex) => assetIndex !== index),
        })),
      replaceRunInputAssets: (assets) => set({ runInputAssets: assets }),
      clearRunInputAssets: () => set({ runInputAssets: [] }),
      addStagedRunAssets: (assets) =>
        set((state) => ({
          stagedRunAssets: [...state.stagedRunAssets, ...assets],
        })),
      removeStagedRunAsset: (assetId) =>
        set((state) => ({
          stagedRunAssets: state.stagedRunAssets.filter((asset) => asset.id !== assetId),
        })),
      clearStagedRunAssets: () => set({ stagedRunAssets: [] }),
      setAsyncSubmit: (value) => set({ asyncSubmit: value }),
      setQuestionDraft: (value) => set({ questionDraft: value }),
      updateBudget: (patch) =>
        set((state) => ({
          budget: {
            ...state.budget,
            ...patch,
          },
        })),
      updateReportOutputConfig: (patch) =>
        set((state) => {
          const next = {
            ...state.reportOutputConfig,
            ...patch,
          };
          const minWords = Math.max(100, Math.round(next.min_words));
          const maxWords = Math.max(minWords, Math.round(next.max_words));
          return {
            reportOutputConfig: {
              min_words: minWords,
              max_words: maxWords,
            },
          };
        }),
      updateAgentConfig: (patch) =>
        set((state) => ({
          agentConfig: {
            ...state.agentConfig,
            ...patch,
          },
        })),
      updateModelConfigOverride: (patch) =>
        set((state) => ({
          modelConfigOverride: {
            ...state.modelConfigOverride,
            ...patch,
          },
        })),
      resetModelConfigOverride: () => set({ modelConfigOverride: {} }),
      hydrateFromServerConfig: (config) => {
        if (get().serverDefaultsApplied) {
          return;
        }
        set({
          budget: config.default_budget,
          agentConfig: config.default_agent_config,
          memoryPolicyOverride: DEFAULT_MEMORY_POLICY,
          sourceSelection:
            get().sourceSelection.length > 0
              ? get().sourceSelection
              : config.default_source_selection,
          serverDefaultsApplied: true,
        });
      },
      resetComposer: () =>
        set({
          selectedProjectId: null,
          profileId: "default",
          profilePreferences: DEFAULT_PROFILE_PREFERENCES,
          memoryPolicyOverride: DEFAULT_MEMORY_POLICY,
          executionMode: "deep",
          requirePlanApproval: false,
          clarifierConfig: {
            enabled: true,
            max_questions: 2,
            ambiguity_threshold: 0.45,
            require_response_for_deep: false,
          },
          sourceSelection: [],
          runInputAssets: [],
          stagedRunAssets: [],
          asyncSubmit: false,
          questionDraft: "",
          budget: DEFAULT_BUDGET,
          reportOutputConfig: DEFAULT_REPORT_OUTPUT_CONFIG,
          agentConfig: DEFAULT_AGENT_CONFIG,
          modelConfigOverride: {},
        }),
      appendEvent: (runId, event) =>
        set((state) => {
          const runStream = ensureRunStreamState(state.runStreams, runId);
          if (event.id <= runStream.lastEventId) {
            return state;
          }
          const events = [...runStream.events, event].slice(-250);
          return {
            runStreams: {
              ...state.runStreams,
              [runId]: {
                ...runStream,
                events,
                lastEventId: event.id,
              },
            },
          };
        }),
      replaceRunEvents: (runId, events) =>
        set((state) => {
          const runStream = ensureRunStreamState(state.runStreams, runId);
          if (events.length === 0 && runStream.events.length > 0) {
            return state;
          }
          const deduped = events
            .slice()
            .sort((left, right) => left.id - right.id)
            .slice(-250);
          return {
            runStreams: {
              ...state.runStreams,
              [runId]: {
                ...runStream,
                events: deduped,
                lastEventId: deduped.at(-1)?.id ?? runStream.lastEventId,
              },
            },
          };
        }),
      clearRunEvents: (runId) =>
        set((state) => ({
          runStreams: {
            ...state.runStreams,
            [runId]: {
              events: [],
              lastEventId: 0,
              connectionState: "idle",
              lastMode: null,
            },
          },
        })),
      setConnectionState: (runId, value, mode = null) =>
        set((state) => {
          const runStream = ensureRunStreamState(state.runStreams, runId);
          return {
            runStreams: {
              ...state.runStreams,
              [runId]: {
                ...runStream,
                connectionState: value,
                lastMode: mode,
              },
            },
          };
        }),
      setRunWorkspace: (runId, workspace) =>
        set((state) => ({
          runWorkspaces: {
            ...state.runWorkspaces,
            [runId]: workspace,
          },
        })),
      clearRunWorkspace: (runId) =>
        set((state) => {
          const next = { ...state.runWorkspaces };
          delete next[runId];
          return { runWorkspaces: next };
        }),
    }),
    {
      name: "open-research-dashboard",
      version: 3,
      migrate: (persistedState) => {
        if (!persistedState || typeof persistedState !== "object") {
          return persistedState as never;
        }
        const state = persistedState as Partial<ResearchStoreState>;
        return {
          apiBaseUrl: state.apiBaseUrl ?? DEFAULT_API_BASE_URL,
          selectedRunId: state.selectedRunId ?? null,
          selectedProjectId: state.selectedProjectId ?? null,
          profileId: state.profileId ?? "default",
          profilePreferences: state.profilePreferences ?? DEFAULT_PROFILE_PREFERENCES,
          memoryPolicyOverride: state.memoryPolicyOverride ?? DEFAULT_MEMORY_POLICY,
          executionMode: "deep",
          requirePlanApproval: state.requirePlanApproval ?? false,
          clarifierConfig: state.clarifierConfig ?? {
            enabled: true,
            max_questions: 2,
            ambiguity_threshold: 0.45,
            require_response_for_deep: false,
          },
          sourceSelection: state.sourceSelection ?? [],
          runInputAssets: state.runInputAssets ?? [],
          stagedRunAssets: state.stagedRunAssets ?? [],
          asyncSubmit: state.asyncSubmit ?? false,
          budget: state.budget ?? DEFAULT_BUDGET,
          reportOutputConfig: state.reportOutputConfig ?? DEFAULT_REPORT_OUTPUT_CONFIG,
          agentConfig: state.agentConfig ?? DEFAULT_AGENT_CONFIG,
          modelConfigOverride: state.modelConfigOverride ?? {},
          serverDefaultsApplied: state.serverDefaultsApplied ?? false,
        };
      },
      partialize: (state) => ({
        apiBaseUrl: state.apiBaseUrl,
        selectedRunId: state.selectedRunId,
        selectedProjectId: state.selectedProjectId,
        profileId: state.profileId,
        profilePreferences: state.profilePreferences,
        memoryPolicyOverride: state.memoryPolicyOverride,
        executionMode: "deep",
        requirePlanApproval: state.requirePlanApproval,
        clarifierConfig: state.clarifierConfig,
        sourceSelection: state.sourceSelection,
        runInputAssets: state.runInputAssets,
        stagedRunAssets: state.stagedRunAssets,
        asyncSubmit: state.asyncSubmit,
        budget: state.budget,
        reportOutputConfig: state.reportOutputConfig,
        agentConfig: state.agentConfig,
        modelConfigOverride: state.modelConfigOverride,
        serverDefaultsApplied: state.serverDefaultsApplied,
      }),
    },
  ),
);
