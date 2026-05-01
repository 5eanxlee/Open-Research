import type {
  AgentConfig,
  BudgetPolicy,
  MemoryInfluencePolicy,
  ProfilePreferences,
  ReportOutputConfig,
} from "./types";

export const DEFAULT_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8010";

export const DEFAULT_BUDGET: BudgetPolicy = {
  max_streams: 30,
  max_replans: 1,
  max_queries_per_stream: 25,
  max_results_per_query: 5,
  max_sources_per_stream: 3,
  per_domain_limit: 2,
};

export const DEFAULT_AGENT_CONFIG: AgentConfig = {
  research_profile: "balanced",
  recency_policy: "auto",
  answer_style: "analyst",
  citation_discipline: "strict",
  claim_granularity: "atomic",
  source_trust_floor: "standard",
  include_counterevidence: true,
};

export const DEFAULT_REPORT_OUTPUT_CONFIG: ReportOutputConfig = {
  min_words: 900,
  max_words: 2400,
};

export const DEFAULT_MEMORY_POLICY: MemoryInfluencePolicy = {
  enabled: true,
  retrieval_limit: 10,
  planning_budget_tokens: 1200,
  research_budget_tokens: 450,
  synthesis_budget_tokens: 900,
  grounding_budget_tokens: 0,
  allow_preference_in_planning: true,
  allow_preference_in_research: true,
  allow_preference_in_synthesis: true,
  allow_preference_in_grounding: false,
  stale_penalty: 0.2,
  conflict_penalty: 0.4,
};

export const DEFAULT_PROFILE_PREFERENCES: ProfilePreferences = {
  preferred_source_patterns: [],
  avoided_source_patterns: [],
  answer_style_bias: null,
  recency_bias: null,
  source_trust_floor_bias: null,
  include_counterevidence_bias: null,
  memory_policy: DEFAULT_MEMORY_POLICY,
};
