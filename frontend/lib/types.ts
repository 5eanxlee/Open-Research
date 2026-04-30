export type RunStatus =
  | "queued"
  | "clarifying"
  | "awaiting_plan_approval"
  | "planning"
  | "researching"
  | "grounding"
  | "completed"
  | "failed"
  | "cancelled";

export type StreamStatus = "queued" | "running" | "completed" | "failed";
export type ExecutionMode = "standard" | "deep" | "hitl";
export type PlanningStage = "preview" | "execution" | "replan";
export type PlanApprovalStatus =
  | "not_required"
  | "pending_clarification"
  | "pending_approval"
  | "approved"
  | "rejected"
  | "changes_requested";
export type ApprovalDecisionKind = "approve" | "reject" | "request_changes";
export type PromptMode = "code" | "template" | "hybrid";
export type ResearchProfile = "balanced" | "official_first" | "wide_net";
export type RecencyPolicy = "auto" | "recent_first" | "evergreen";
export type AnswerStyle = "analyst" | "executive" | "technical";
export type CitationDiscipline = "strict" | "balanced";
export type ClaimGranularity = "atomic" | "balanced";
export type SourceTrustTier = "primary" | "high" | "standard" | "low" | "unknown";
export type ResearchAssetUsage = "reference_source" | "planning_context";
export type ResearchAssetType = "url" | "file";
export type AssetProcessingStatus = "pending" | "processing" | "ready" | "failed";
export type AssetExtractionMethod =
  | "url"
  | "text"
  | "json"
  | "csv"
  | "html"
  | "pdf_text"
  | "pdf_ocr"
  | "docx"
  | "image_ocr"
  | "unknown";
export type ContextPhase = "plan" | "replan" | "research" | "synthesize" | "ground";
export type ContextFragmentKind = "operational" | "preference" | "evidence_adjacent" | "system";
export type AssessmentKind = "source_fit" | "style_fit" | "operational" | "memory_usefulness";
export type AssessmentSource = "system" | "user";
export type StreamConnectionState =
  | "idle"
  | "connecting"
  | "replay"
  | "live"
  | "terminal"
  | "error";
export type RunConversationRole = "user" | "assistant";

export interface BudgetPolicy {
  max_streams: number;
  max_replans: number;
  max_queries_per_stream: number;
  max_results_per_query: number;
  max_sources_per_stream: number;
  per_domain_limit: number;
}

export interface RecommendedBudget extends BudgetPolicy {
  rationale_summary: string | null;
}

export interface BudgetRecommendationRationale {
  summary: string;
  coverage_axes: string[];
  evidence_gaps: string[];
  source_diversity_reasoning: string | null;
  grounding_difficulty: string | null;
}

export interface ModelConfig {
  lead_model: string;
  planner_model: string;
  worker_model: string;
  verifier_model: string;
  embedding_model: string;
  reranker_model: string;
}

export interface ModelConfigOverride {
  lead_model?: string | null;
  planner_model?: string | null;
  worker_model?: string | null;
  verifier_model?: string | null;
  embedding_model?: string | null;
  reranker_model?: string | null;
}

export interface AgentConfig {
  research_profile: ResearchProfile;
  recency_policy: RecencyPolicy;
  answer_style: AnswerStyle;
  citation_discipline: CitationDiscipline;
  claim_granularity: ClaimGranularity;
  source_trust_floor: SourceTrustTier;
  include_counterevidence: boolean;
}

export interface ClarifierConfig {
  enabled: boolean;
  max_questions: number;
  ambiguity_threshold: number;
  require_response_for_deep: boolean;
}

export interface ResearchStreamPlan {
  name: string;
  objective: string;
  queries: string[];
  model: string;
}

export interface PlanningDiscoveryRecord {
  query: string;
  provider?: string | null;
  result_count: number;
  titles: string[];
  urls: string[];
  summary?: string | null;
}

export interface PlanningArtifact {
  stage: PlanningStage;
  approved_preview_version?: number | null;
  task_breakdown?: string | null;
  table_of_contents: string[];
  constraints: string[];
  planned_deliverables: string[];
  key_questions: string[];
  available_documents: string[];
  discovery_queries: string[];
  discovery_records: PlanningDiscoveryRecord[];
  source_selection: string[];
  min_total_sources_retrieved: number;
  min_total_cited_sources: number;
  validation_checks: string[];
  validation_notes: string[];
}

export interface ResearchPlan {
  summary: string;
  hypothesis: string;
  streams: ResearchStreamPlan[];
  success_criteria: string[];
  planning_artifact?: PlanningArtifact | null;
  recommended_budget?: RecommendedBudget | null;
  budget_rationale?: BudgetRecommendationRationale | null;
  recommended_execution_mode?: ExecutionMode | null;
  approval_required?: boolean;
  complexity_factors?: string[];
}

export interface ClarificationQuestion {
  id: string;
  prompt: string;
  rationale: string;
  required: boolean;
  created_at: string;
}

export interface ClarificationTurn {
  question_id: string;
  prompt: string;
  response: string;
  created_at: string;
}

export interface ClarificationSession {
  status: PlanApprovalStatus;
  rationale: string;
  questions: ClarificationQuestion[];
  turns: ClarificationTurn[];
  iteration_count: number;
  resolved_question: string | null;
  created_at: string;
  updated_at: string;
}

export interface MemoryInfluencePolicy {
  enabled: boolean;
  retrieval_limit: number;
  planning_budget_tokens: number;
  research_budget_tokens: number;
  synthesis_budget_tokens: number;
  grounding_budget_tokens: number;
  allow_preference_in_planning: boolean;
  allow_preference_in_research: boolean;
  allow_preference_in_synthesis: boolean;
  allow_preference_in_grounding: boolean;
  stale_penalty: number;
  conflict_penalty: number;
}

export type MemoryPolicyNumericField =
  | "retrieval_limit"
  | "planning_budget_tokens"
  | "research_budget_tokens"
  | "synthesis_budget_tokens"
  | "grounding_budget_tokens"
  | "stale_penalty"
  | "conflict_penalty";

export interface ProfilePreferences {
  preferred_source_patterns: string[];
  avoided_source_patterns: string[];
  answer_style_bias: AnswerStyle | null;
  recency_bias: RecencyPolicy | null;
  source_trust_floor_bias: SourceTrustTier | null;
  include_counterevidence_bias: boolean | null;
  memory_policy: MemoryInfluencePolicy;
}

export interface ProfileRecord {
  profile_id: string;
  preferences: ProfilePreferences;
  created_at: string;
  updated_at: string;
}

export interface ProfileFeedback {
  profile_id: string;
  run_id: string | null;
  source_fit: number | null;
  style_fit: number | null;
  usefulness: number | null;
  correction: string | null;
  preferred_source_patterns: string[];
  avoided_source_patterns: string[];
  answer_style_bias: AnswerStyle | null;
  include_counterevidence_bias: boolean | null;
  created_at?: string;
}

export interface SourceCatalogEntry {
  id: string;
  name: string;
  description: string;
  backend_kind: string;
  default_enabled: boolean;
  configured: boolean;
  auth_required: boolean;
  supports_search: boolean;
  supports_fetch: boolean;
  supports_internal_docs: boolean;
  supports_recency: boolean;
  supports_primary_sources: boolean;
  supports_advanced_search: boolean;
  status_reason: string | null;
}

export interface AsyncJob {
  job_id: string;
  run_id: string;
  status: string;
  submission_mode: string;
  submitted_at: string;
  started_at: string | null;
  ended_at: string | null;
  last_heartbeat_at: string | null;
  owner_id: string | null;
  resume_cursor: number;
  metadata: Record<string, unknown>;
}

export interface ResearchInputAsset {
  source_type: ResearchAssetType;
  usage: ResearchAssetUsage;
  label: string;
  description?: string | null;
  url?: string | null;
  content_text?: string | null;
  content_type?: string | null;
  file_name?: string | null;
}

export interface ResearchAssetRecord extends ResearchInputAsset {
  id: string;
  project_id: string | null;
  run_id: string | null;
  extracted_text?: string | null;
  processing_status: AssetProcessingStatus;
  extraction_method: AssetExtractionMethod;
  ocr_used: boolean;
  page_count: number | null;
  file_size_bytes: number | null;
  sha256: string | null;
  warnings: string[];
  preview_excerpt: string | null;
  processing_error: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface StagedAssetRecord {
  id: string;
  source_type: ResearchAssetType;
  usage: ResearchAssetUsage;
  label: string;
  description?: string | null;
  url?: string | null;
  extracted_text?: string | null;
  content_type?: string | null;
  file_name?: string | null;
  processing_status: AssetProcessingStatus;
  extraction_method: AssetExtractionMethod;
  ocr_used: boolean;
  page_count: number | null;
  file_size_bytes: number | null;
  sha256: string | null;
  warnings: string[];
  preview_excerpt: string | null;
  processing_error: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectDetail extends ProjectSummary {
  assets: ResearchAssetRecord[];
  runs: RunSummary[];
}

export interface RunSummary {
  id: string;
  question: string;
  profile_id: string;
  project_id: string | null;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  estimated_cost_usd: number;
  final_report_markdown: string | null;
  error_message: string | null;
  worker_id: string | null;
  workflow_backend: string | null;
  last_heartbeat_at: string | null;
  terminal_reason: string | null;
  approval_status: PlanApprovalStatus | null;
  job_id: string | null;
}

export interface ResearchStreamView {
  id: string;
  name: string;
  objective: string;
  model: string;
  status: StreamStatus;
  sources_examined: number;
  elapsed_ms: number;
  cost_so_far: number;
  confidence: number | null;
}

export interface CitationRecord {
  claim: string;
  support_label: string;
  source_id: string;
  source_title: string;
  source_url: string;
  citation_key: string | null;
  passage_index: number;
  quote: string;
  confidence: number;
}

export interface FinalReport {
  markdown: string;
  citations: CitationRecord[];
  unsupported_claims: string[];
  confidence: number;
}

export interface RunNoteRecord {
  id: string;
  stream_id: string;
  stream_name: string;
  stream_objective: string;
  source_id: string | null;
  source_title: string | null;
  source_url: string | null;
  source_kind: string | null;
  retrieval_method: string | null;
  trust_tier: SourceTrustTier | null;
  trust_rationale: string | null;
  summary: string;
  key_facts: string[];
  open_questions: string[];
  confidence: number;
}

export interface PassageInspectionRecord {
  source_id: string;
  source_title: string;
  source_url: string;
  passage_index: number;
  text: string;
  search_document: string | null;
  start_offset: number;
  end_offset: number;
  token_count: number;
  source_kind: string | null;
  retrieval_method: string | null;
  trust_tier: SourceTrustTier | null;
  trust_rationale: string | null;
}

export interface ArtifactRecord {
  id: string;
  run_id: string;
  source_id: string;
  kind: string;
  uri: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
}

export interface CitationAuditRecord {
  id: string;
  run_id: string;
  section_title: string;
  ordinal: number;
  claim: string;
  decision: "kept" | "removed";
  reasons: string[];
  source_id: string | null;
  citation_key: string | null;
  source_url: string | null;
  normalized_url: string | null;
  matched_strategy: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface SourceRegistryEntry {
  id: string;
  run_id: string;
  source_id: string | null;
  asset_id: string | null;
  asset_origin: string | null;
  user_supplied: boolean;
  url: string;
  canonical_url: string;
  normalized_url: string;
  citation_key: string | null;
  title: string | null;
  provider: string | null;
  discovered_via: string;
  survived_final_citation: boolean | null;
  removed_in_audit: boolean | null;
  audit_reasons: string[];
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface RunEvent {
  id: number;
  run_id: string;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface RunConversationMessage {
  id: string;
  run_id: string;
  role: RunConversationRole;
  content: string;
  model?: string | null;
  references: string[];
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface RunConversationReply {
  user_message: RunConversationMessage;
  assistant_message: RunConversationMessage;
}

export interface ContextFragment {
  id: string;
  kind: ContextFragmentKind;
  phase: ContextPhase;
  memory_id: string | null;
  title: string;
  content: string;
  token_estimate: number;
  score: number;
  freshness_score: number;
  trust_score: number;
  selected_reason: string;
  dropped_reason: string | null;
  metadata: Record<string, unknown>;
}

export interface ContextPack {
  id: string;
  run_id: string;
  profile_id: string | null;
  phase: ContextPhase;
  summary: string;
  token_budget: number;
  used_tokens: number;
  fragments: ContextFragment[];
  dropped_fragments: ContextFragment[];
  created_at: string;
}

export interface BehaviorAssessment {
  id: string;
  run_id: string;
  profile_id: string | null;
  kind: AssessmentKind;
  source: AssessmentSource;
  score: number;
  rationale: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ApprovalDecision {
  decision: ApprovalDecisionKind;
  note: string | null;
  actor: string | null;
  created_at: string;
}

export interface PlanPreview {
  version: number;
  summary: string;
  hypothesis: string;
  plan: ResearchPlan;
  requested_budget: BudgetPolicy;
  recommended_budget: RecommendedBudget | null;
  effective_budget: BudgetPolicy;
  budget_decision_reason: string;
  approval_required: boolean;
  recommended_execution_mode: ExecutionMode;
  source_selection: string[];
  clarification_summary: string | null;
  created_at: string;
}

export interface RunDetail extends RunSummary {
  budget: BudgetPolicy | null;
  requested_budget: BudgetPolicy | null;
  recommended_budget: RecommendedBudget | null;
  effective_budget: BudgetPolicy | null;
  budget_decision_reason: string | null;
  agent_config: AgentConfig | null;
  metadata: Record<string, unknown>;
  cancel_requested: boolean;
  execution_mode: ExecutionMode;
  source_selection: string[];
  input_assets: ResearchAssetRecord[];
  project_assets_used: ResearchAssetRecord[];
  run_assets_used: ResearchAssetRecord[];
  planning_assets_used: ResearchAssetRecord[];
  reference_assets_used: ResearchAssetRecord[];
  asset_processing_errors: string[];
  clarification_session: ClarificationSession | null;
  plan_preview: PlanPreview | null;
  latest_approval_decision: ApprovalDecision | null;
  source_registry_entries: SourceRegistryEntry[];
  job: AsyncJob | null;
  streams: ResearchStreamView[];
  latest_plan: ResearchPlan | null;
  final_report: FinalReport | null;
  artifacts: ArtifactRecord[];
  citation_audits: CitationAuditRecord[];
  active_context_pack_ids: string[];
  behavior_assessments: BehaviorAssessment[];
  conversation_messages: RunConversationMessage[];
  events: RunEvent[];
}

export type WorkspacePhaseKey =
  | "intake"
  | "clarify"
  | "plan"
  | "execute"
  | "ground"
  | "audit"
  | "deliver";

export type WorkspacePhaseStatus =
  | "idle"
  | "active"
  | "blocked"
  | "complete"
  | "failed";

export type WorkspaceDecisionCategory =
  | "planning"
  | "clarification"
  | "stream_launch"
  | "replan"
  | "source_selection"
  | "verification"
  | "claim_repair"
  | "audit";

export type WorkspaceSourceOrigin =
  | "project_corpus"
  | "run_attachment"
  | "user_reference"
  | "web_discovered";

export type WorkspaceSourceState =
  | "discovered"
  | "fetched"
  | "chunked"
  | "retrieved"
  | "cited"
  | "removed";

export type WorkspaceConnectionTransport =
  | "idle"
  | "connecting"
  | "replay"
  | "live"
  | "terminal"
  | "error";

export interface WorkspacePhaseState {
  key: WorkspacePhaseKey;
  label: string;
  status: WorkspacePhaseStatus;
  started_at: string | null;
  completed_at: string | null;
  blocked_reason: string | null;
  event_count: number;
}

export interface WorkspaceTaskView {
  id: string;
  stream_id: string | null;
  stream_name: string | null;
  task_type: string;
  objective: string;
  status: string;
  query_count: number;
  selected_source_count: number;
  notes_produced: number;
  elapsed_ms: number | null;
  started_at: string | null;
  completed_at: string | null;
  next_action: string | null;
  blocker_reason: string | null;
  latest_sources: string[];
  latest_note_summary: string | null;
  last_tool_call: string | null;
  last_decision: string | null;
  metadata: Record<string, unknown>;
}

export interface WorkspaceStreamView {
  id: string;
  name: string;
  objective: string;
  model: string;
  status: StreamStatus;
  query_count: number;
  selected_source_count: number;
  note_count: number;
  latest_source_titles: string[];
  latest_note_summary: string | null;
  tasks: WorkspaceTaskView[];
  confidence: number | null;
  elapsed_ms: number | null;
  cost_so_far: number;
}

export interface WorkspaceDecisionView {
  id: string;
  category: WorkspaceDecisionCategory;
  title: string;
  rationale: string;
  affected_stream_id: string | null;
  affected_stream_name: string | null;
  affected_section: string | null;
  affected_claim: string | null;
  confidence: number | null;
  uncertainty: string | null;
  supporting_evidence_count: number;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface WorkspaceSourceView {
  id: string;
  source_id: string | null;
  asset_id: string | null;
  title: string | null;
  url: string;
  origin: WorkspaceSourceOrigin;
  state: WorkspaceSourceState;
  provider: string | null;
  trust_tier: SourceTrustTier | null;
  stream_ids: string[];
  stream_names: string[];
  report_sections: string[];
  citation_status: string | null;
  survived_final_citation: boolean | null;
  removed_in_audit: boolean | null;
  audit_reasons: string[];
  note_summaries: string[];
  passages_used: number;
  metadata: Record<string, unknown>;
}

export interface WorkspaceCitationView {
  id: string;
  section_title: string;
  claim: string;
  status: string;
  source_id: string | null;
  source_title: string | null;
  source_url: string | null;
  citation_key: string | null;
  support_label: string | null;
  quote: string | null;
  confidence: number | null;
  trust_tier: SourceTrustTier | null;
  audit_status: "kept" | "removed" | null;
  audit_reasons: string[];
  metadata: Record<string, unknown>;
}

export interface WorkspaceReportClaimView {
  section_title: string;
  ordinal: number;
  claim: string;
  support_label: string | null;
  confidence: number | null;
  citation_count: number;
  removed_citation_count: number;
  claim_repair_ran: boolean;
}

export interface WorkspaceReportSectionView {
  id: string;
  title: string;
  body_markdown: string;
  draft_status: string;
  grounded_claim_count: number;
  unsupported_claim_count: number;
  citation_count: number;
  removed_citation_count: number;
  claim_repair_count: number;
  claims: WorkspaceReportClaimView[];
}

export interface WorkspaceConnectionState {
  transport: WorkspaceConnectionTransport;
  stream_mode: string | null;
  backend_mode: string | null;
  workflow_backend: string | null;
  last_event_id: number;
  event_count: number;
  replay_lag: number;
  reconnect_state: string | null;
  last_event_at: string | null;
}

export interface WorkspacePlanView {
  clarification_session: ClarificationSession | null;
  plan_preview: PlanPreview | null;
  approved_plan: ResearchPlan | null;
  requested_budget: BudgetPolicy | null;
  recommended_budget: RecommendedBudget | null;
  effective_budget: BudgetPolicy | null;
  budget_decision_reason: string | null;
  planning_assets: ResearchAssetRecord[];
  project_assets: ResearchAssetRecord[];
  reference_assets: ResearchAssetRecord[];
  approval_history: ApprovalDecision[];
}

export interface RunWorkspaceSnapshot {
  run_id: string;
  question: string;
  project_id: string | null;
  status: RunStatus;
  execution_mode: ExecutionMode;
  approval_status: PlanApprovalStatus | null;
  current_phase: WorkspacePhaseKey;
  phases: WorkspacePhaseState[];
  plan: WorkspacePlanView;
  streams: WorkspaceStreamView[];
  decisions: WorkspaceDecisionView[];
  sources: WorkspaceSourceView[];
  citations: WorkspaceCitationView[];
  report_sections: WorkspaceReportSectionView[];
  connection: WorkspaceConnectionState;
  source_selection: string[];
  project_assets_available: ResearchAssetRecord[];
  run_assets_available: ResearchAssetRecord[];
  asset_processing_errors: string[];
  final_report_markdown: string | null;
  estimated_cost_usd: number;
  created_at: string;
  updated_at: string;
  generated_at: string;
  job: AsyncJob | null;
}

export interface RunWorkspaceDelta {
  run_id: string;
  event_id: number;
  changed: string[];
  reason: string | null;
  generated_at: string;
}

export interface ToolCatalogEntry {
  name: string;
  display_name: string;
  category: string;
  owner: string;
  description: string;
  enabled: boolean;
  backend: string | null;
  provider: string | null;
  budget_categories: string[];
  per_run_limit: number | null;
  risk: string;
  requires_auth: boolean;
  failure_mode: string;
}

export interface PublicRuntimeConfig {
  app_name: string;
  environment: string;
  prompt_profile_version: string;
  source_trust_policy_version: string;
  default_budget: BudgetPolicy;
  default_agent_config: AgentConfig;
  backends: Record<string, string>;
  models: ModelConfig;
  prompt_mode: PromptMode;
  available_sources: SourceCatalogEntry[];
  default_source_selection: string[];
  tool_catalog: ToolCatalogEntry[];
  capabilities: RuntimeCapabilities;
}

export interface NumericRange {
  min: number;
  max: number;
}

export interface RuntimeCapabilities {
  supports_temporal?: boolean;
  supports_artifacts?: boolean;
  supports_embeddings?: boolean;
  supports_metrics?: boolean;
  supports_prompt_profiles?: boolean;
  supports_replayable_sse?: boolean;
  supports_profiles?: boolean;
  supports_memory_harness?: boolean;
  supports_behavior_assessment?: boolean;
  supports_deep_approval?: boolean;
  supports_async_jobs?: boolean;
  supports_source_registry?: boolean;
  supports_debug_console?: boolean;
  supports_projects?: boolean;
  asset_upload_limits?: {
    max_file_size_bytes: number;
    max_files_per_batch: number;
    max_ocr_pdf_pages: number;
  };
  budget_limits?: Partial<Record<keyof BudgetPolicy, NumericRange>>;
  memory_policy_limits?: Partial<Record<MemoryPolicyNumericField, NumericRange>>;
  custom_responses_contract?: {
    endpoint: string;
    tool_names: string[];
    enabled_tool_names?: string[];
    completion_gate?: Record<string, number>;
    planner_discovery?: Record<string, number>;
  };
  [key: string]: unknown;
}

export interface CreateRunPayload {
  question: string;
  budget: BudgetPolicy;
  agent_config: AgentConfig;
  model_config_override?: ModelConfigOverride | null;
  profile_id?: string;
  project_id?: string | null;
  memory_policy_override?: MemoryInfluencePolicy | null;
  execution_mode?: ExecutionMode;
  require_plan_approval?: boolean | null;
  clarifier_config?: ClarifierConfig | null;
  source_selection?: string[] | null;
  input_assets?: ResearchInputAsset[];
  staged_asset_ids?: string[];
  async_submit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface StreamEnvelope {
  id: number | null;
  run_id: string | null;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}
