"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { promoteRunAsset } from "@/lib/api";
import type {
  ArtifactRecord,
  BehaviorAssessment,
  CitationAuditRecord,
  ContextPack,
  PublicRuntimeConfig,
  RunDetail,
} from "@/lib/types";
import { useResearchStore } from "@/store/use-research-store";

interface RunInspectorProps {
  detail: RunDetail | undefined;
  artifacts: ArtifactRecord[] | undefined;
  audits: CitationAuditRecord[] | undefined;
  contextPacks: ContextPack[] | undefined;
  assessments: BehaviorAssessment[] | undefined;
  publicConfig: PublicRuntimeConfig | undefined;
}

export function RunInspector({
  detail,
  artifacts,
  audits,
  contextPacks,
  assessments,
  publicConfig,
}: RunInspectorProps) {
  const queryClient = useQueryClient();
  const apiBaseUrl = useResearchStore((state) => state.apiBaseUrl);
  const appliedModels = (() => {
    const raw = detail?.metadata?.model_config;
    if (
      raw &&
      typeof raw === "object" &&
      !Array.isArray(raw) &&
      "lead_model" in raw &&
      "planner_model" in raw &&
      "worker_model" in raw &&
      "verifier_model" in raw
    ) {
      return raw as {
        lead_model: string;
        planner_model: string;
        worker_model: string;
        verifier_model: string;
      };
    }
    return publicConfig?.models ?? null;
  })();
  const promoteMutation = useMutation({
    mutationFn: ({ runId, assetId, projectId }: { runId: string; assetId: string; projectId: string }) =>
      promoteRunAsset(apiBaseUrl, runId, assetId, projectId),
    onSuccess: async (_, variables) => {
      await queryClient.invalidateQueries({
        queryKey: ["run-detail", apiBaseUrl, variables.runId],
      });
      if (variables.projectId) {
        await queryClient.invalidateQueries({
          queryKey: ["project-detail", apiBaseUrl, variables.projectId],
        });
      }
    },
  });

  const renderAssetList = (
    title: string,
    assets:
      | RunDetail["project_assets_used"]
      | RunDetail["run_assets_used"]
      | RunDetail["planning_assets_used"]
      | RunDetail["reference_assets_used"],
    options: { promotable?: boolean } = {},
  ) => (
    <div className="inspector-section">
      <div className="section-heading-row">
        <h3>{title}</h3>
        <span className="pill muted">{assets.length}</span>
      </div>
      <div className="scroll-list">
        {assets.map((asset) => (
          <div className="list-card" key={asset.id}>
            <div className="audit-row">
              <strong>{asset.label}</strong>
              <span className="pill muted">{asset.usage.replaceAll("_", " ")}</span>
            </div>
            <span className="muted-text">
              {asset.project_id ? "Project corpus" : "Run attachment"} · {asset.source_type}
            </span>
            <span className="muted-text">
              {asset.processing_status.replaceAll("_", " ")} ·{" "}
              {asset.extraction_method.replaceAll("_", " ")}
              {asset.ocr_used ? " · OCR" : ""}
            </span>
            {asset.url ? <code>{asset.url}</code> : null}
            {asset.preview_excerpt ? <span>{asset.preview_excerpt}</span> : null}
            {asset.processing_error ? (
              <span className="danger-text">{asset.processing_error}</span>
            ) : null}
            {asset.warnings.length ? (
              <span className="muted-text">Warnings: {asset.warnings.join(", ")}</span>
            ) : null}
            {options.promotable && !asset.project_id && detail?.project_id ? (
              <div className="button-row">
                <button
                  className="secondary-button"
                  onClick={() =>
                    promoteMutation.mutate({
                      runId: detail.id,
                      assetId: asset.id,
                      projectId: detail.project_id!,
                    })
                  }
                  type="button"
                  disabled={promoteMutation.isPending}
                >
                  Promote to project corpus
                </button>
              </div>
            ) : null}
          </div>
        ))}
        {assets.length === 0 ? <span className="muted-text">No assets in this section.</span> : null}
      </div>
    </div>
  );

  return (
    <section className="panel inspector-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Evidence</p>
          <h2 className="panel-title">Run inspector</h2>
        </div>
      </div>

      {!detail ? (
        <div className="empty-state compact">
          <p>Select a run.</p>
          <span>Artifacts, citation audits, and applied settings appear here.</span>
        </div>
      ) : (
        <div className="inspector-stack">
          <div className="inspector-section">
            <h3>Applied policy</h3>
            <dl className="definition-list">
              <div>
                <dt>Profile ID</dt>
                <dd>{detail.profile_id}</dd>
              </div>
              <div>
                <dt>Prompt profile</dt>
                <dd>
                  {publicConfig?.prompt_profile_version ??
                    String(detail.metadata.prompt_profile_version ?? "unknown")}
                </dd>
              </div>
              <div>
                <dt>Research profile</dt>
                <dd>{detail.agent_config?.research_profile ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Answer style</dt>
                <dd>{detail.agent_config?.answer_style ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Trust floor</dt>
                <dd>{detail.agent_config?.source_trust_floor ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Budget</dt>
                <dd>
                  {detail.budget
                    ? `${detail.budget.max_streams} streams / ${detail.budget.max_sources_per_stream} sources`
                    : "unknown"}
                </dd>
              </div>
              <div>
                <dt>Execution mode</dt>
                <dd>{detail.execution_mode}</dd>
              </div>
              <div>
                <dt>Approval</dt>
                <dd>{detail.approval_status ?? "not_required"}</dd>
              </div>
              <div>
                <dt>Prompt mode</dt>
                <dd>{publicConfig?.prompt_mode ?? "code"}</dd>
              </div>
              <div>
                <dt>Lead model</dt>
                <dd>{appliedModels?.lead_model ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Planner model</dt>
                <dd>{appliedModels?.planner_model ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Worker model</dt>
                <dd>{appliedModels?.worker_model ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Verifier model</dt>
                <dd>{appliedModels?.verifier_model ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Source selection</dt>
                <dd>
                  {detail.source_selection.length > 0
                    ? detail.source_selection.join(", ")
                    : "deployment default"}
                </dd>
              </div>
            </dl>
          </div>

          <div className="inspector-section">
            <h3>Budget provenance</h3>
            <dl className="definition-list">
              <div>
                <dt>Requested</dt>
                <dd>
                  {detail.requested_budget
                    ? `${detail.requested_budget.max_streams} streams / ${detail.requested_budget.max_queries_per_stream} queries`
                    : "n/a"}
                </dd>
              </div>
              <div>
                <dt>Recommended</dt>
                <dd>
                  {detail.recommended_budget
                    ? `${detail.recommended_budget.max_streams} streams / ${detail.recommended_budget.max_queries_per_stream} queries`
                    : "n/a"}
                </dd>
              </div>
              <div>
                <dt>Effective</dt>
                <dd>
                  {detail.effective_budget
                    ? `${detail.effective_budget.max_streams} streams / ${detail.effective_budget.max_queries_per_stream} queries`
                    : "n/a"}
                </dd>
              </div>
              <div>
                <dt>Decision reason</dt>
                <dd>{detail.budget_decision_reason ?? "n/a"}</dd>
              </div>
            </dl>
          </div>

          {detail.latest_plan ? (
            <div className="inspector-section">
              <div className="section-heading-row">
                <h3>Latest plan</h3>
                <span className="pill muted">{detail.latest_plan.streams.length} streams</span>
              </div>
              <details className="list-card" open>
                <summary className="audit-row">
                  <strong>{detail.latest_plan.summary}</strong>
                  <span>{detail.latest_plan.hypothesis}</span>
                </summary>
                <div className="inspector-detail-stack">
                  {detail.latest_plan.streams.map((stream) => (
                    <article className="inspector-inline-card" key={stream.name}>
                      <strong>{stream.name}</strong>
                      <span>{stream.objective}</span>
                      <span>{stream.queries.join(" · ")}</span>
                    </article>
                  ))}
                </div>
              </details>
            </div>
          ) : null}

          {detail.plan_preview ? (
            <div className="inspector-section">
              <div className="section-heading-row">
                <h3>Plan preview</h3>
                <span className="pill muted">{detail.plan_preview.recommended_execution_mode}</span>
              </div>
              <div className="list-card">
                <strong>{detail.plan_preview.summary}</strong>
                <span>{detail.plan_preview.hypothesis}</span>
                <span className="muted-text">
                  Effective budget: {detail.plan_preview.effective_budget.max_streams} streams /{" "}
                  {detail.plan_preview.effective_budget.max_queries_per_stream} queries /{" "}
                  {detail.plan_preview.effective_budget.max_sources_per_stream} sources
                </span>
                <span className="muted-text">{detail.plan_preview.budget_decision_reason}</span>
              </div>
            </div>
          ) : null}

          {detail.clarification_session ? (
            <div className="inspector-section">
              <div className="section-heading-row">
                <h3>Clarification</h3>
                <span className="pill muted">{detail.clarification_session.status}</span>
              </div>
              <div className="scroll-list">
                {detail.clarification_session.questions.map((question) => {
                  const turn = detail.clarification_session?.turns.find(
                    (entry) => entry.question_id === question.id,
                  );
                  return (
                    <div className="list-card" key={question.id}>
                      <strong>{question.prompt}</strong>
                      <span className="muted-text">{question.rationale}</span>
                      <span>{turn?.response ?? "Awaiting response"}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          {detail.job ? (
            <div className="inspector-section">
              <div className="section-heading-row">
                <h3>Async job</h3>
                <span className="pill muted">{detail.job.status}</span>
              </div>
              <dl className="definition-list">
                <div>
                  <dt>Job ID</dt>
                  <dd>{detail.job.job_id}</dd>
                </div>
                <div>
                  <dt>Submitted</dt>
                  <dd>{detail.job.submitted_at}</dd>
                </div>
                <div>
                  <dt>Last heartbeat</dt>
                  <dd>{detail.job.last_heartbeat_at ?? "n/a"}</dd>
                </div>
              </dl>
            </div>
          ) : null}

          {renderAssetList("Project corpus used", detail.project_assets_used)}
          {renderAssetList("Run-only assets used", detail.run_assets_used, { promotable: true })}
          {renderAssetList("Planning assets used", detail.planning_assets_used)}
          {renderAssetList("Reference assets used", detail.reference_assets_used)}

          {detail.asset_processing_errors.length ? (
            <div className="inspector-section">
              <div className="section-heading-row">
                <h3>Asset processing errors</h3>
                <span className="pill danger">{detail.asset_processing_errors.length}</span>
              </div>
              <div className="scroll-list">
                {detail.asset_processing_errors.map((error, index) => (
                  <div className="list-card" key={`${error}-${index}`}>
                    <span className="danger-text">{error}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="inspector-section">
            <div className="section-heading-row">
              <h3>Artifacts</h3>
              <span className="pill muted">{artifacts?.length ?? 0}</span>
            </div>
            <div className="scroll-list">
              {(artifacts ?? []).map((artifact) => (
                <div className="list-card" key={artifact.id}>
                  <strong>{artifact.kind}</strong>
                  <span>{artifact.content_type}</span>
                  <span>{Math.max(artifact.size_bytes / 1024, 0.1).toFixed(1)} KB</span>
                  <code>{artifact.uri}</code>
                </div>
              ))}
              {!artifacts?.length ? <span className="muted-text">No artifacts yet.</span> : null}
            </div>
          </div>

          <div className="inspector-section">
            <div className="section-heading-row">
              <h3>Source provenance</h3>
              <span className="pill muted">{detail.source_registry_entries.length}</span>
            </div>
            <div className="scroll-list">
              {detail.source_registry_entries.map((entry) => (
                <div className="list-card" key={entry.id}>
                  <div className="audit-row">
                    <strong>{entry.title ?? entry.normalized_url}</strong>
                    <span
                      className={`pill ${
                        entry.survived_final_citation
                          ? "success"
                          : entry.removed_in_audit
                            ? "danger"
                            : "muted"
                      }`}
                    >
                      {entry.survived_final_citation
                        ? "cited"
                        : entry.removed_in_audit
                          ? "removed"
                          : "uncited"}
                    </span>
                  </div>
                  <span>{entry.discovered_via}</span>
                  <span className="muted-text">
                    {entry.provider ?? "unknown provider"} ·{" "}
                    {entry.asset_origin
                      ? `${entry.asset_origin} asset`
                      : entry.user_supplied
                        ? "user supplied"
                        : "web discovered"}
                  </span>
                  {entry.asset_id ? <code>Asset: {entry.asset_id}</code> : null}
                  {entry.audit_reasons.length ? (
                    <span className="muted-text">Audit reasons: {entry.audit_reasons.join(", ")}</span>
                  ) : null}
                </div>
              ))}
              {detail.source_registry_entries.length === 0 ? (
                <span className="muted-text">No source registry entries yet.</span>
              ) : null}
            </div>
          </div>

          <div className="inspector-section">
            <div className="section-heading-row">
              <h3>Citation audit</h3>
              <span className="pill muted">{audits?.length ?? 0}</span>
            </div>
            <div className="scroll-list">
              {(audits ?? []).map((audit) => (
                <div className="list-card" key={audit.id}>
                  <div className="audit-row">
                    <span className={`pill ${audit.decision === "removed" ? "danger" : "success"}`}>
                      {audit.decision}
                    </span>
                    <span>
                      {audit.section_title} · {audit.ordinal}
                    </span>
                  </div>
                  <strong>{audit.claim}</strong>
                  {audit.reasons.length ? (
                    <span className="muted-text">Reasons: {audit.reasons.join(", ")}</span>
                  ) : null}
                </div>
              ))}
              {!audits?.length ? <span className="muted-text">No audit rows yet.</span> : null}
            </div>
          </div>

          <div className="inspector-section">
            <div className="section-heading-row">
              <h3>Context packs</h3>
              <span className="pill muted">{contextPacks?.length ?? 0}</span>
            </div>
            <div className="scroll-list">
              {(contextPacks ?? []).map((pack) => (
                <details className="list-card" key={pack.id}>
                  <summary className="audit-row">
                    <span className="pill success">{pack.phase}</span>
                    <span>
                      {pack.used_tokens} / {pack.token_budget} tokens
                    </span>
                  </summary>
                  <strong>{pack.summary}</strong>
                  <span className="muted-text">
                    {pack.fragments.length} selected · {pack.dropped_fragments.length} dropped
                  </span>
                  <div className="inspector-detail-stack">
                    {pack.fragments.map((fragment) => (
                      <details className="inspector-inline-card" key={fragment.id}>
                        <summary className="audit-row">
                          <strong>{fragment.title}</strong>
                          <span>{fragment.kind}</span>
                        </summary>
                        <p className="muted-text">{fragment.content}</p>
                        <span className="muted-text">Selected: {fragment.selected_reason}</span>
                      </details>
                    ))}
                    {pack.dropped_fragments.map((fragment) => (
                      <article className="inspector-inline-card" key={fragment.id}>
                        <strong>{fragment.title}</strong>
                        <span>Dropped: {fragment.dropped_reason ?? "unknown"}</span>
                      </article>
                    ))}
                  </div>
                </details>
              ))}
              {!contextPacks?.length ? <span className="muted-text">No context packs yet.</span> : null}
            </div>
          </div>

          <div className="inspector-section">
            <div className="section-heading-row">
              <h3>Behavior assessments</h3>
              <span className="pill muted">{assessments?.length ?? 0}</span>
            </div>
            <div className="scroll-list">
              {(assessments ?? []).map((assessment) => (
                <div className="list-card" key={assessment.id}>
                  <div className="audit-row">
                    <span className={`pill ${assessment.source === "user" ? "danger" : "success"}`}>
                      {assessment.source}
                    </span>
                    <span>{assessment.kind}</span>
                  </div>
                  <strong>{Math.round(assessment.score * 100)} / 100</strong>
                  <span className="muted-text">{assessment.rationale}</span>
                </div>
              ))}
              {!assessments?.length ? <span className="muted-text">No behavior assessments yet.</span> : null}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
