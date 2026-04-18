import type {
  PipelineBehavior,
  PipelineDecision,
  PipelineDocumentRecord,
  PipelineTrace,
  SearchPipelineResult,
} from '../worker/search_pipeline.ts'
import { FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET } from '../worker/search_pipeline.ts'
import type { RejectTier, SearchRejection } from '../worker/vector_engine.ts'

export type SearchTraceData = {
  query: string
  results: PipelineDocumentRecord[]
  retrievalDecision?: PipelineDecision | null
  decision?: PipelineDecision | null
  rejection?: SearchRejection | null
  weakResultsCount?: number
  querySignals?: PipelineTrace['querySignals'] | null
  retrievalSignals?: PipelineTrace['retrievalSignals'] | null
  stats?: {
    totalMs: string
    searchMs: string
    fetchMs: string
    rejected?: boolean
    rejection?: SearchRejection | null
    weakResultsCount?: number
  }
}

type RejectReason = SearchRejection['reason'] | null | undefined

export const ORIGINAL_SNIPPET_THRESHOLD =
  FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.bestSentenceThreshold

export function buildSearchTraceData(searchResult: SearchPipelineResult): SearchTraceData {
  const trace = searchResult.trace

  return {
    query: searchResult.query,
    results: searchResult.results,
    retrievalDecision: searchResult.retrievalDecision ?? null,
    stats: {
      totalMs: Number(trace?.totalMs ?? 0).toFixed(1),
      searchMs: Number(trace?.searchMs ?? 0).toFixed(1),
      fetchMs: Number(trace?.fetchMs ?? 0).toFixed(1),
      rejected: searchResult.finalDecision?.behavior === 'reject',
      rejection: searchResult.rejection ?? null,
      weakResultsCount: searchResult.weakResults.length,
    },
    decision: searchResult.finalDecision ?? null,
    rejection: searchResult.rejection ?? null,
    weakResultsCount: searchResult.weakResults.length,
    querySignals: trace?.querySignals ?? null,
    retrievalSignals: trace?.retrievalSignals ?? null,
  }
}

export function splitPrimaryResults(
  results: PipelineDocumentRecord[],
  heroLimit = 3,
  visibleLimit = 10,
) {
  return {
    heroResults: results.slice(0, heroLimit),
    compactResults: results.slice(heroLimit, visibleLimit),
  }
}

export function getBehaviorLabel(behavior?: PipelineBehavior | null): string {
  switch (behavior) {
    case 'answer':
      return '回答'
    case 'reject':
      return '拒答'
    default:
      return '未决'
  }
}

export function getBehaviorClass(behavior?: PipelineBehavior | null): string {
  switch (behavior) {
    case 'answer':
      return 'border-emerald-500/25 bg-emerald-500/10 text-emerald-200'
    case 'reject':
      return 'border-rose-500/25 bg-rose-500/10 text-rose-200'
    default:
      return 'border-white/10 bg-white/[0.03] text-slate-300'
  }
}

export function getRejectReasonLabel(reason?: RejectReason): string {
  switch (reason) {
    case 'low_topic_coverage':
      return '主题覆盖不足'
    case 'low_consistency':
      return '主题一致性不足'
    case 'invalid_input':
      return '输入无效'
    default:
      return ''
  }
}

export function getRejectTierLabel(tier?: RejectTier | null): string {
  switch (tier) {
    case 'hard_reject':
      return '硬拒答'
    case 'boundary_uncertain':
      return '边界不确定'
    case 'invalid_input':
      return '无效输入'
    default:
      return ''
  }
}

export function getRejectDescription(reason?: RejectReason): string {
  if (reason === 'low_topic_coverage') {
    return '当前知识库内没有形成同主题、可直接回答的稳定锚点，因此系统只保留弱相关入口，不直接给出答案。'
  }
  if (reason === 'low_consistency') {
    return '当前候选结果主题分散，或者证据只停留在弱相似层，系统拒绝输出不稳定答案。'
  }
  return '当前问题没有达到可安全展示的直接回答条件，系统进入拒答模式。'
}

export function getEmptyStateCopy(query: string) {
  return query.trim()
    ? {
        title: '当前问题未形成可信的直接答案',
        subtitle: '系统未能形成稳定且可直接展示的结果',
      }
    : {
        title: '输入关键词开始检索',
        subtitle: 'Top 1-3 展示原话，Top 4-10 展示紧凑列表',
      }
}

export function getDecisionSubtitle(traceData?: SearchTraceData | null): string {
  const decision = traceData?.decision
  if (!decision) {
    return '等待统一 pipeline 返回行为决策。'
  }

  if (decision.behavior === 'reject') {
    const rejectReason = getRejectReasonLabel(
      decision.rejectionReason ?? traceData?.rejection?.reason,
    )
    return rejectReason
      ? `系统选择拒答，主因是：${rejectReason}。`
      : '系统选择拒答，当前结果未达到稳定可展示条件。'
  }

  return '系统已进入直接回答链路，并返回抓取后的候选文档。'
}
