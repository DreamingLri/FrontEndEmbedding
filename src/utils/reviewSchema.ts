export type ReviewGuideOption = {
  value: string
  label: string
  summary: string
  detail: string
  examples?: string[]
}

export type ReviewGuideSection = {
  key: string
  label: string
  description: string
  options: ReviewGuideOption[]
}

export type CandidateKpidOption = {
  value: string
  label: string
  isEvidence: boolean
}

export type ReviewDocKp = {
  kpid: string
  text: string
  isEvidence: boolean
  isSuggestedMain: boolean
  isSelectedMain: boolean
  isSelectedSupport: boolean
}

export type ReviewBundleItem = {
  id: string
  sourceSeedId: string
  sourceDataset: string
  query: string
  queryType: string
  themeFamily: string
  targetTitle: string
  article: {
    otid: string
    title: string
    publishTime: string
    link: string
    excerpt: string
    fullText: string
  }
  ai: {
    priority: string
    uniquenessRisk: string
    reviewLane: string
    reviewGuidance: string
    generationMode: string
    seedFailureRisks: string[]
    seedBadCaseQueries: string[]
    evidenceText: string
    flagLowSpecificity: boolean | null
    flagGenericQuery: boolean | null
  }
  p0: {
    included: boolean
    reason: string
  }
  suggested: {
    expectedOtid: string
    expectedKpid: string
    supportKpids: string[]
    queryScope: string
    preferredGranularity: string
    supportPattern: string
    granularitySensitive: boolean | null
  }
  decisionDraft: ReviewDecision
  docKps: ReviewDocKp[]
  evidenceKpids: string[]
  candidateKpidOptions: CandidateKpidOption[]
}

export type ReviewBundleSource = {
  reviewSheetFile?: string
  p0File?: string
  articleFile?: string
  kpFile?: string
  bundleFile?: string
  publicBundleFile?: string
  defaultDraftFile?: string
  [key: string]: string | undefined
}

export type ReviewBundle = {
  bundleId: string
  generatedAt: string
  source: ReviewBundleSource
  summary: {
    totalItems: number
    p0Items: number
  }
  items: ReviewBundleItem[]
}

export type ReviewDecision = {
  finalExpectedOtid: string
  finalExpectedKpid: string
  finalSupportKpids: string[]
  otidStatus: string
  kpidStatus: string
  finalQueryScope: string
  finalPreferredGranularity: string
  finalSupportPattern: string
  finalGranularitySensitive: boolean | null
  includeInFormalSet: string
  reviewRound: string
  reviewer: string
  notes: string
}

export const REVIEW_STATUS_OPTIONS: ReviewGuideOption[] = [
  {
    value: '待复核',
    label: '待复核',
    summary: '还没有人工做最终判断。',
    detail: '用于刚进入工作台、还没完成确认的条目。',
  },
  {
    value: '已确认',
    label: '已确认',
    summary: '你已经确认这项判断是正确的。',
    detail: '常用于 otid 或 kpid 已经核实无误的情况。',
  },
  {
    value: '需修改',
    label: '需修改',
    summary: 'AI 给出的建议不准确，需要你改写。',
    detail: '例如建议 kpid 不对、需要改到别的主支撑点。',
  },
  {
    value: '存疑',
    label: '存疑',
    summary: '目前还不够稳，先保留疑问。',
    detail: '适合暂时拿不准、需要回头二次确认的条目。',
  },
]

export const INCLUDE_OPTIONS: ReviewGuideOption[] = [
  {
    value: '待定',
    label: '待定',
    summary: '先保留，等后续统一裁剪。',
    detail: '适合第一轮复核后还没决定是否进正式集的条目。',
  },
  {
    value: '是',
    label: '纳入正式集',
    summary: '这条样本足够稳定，建议保留。',
    detail: '通常用于粒度敏感、证据边界清楚的样本。',
  },
  {
    value: '否',
    label: '不纳入',
    summary: '这条样本不适合进入最终挑战集。',
    detail: '常见原因包括歧义大、重复度高、粒度敏感性不足。',
  },
]

export const BOOLEAN_OPTIONS: ReviewGuideOption[] = [
  {
    value: 'true',
    label: 'true',
    summary: '这条样本能明显区分粒度差异。',
    detail: '例如单点事实明显偏 KP，整体概述明显偏 OT，或者需要多 KP 联合。',
  },
  {
    value: 'false',
    label: 'false',
    summary: '这条样本不太能体现粒度差异。',
    detail: '如果不管 KP 还是 OT 都差不多能答，就更适合标为 false。',
  },
]

export const QUERY_TYPE_OPTIONS: ReviewGuideOption[] = [
  {
    value: 'standard',
    label: 'standard',
    summary: '正常完整问句。',
    detail: '最常见，语义明确，接近日常提问但不特别口语化。',
    examples: ['面试形式是什么', '申请答辩需要满足哪些条件'],
  },
  {
    value: 'short_keyword',
    label: 'short_keyword',
    summary: '关键词式短查询。',
    detail: '通常没有完整句法，像搜索框里直接输入的一串关键词。',
    examples: ['2025 录取通知书 地址核对 截止'],
  },
  {
    value: 'situational',
    label: 'situational',
    summary: '带身份、背景或假设条件的问句。',
    detail: '往往包含“我是……”“如果我……”这类上下文描述。',
    examples: ['如果我错过报到时间，会有什么后果'],
  },
]

export const QUERY_SCOPE_OPTIONS: ReviewGuideOption[] = [
  {
    value: 'fact_detail',
    label: 'fact_detail',
    summary: '问题在问一个具体事实点。',
    detail: '常见于网址、电话、面试形式、材料名称等单点信息。判断口诀：答案能否压缩成一句具体信息。',
    examples: ['平台网址是什么', '面试形式是什么'],
  },
  {
    value: 'procedure',
    label: 'procedure',
    summary: '问题在问步骤、顺序或办理流程。',
    detail: '常见于“先做什么、后做什么、要不要先打印再报到”。判断口诀：答案是否天然带先后关系。',
    examples: ['申请答辩具体怎么走', '是否需要先打印准考证再报到'],
  },
  {
    value: 'policy_overview',
    label: 'policy_overview',
    summary: '问题在问整体要求、总体措施或政策概览。',
    detail: '通常要概括多个段落，不能只靠单点信息。判断口诀：答案是否必须总结全文结构。',
    examples: ['总体要求和关键措施有哪些'],
  },
  {
    value: 'time_location',
    label: 'time_location',
    summary: '问题核心是时间或地点。',
    detail: '如果去掉时间地点后问题价值明显下降，通常就是这类。',
    examples: ['地址核对截止', '报到地点在哪里'],
  },
  {
    value: 'eligibility_condition',
    label: 'eligibility_condition',
    summary: '问题在问是否符合条件、需要满足哪些条件。',
    detail: '答案核心是“要满足什么”而不是“怎么做”。',
    examples: ['申请答辩需要满足哪些条件'],
  },
]

export const PREFERRED_GRANULARITY_OPTIONS: ReviewGuideOption[] = [
  {
    value: 'KP',
    label: 'KP',
    summary: '最适合命中 query 的是单个局部知识点。',
    detail: '这是检索层标签。适合单点事实、局部条件、局部细节。',
    examples: ['面试形式是什么', '地址核对截止'],
  },
  {
    value: 'OT',
    label: 'OT',
    summary: '最适合命中 query 的是整篇通知或跨段上下文。',
    detail: '这是检索层标签。适合整体要求、总体措施、全文概览。',
    examples: ['总体要求和关键措施有哪些'],
  },
  {
    value: 'Q',
    label: 'Q',
    summary: '主要难点来自问句表达本身。',
    detail: '很少用。只有当问法本身造成检索难点，而不明显依赖 KP 或 OT 时才使用。',
  },
  {
    value: 'Q+KP',
    label: 'Q+KP',
    summary: '问句表达影响召回，最直接证据仍落在局部 KP。',
    detail: '适合自然问法明显，但答案本质还是单点局部证据的情况。',
  },
  {
    value: 'KP+OT',
    label: 'KP+OT',
    summary: '局部条目和全文上下文都重要。',
    detail: '常见于局部事实要靠 KP 命中，但完整理解还要补 OT 约束或上下文。',
  },
  {
    value: 'Q+OT',
    label: 'Q+OT',
    summary: '问句表达影响召回，且答案依赖全文级理解。',
    detail: '应比 KP+OT 更少见。',
  },
]

export const SUPPORT_PATTERN_OPTIONS: ReviewGuideOption[] = [
  {
    value: 'single_kp',
    label: 'single_kp',
    summary: '一条 KP 就足以回答核心问题。',
    detail: '这是证据层标签。常见于单点事实、单一条件、单一材料。',
    examples: ['面试形式是什么'],
  },
  {
    value: 'multi_kp',
    label: 'multi_kp',
    summary: '至少两条 KP 联合才能完整回答。',
    detail: '这是证据层标签。适合需要两个或以上局部信息拼起来的情况。',
    examples: ['是否要先打印准考证再按时报到'],
  },
  {
    value: 'ot_required',
    label: 'ot_required',
    summary: '必须依赖整篇通知或跨段整合。',
    detail: '如果少量 KP 摘录都不足以稳定回答，通常就是 ot_required。',
    examples: ['总体要求和关键措施有哪些'],
  },
]

export const GUIDE_SECTIONS: ReviewGuideSection[] = [
  {
    key: 'query_scope',
    label: 'Query Scope',
    description: '回答“这个问题到底在问哪一类信息”。',
    options: QUERY_SCOPE_OPTIONS,
  },
  {
    key: 'preferred_granularity',
    label: 'Preferred Granularity',
    description: '检索层标签，回答“最适合检出它的表示粒度是什么”。',
    options: PREFERRED_GRANULARITY_OPTIONS,
  },
  {
    key: 'support_pattern',
    label: 'Support Pattern',
    description: '证据层标签，回答“真正回答时证据如何支撑答案”。',
    options: SUPPORT_PATTERN_OPTIONS,
  },
  {
    key: 'query_type',
    label: 'Query Type',
    description: '问题的表述形式。',
    options: QUERY_TYPE_OPTIONS,
  },
]

export const GUIDE_SECTION_MAP = Object.fromEntries(
  GUIDE_SECTIONS.map(section => [section.key, section]),
)

export function boolToFormValue(value: boolean | null | undefined): string {
  if (value === true) return 'true'
  if (value === false) return 'false'
  return ''
}

export function formValueToBool(value: string): boolean | null {
  if (value === 'true') return true
  if (value === 'false') return false
  return null
}

export function cloneDecision(decision: ReviewDecision): ReviewDecision {
  return {
    finalExpectedOtid: decision.finalExpectedOtid ?? '',
    finalExpectedKpid: decision.finalExpectedKpid ?? '',
    finalSupportKpids: Array.isArray(decision.finalSupportKpids)
      ? [...decision.finalSupportKpids]
      : [],
    otidStatus: decision.otidStatus ?? '待复核',
    kpidStatus: decision.kpidStatus ?? '待复核',
    finalQueryScope: decision.finalQueryScope ?? '',
    finalPreferredGranularity: decision.finalPreferredGranularity ?? '',
    finalSupportPattern: decision.finalSupportPattern ?? '',
    finalGranularitySensitive: decision.finalGranularitySensitive ?? null,
    includeInFormalSet: decision.includeInFormalSet ?? '待定',
    reviewRound: decision.reviewRound ?? 'ui_workbench',
    reviewer: decision.reviewer ?? '',
    notes: decision.notes ?? '',
  }
}
