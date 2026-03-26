<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  FileUp,
  RefreshCcw,
} from 'lucide-vue-next'
import {
  BOOLEAN_OPTIONS,
  GUIDE_SECTIONS,
  GUIDE_SECTION_MAP,
  INCLUDE_OPTIONS,
  PREFERRED_GRANULARITY_OPTIONS,
  QUERY_SCOPE_OPTIONS,
  REVIEW_STATUS_OPTIONS,
  SUPPORT_PATTERN_OPTIONS,
  boolToFormValue,
  cloneDecision,
  formValueToBool,
  type ReviewBundle,
  type ReviewBundleItem,
  type ReviewDecision,
} from '../../utils/reviewSchema'

const DEFAULT_BUNDLE_PATHS = [
  '/review_bundle_latest.json',
  '/review_bundle_ai_preproduction_v2.json',
]

const bundle = ref<ReviewBundle | null>(null)
const decisions = ref<Record<string, ReviewDecision>>({})
const activeItemId = ref('')
const activeGuideKey = ref('query_scope')
const bundleLoadError = ref('')
const loadingDefaultBundle = ref(false)
const loadingServerDraft = ref(false)
const savingDraftToServer = ref(false)
const applyingToServer = ref(false)
const serverActionMessage = ref('')
const serverActionError = ref('')

const searchText = ref('')
const priorityFilter = ref('all')
const completionFilter = ref('pending')
const riskFilter = ref('all')
const p0Only = ref(true)
const reviewerDefault = ref('')

function storageKey(bundleId: string) {
  return `review-workbench:${bundleId}`
}

function normalizeSourcePath(value: string | undefined) {
  return typeof value === 'string' ? value.trim() : ''
}

function sanitizeBundleIdForFileName(bundleId: string) {
  const normalized = bundleId.trim().replace(/[^a-zA-Z0-9._-]+/g, '_')
  const cleaned = normalized.replace(/^[_\-.]+|[_\-.]+$/g, '')
  return cleaned || 'review_bundle'
}

function inferDraftPath(bundleData: ReviewBundle | null) {
  if (!bundleData) return ''

  const explicitDraftPath = normalizeSourcePath(bundleData.source.defaultDraftFile)
  if (explicitDraftPath) return explicitDraftPath

  const reviewSheetPath = normalizeSourcePath(bundleData.source.reviewSheetFile)
  if (!reviewSheetPath) return ''

  const normalizedReviewSheetPath = reviewSheetPath.replace(/\\/g, '/')
  const lastSlashIndex = normalizedReviewSheetPath.lastIndexOf('/')
  const directory = lastSlashIndex >= 0
    ? normalizedReviewSheetPath.slice(0, lastSlashIndex)
    : ''
  const reviewSheetFileName = lastSlashIndex >= 0
    ? normalizedReviewSheetPath.slice(lastSlashIndex + 1)
    : normalizedReviewSheetPath
  const fallbackBaseName = reviewSheetFileName.replace(/\.[^.]+$/, '') || 'review_sheet'
  const safeBundleId = sanitizeBundleIdForFileName(bundleData.bundleId || fallbackBaseName)
  const draftFileName = `${safeBundleId}_autosave.json`
  return directory ? `${directory}/${draftFileName}` : draftFileName
}

function buildLoadDraftUrl(bundleData: ReviewBundle) {
  const params = new URLSearchParams()
  if (bundleData.bundleId) {
    params.set('bundleId', bundleData.bundleId)
  }

  const reviewSheetFile = normalizeSourcePath(bundleData.source.reviewSheetFile)
  if (reviewSheetFile) {
    params.set('reviewSheetFile', reviewSheetFile)
  }

  const defaultDraftFile = normalizeSourcePath(bundleData.source.defaultDraftFile)
  if (defaultDraftFile) {
    params.set('defaultDraftFile', defaultDraftFile)
  }

  const query = params.toString()
  return query ? `/api/review/load-draft?${query}` : '/api/review/load-draft'
}

function normalizeStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map(item => String(item ?? '').trim())
      .filter(Boolean)
  }
  if (typeof value === 'string') {
    return value
      .split('|')
      .map(item => item.trim())
      .filter(Boolean)
  }
  return []
}

function orderKpids(item: ReviewBundleItem, kpids: string[]) {
  const selected = new Set(kpids.filter(Boolean))
  return item.candidateKpidOptions
    .map(option => option.value)
    .filter(kpid => selected.has(kpid))
}

function buildDefaultDecision(item: ReviewBundleItem): ReviewDecision {
  return cloneDecision(item.decisionDraft)
}

function mergeDecisionMap(
  source: Record<string, Partial<ReviewDecision>>,
  reason: 'server' | 'local',
) {
  if (!bundle.value) return

  const baseDecisions = { ...decisions.value }
  for (const [itemId, incoming] of Object.entries(source)) {
    if (!baseDecisions[itemId]) continue
    baseDecisions[itemId] = normalizeIncomingDecision(incoming, baseDecisions[itemId])
  }
  decisions.value = baseDecisions

  if (reason === 'server') {
    localStorage.setItem(storageKey(bundle.value.bundleId), JSON.stringify(baseDecisions))
  }
}

function normalizeIncomingDecision(raw: Partial<ReviewDecision>, base: ReviewDecision): ReviewDecision {
  return {
    finalExpectedOtid: raw.finalExpectedOtid ?? base.finalExpectedOtid,
    finalExpectedKpid: raw.finalExpectedKpid ?? base.finalExpectedKpid,
    finalSupportKpids:
      raw.finalSupportKpids !== undefined
        ? normalizeStringList(raw.finalSupportKpids)
        : [...base.finalSupportKpids],
    otidStatus: raw.otidStatus ?? base.otidStatus,
    kpidStatus: raw.kpidStatus ?? base.kpidStatus,
    finalQueryScope: raw.finalQueryScope ?? base.finalQueryScope,
    finalPreferredGranularity:
      raw.finalPreferredGranularity ?? base.finalPreferredGranularity,
    finalSupportPattern: raw.finalSupportPattern ?? base.finalSupportPattern,
    finalGranularitySensitive:
      raw.finalGranularitySensitive ?? base.finalGranularitySensitive,
    includeInFormalSet: raw.includeInFormalSet ?? base.includeInFormalSet,
    reviewRound: raw.reviewRound ?? base.reviewRound,
    reviewer: raw.reviewer ?? base.reviewer,
    notes: raw.notes ?? base.notes,
  }
}

function loadBundleData(data: ReviewBundle) {
  bundle.value = data
  const baseDecisions = Object.fromEntries(
    data.items.map(item => [item.id, buildDefaultDecision(item)]),
  ) as Record<string, ReviewDecision>

  const rawSaved = localStorage.getItem(storageKey(data.bundleId))
  if (rawSaved) {
    try {
      const parsed = JSON.parse(rawSaved) as Record<string, Partial<ReviewDecision>>
      for (const [itemId, savedDecision] of Object.entries(parsed)) {
        if (!baseDecisions[itemId]) continue
        baseDecisions[itemId] = normalizeIncomingDecision(
          savedDecision,
          baseDecisions[itemId],
        )
      }
    } catch {
      // 忽略损坏缓存
    }
  }

  decisions.value = baseDecisions
  if (!data.items.find(item => item.id === activeItemId.value)) {
    activeItemId.value = data.items[0]?.id ?? ''
  }
  bundleLoadError.value = ''
  void tryLoadServerDraft(data)
}

async function tryLoadDefaultBundle() {
  loadingDefaultBundle.value = true
  bundleLoadError.value = ''
  for (const path of DEFAULT_BUNDLE_PATHS) {
    try {
      const response = await fetch(path, { cache: 'no-store' })
      if (!response.ok) continue
      const data = (await response.json()) as ReviewBundle
      if (data?.items?.length) {
        loadBundleData(data)
        loadingDefaultBundle.value = false
        return
      }
    } catch {
      // 继续尝试下一个默认路径
    }
  }
  loadingDefaultBundle.value = false
  bundleLoadError.value =
    '没有自动找到 review bundle。你可以先运行导出脚本，或手动上传 bundle JSON。'
}

async function tryLoadServerDraft(targetBundle: ReviewBundle) {
  if (!targetBundle.bundleId) return

  loadingServerDraft.value = true
  try {
    const response = await fetch(buildLoadDraftUrl(targetBundle), {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    })

    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data?.detail || `读取固定草稿失败（${response.status}）`)
    }
    if (!data?.exists || !data?.payload) {
      return
    }

    const payload = data.payload as {
      bundleId?: string
      items?: Array<{ id?: string; decision?: Partial<ReviewDecision> }>
    }

    if (payload.bundleId && payload.bundleId !== targetBundle.bundleId) {
      serverActionMessage.value = `检测到草稿，但它属于其他 bundle：${payload.bundleId}，已跳过自动恢复。`
      return
    }

    const decisionMap = Object.fromEntries(
      (payload.items ?? [])
        .filter(item => item?.id && item?.decision)
        .map(item => [item.id as string, item.decision as Partial<ReviewDecision>]),
    ) as Record<string, Partial<ReviewDecision>>

    if (!Object.keys(decisionMap).length) {
      return
    }

    mergeDecisionMap(decisionMap, 'server')
    serverActionMessage.value = `已自动恢复当前 bundle 草稿：${data.outputPath}`
    serverActionError.value = ''
  } catch (error) {
    serverActionError.value = error instanceof Error ? error.message : '读取当前 bundle 草稿失败'
  } finally {
    loadingServerDraft.value = false
  }
}

function handleBundleFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  const reader = new FileReader()
  reader.onload = loaded => {
    try {
      const parsed = JSON.parse(String(loaded.target?.result ?? '{}')) as ReviewBundle
      loadBundleData(parsed)
    } catch {
      bundleLoadError.value = 'bundle JSON 解析失败，请确认文件格式正确。'
    }
  }
  reader.readAsText(file, 'utf-8')
  target.value = ''
}

function isCompleted(itemId: string) {
  const decision = decisions.value[itemId]
  if (!decision) return false
  if (!decision.finalQueryScope) return false
  if (!decision.finalPreferredGranularity) return false
  if (!decision.finalSupportPattern) return false
  if (decision.finalGranularitySensitive === null) return false
  if (!decision.otidStatus || decision.otidStatus === '待复核') return false
  if (
    decision.finalSupportPattern === 'single_kp'
    && (!decision.finalExpectedKpid || !decision.kpidStatus || decision.kpidStatus === '待复核')
  ) {
    return false
  }
  if (
    decision.finalSupportPattern === 'multi_kp'
    && decision.finalSupportKpids.length < 2
  ) {
    return false
  }
  if (!decision.includeInFormalSet || decision.includeInFormalSet === '待定') return false
  return true
}

function getDecisionFor(itemId: string): ReviewDecision | null {
  return decisions.value[itemId] ?? null
}

function getDecisionWarnings(item: ReviewBundleItem, decision: ReviewDecision | null) {
  if (!decision) return [] as string[]

  const warnings: string[] = []
  if (decision.finalSupportPattern === 'single_kp' && !decision.finalExpectedKpid) {
    warnings.push('`single_kp` 通常应给出明确的主支撑 `kpid`。')
  }
  if (
    decision.finalSupportPattern === 'multi_kp'
    && decision.finalSupportKpids.length < 2
  ) {
    warnings.push('`multi_kp` 通常应至少勾选 2 条联合支撑 `kpid`。')
  }
  if (
    decision.finalPreferredGranularity === 'OT'
    && decision.finalSupportPattern === 'single_kp'
  ) {
    warnings.push('当前组合是 `OT + single_kp`，请确认是否真的需要 OT 检索而不是 KP。')
  }
  if (
    decision.finalPreferredGranularity === 'KP'
    && decision.finalSupportPattern === 'ot_required'
  ) {
    warnings.push('当前组合是 `KP + ot_required`，请确认是否真的只靠 KP 就适合检出该问题。')
  }
  if (
    decision.finalSupportPattern === 'ot_required'
    && decision.finalExpectedKpid
  ) {
    warnings.push('`ot_required` 可以保留主支撑点，但如果没有稳定单一主 `kpid`，允许留空。')
  }
  if (
    decision.finalQueryScope === 'procedure'
    && item.queryType === 'short_keyword'
  ) {
    warnings.push('短关键词被标为 `procedure` 时，请确认它问的是流程而不是时间点或单点事实。')
  }
  if (
    decision.finalExpectedKpid
    && decision.finalSupportKpids.length
    && !decision.finalSupportKpids.includes(decision.finalExpectedKpid)
  ) {
    warnings.push('当前主支撑 `kpid` 还没有纳入“联合支撑 kpids”列表。')
  }
  if (
    decision.finalSupportPattern === 'single_kp'
    && decision.finalSupportKpids.length > 1
  ) {
    warnings.push('当前标为 `single_kp`，但联合支撑里选择了多条 `kpid`，请确认是否应改成 `multi_kp`。')
  }
  if (decision.otidStatus === '存疑') {
    warnings.push('当前 `otid` 仍标记为“存疑”，建议优先回看文档锚点是否足够唯一。')
  }
  if (
    decision.finalSupportPattern === 'single_kp'
    && decision.kpidStatus === '存疑'
  ) {
    warnings.push('当前主支撑 `kpid` 仍标记为“存疑”，建议结合证据区重新确认。')
  }
  return warnings
}

function getAiRiskMessages(item: ReviewBundleItem) {
  const risks: string[] = []

  if (item.ai.uniquenessRisk && item.ai.uniquenessRisk !== 'low') {
    risks.push(`AI 预估唯一性风险为 \`${item.ai.uniquenessRisk}\`，建议多看标题锚点和年份锚点。`)
  }
  if (item.ai.flagLowSpecificity) {
    risks.push('AI 标记这条 query 可能特异性偏低，进入正式集前建议确认是否容易命中多个通知。')
  }
  if (item.ai.flagGenericQuery) {
    risks.push('AI 标记这条 query 可能过泛，建议确认是否缺少主题或对象锚点。')
  }
  if (item.ai.priority === 'review_carefully') {
    risks.push('AI 优先级为 `review_carefully`，说明这条更适合优先细看。')
  }
  if (item.ai.seedFailureRisks.length) {
    risks.push(`这条样本来自 bad case 定向扩样，失败风险包括：${item.ai.seedFailureRisks.join('、')}。`)
  }

  return risks
}

function getConflictCount(item: ReviewBundleItem) {
  return getDecisionWarnings(item, getDecisionFor(item.id)).length
}

function getAiRiskCount(item: ReviewBundleItem) {
  return getAiRiskMessages(item).length
}

function hasDecisionConflict(item: ReviewBundleItem) {
  return getConflictCount(item) > 0
}

function hasAiRisk(item: ReviewBundleItem) {
  return getAiRiskCount(item) > 0
}

function hasConflictOrRisk(item: ReviewBundleItem) {
  return hasDecisionConflict(item) || hasAiRisk(item)
}

const currentReviewSheetPath = computed(() =>
  normalizeSourcePath(bundle.value?.source.reviewSheetFile),
)

const currentDraftPath = computed(() => inferDraftPath(bundle.value))

const filteredItems = computed(() => {
  const items = bundle.value?.items ?? []
  const query = searchText.value.trim().toLowerCase()

  return items.filter(item => {
    if (p0Only.value && !item.p0.included) return false
    if (priorityFilter.value !== 'all' && item.ai.priority !== priorityFilter.value) {
      return false
    }

    const done = isCompleted(item.id)
    if (completionFilter.value === 'pending' && done) return false
    if (completionFilter.value === 'completed' && !done) return false
    if (riskFilter.value === 'conflict_or_risk' && !hasConflictOrRisk(item)) {
      return false
    }
    if (riskFilter.value === 'conflict_only' && !hasDecisionConflict(item)) {
      return false
    }
    if (riskFilter.value === 'ai_risk_only' && !hasAiRisk(item)) {
      return false
    }

    if (!query) return true
    const haystack = [
      item.query,
      item.targetTitle,
      item.themeFamily,
      item.article.otid,
      item.ai.priority,
      item.ai.uniquenessRisk,
    ]
      .join(' ')
      .toLowerCase()
    return haystack.includes(query)
  })
})

const currentItem = computed(() =>
  filteredItems.value.find(item => item.id === activeItemId.value)
  ?? bundle.value?.items.find(item => item.id === activeItemId.value)
  ?? filteredItems.value[0]
  ?? null,
)

watch(filteredItems, items => {
  if (!items.length) {
    activeItemId.value = ''
    return
  }
  if (!items.find(item => item.id === activeItemId.value)) {
    activeItemId.value = items[0].id
  }
})

watch(
  () => bundle.value?.bundleId,
  bundleId => {
    if (!bundleId) return
    const rawReviewer = localStorage.getItem(`${storageKey(bundleId)}:reviewer`)
    reviewerDefault.value = rawReviewer ?? ''
  },
)

watch(reviewerDefault, value => {
  if (!bundle.value) return
  localStorage.setItem(`${storageKey(bundle.value.bundleId)}:reviewer`, value)
})

watch(
  decisions,
  value => {
    if (!bundle.value) return
    localStorage.setItem(storageKey(bundle.value.bundleId), JSON.stringify(value))
  },
  { deep: true },
)

function currentDecision(): ReviewDecision | null {
  if (!currentItem.value) return null
  return decisions.value[currentItem.value.id] ?? null
}

function updateDecision<K extends keyof ReviewDecision>(key: K, value: ReviewDecision[K]) {
  const item = currentItem.value
  if (!item) return
  decisions.value[item.id] = {
    ...decisions.value[item.id],
    [key]: value,
    reviewer: decisions.value[item.id]?.reviewer || reviewerDefault.value,
  }
}

function updateSupportKpids(kpids: string[]) {
  const item = currentItem.value
  if (!item) return
  updateDecision('finalSupportKpids', orderKpids(item, kpids))
}

function toggleSupportKpid(kpid: string, checked: boolean) {
  const decision = currentDecision()
  if (!decision) return
  const next = new Set(decision.finalSupportKpids)
  if (checked) {
    next.add(kpid)
  } else {
    next.delete(kpid)
  }
  updateSupportKpids(Array.from(next))
}

function useAiSupportKpids() {
  const item = currentItem.value
  if (!item) return
  updateSupportKpids(item.evidenceKpids)
}

function clearSupportKpids() {
  updateSupportKpids([])
}

function handleExpectedKpidChange(value: string) {
  const item = currentItem.value
  if (!item) return
  const base = decisions.value[item.id]
  const nextSupport = new Set(base.finalSupportKpids)
  if (value) {
    nextSupport.add(value)
  }
  decisions.value[item.id] = {
    ...base,
    finalExpectedKpid: value,
    finalSupportKpids: orderKpids(item, Array.from(nextSupport)),
    reviewer: base.reviewer || reviewerDefault.value,
  }
}

function handleSupportPatternChange(value: string) {
  const item = currentItem.value
  if (!item) return
  const base = decisions.value[item.id]
  let nextSupport = [...base.finalSupportKpids]

  if (value === 'single_kp' && base.finalExpectedKpid) {
    nextSupport = [base.finalExpectedKpid]
  }
  if (value === 'ot_required' && !base.finalExpectedKpid) {
    nextSupport = []
  }

  decisions.value[item.id] = {
    ...base,
    finalSupportPattern: value,
    finalSupportKpids: orderKpids(item, nextSupport),
    reviewer: base.reviewer || reviewerDefault.value,
  }
}

function useSuggestedDraft() {
  const item = currentItem.value
  if (!item) return
  decisions.value[item.id] = {
    ...buildDefaultDecision(item),
    reviewer: reviewerDefault.value || item.decisionDraft.reviewer,
  }
}

function markConfirmed() {
  const item = currentItem.value
  if (!item) return
  const base = decisions.value[item.id]
  decisions.value[item.id] = {
    ...base,
    otidStatus: '已确认',
    kpidStatus: base.finalSupportPattern === 'ot_required' ? '待复核' : '已确认',
    includeInFormalSet: base.includeInFormalSet === '待定' ? '是' : base.includeInFormalSet,
    reviewer: base.reviewer || reviewerDefault.value,
  }
}

const progress = computed(() => {
  const items = filteredItems.value
  const completed = items.filter(item => isCompleted(item.id)).length
  const flagged = items.filter(item => hasConflictOrRisk(item)).length
  return {
    total: items.length,
    completed,
    pending: Math.max(items.length - completed, 0),
    flagged,
  }
})

const currentConflictMessages = computed(() => {
  const item = currentItem.value
  const decision = currentDecision()
  if (!item) return []
  return getDecisionWarnings(item, decision)
})

const currentDocKps = computed(() => {
  const item = currentItem.value
  const decision = currentDecision()
  if (!item) return []

  const supportKpidSet = new Set(decision?.finalSupportKpids ?? [])
  return item.docKps.map(kp => ({
    ...kp,
    isSelectedMain: kp.kpid === decision?.finalExpectedKpid,
    isSelectedSupport: supportKpidSet.has(kp.kpid),
  }))
})

function jumpToNeighbor(direction: 'prev' | 'next') {
  const items = filteredItems.value
  if (!items.length || !currentItem.value) return
  const currentIndex = items.findIndex(item => item.id === currentItem.value?.id)
  if (currentIndex < 0) return
  const nextIndex = direction === 'next'
    ? Math.min(currentIndex + 1, items.length - 1)
    : Math.max(currentIndex - 1, 0)
  activeItemId.value = items[nextIndex].id
}

function jumpToNextPending() {
  const pending = filteredItems.value.find(item => !isCompleted(item.id))
  if (pending) {
    activeItemId.value = pending.id
  }
}

function downloadFile(fileName: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = fileName
  anchor.click()
  URL.revokeObjectURL(url)
}

function buildDecisionPayload() {
  if (!bundle.value) return null
  return {
    bundleId: bundle.value.bundleId,
    source: { ...bundle.value.source },
    exportedAt: new Date().toISOString(),
    reviewerDefault: reviewerDefault.value,
    items: bundle.value.items.map(item => ({
      id: item.id,
      sourceSeedId: item.sourceSeedId,
      query: item.query,
      targetTitle: item.targetTitle,
      decision: decisions.value[item.id],
    })),
  }
}

async function postWorkbenchPayload(url: string) {
  const payload = buildDecisionPayload()
  if (!payload) {
    throw new Error('当前还没有加载 review bundle。')
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data?.detail || `请求失败（${response.status}）`)
  }
  return data as {
    ok: boolean
    outputPath: string
    backupPath?: string | null
    updatedItems?: number
    savedItems?: number
  }
}

async function saveDraftOnServer() {
  serverActionError.value = ''
  serverActionMessage.value = ''
  savingDraftToServer.value = true
  try {
    const data = await postWorkbenchPayload('/api/review/save-draft')
    serverActionMessage.value = `已保存到当前 bundle 草稿：${data.outputPath}${data.savedItems !== undefined ? `（${data.savedItems} 条）` : ''}`
  } catch (error) {
    serverActionError.value = error instanceof Error ? error.message : '保存草稿失败'
  } finally {
    savingDraftToServer.value = false
  }
}

async function applyDecisionsOnServer() {
  serverActionError.value = ''
  serverActionMessage.value = ''
  applyingToServer.value = true
  try {
    const data = await postWorkbenchPayload('/api/review/apply-decisions')
    serverActionMessage.value = `已回写当前复核表：${data.outputPath}${data.updatedItems !== undefined ? `（更新 ${data.updatedItems} 条）` : ''}${data.backupPath ? `；备份：${data.backupPath}` : ''}`
  } catch (error) {
    serverActionError.value = error instanceof Error ? error.message : '回写复核表失败'
  } finally {
    applyingToServer.value = false
  }
}

function exportDecisionsJson() {
  if (!bundle.value) return
  const payload = buildDecisionPayload()
  if (!payload) return
  downloadFile(
    `${bundle.value.bundleId}_decisions.json`,
    JSON.stringify(payload, null, 2),
    'application/json;charset=utf-8',
  )
}

function escapeCsvCell(value: string) {
  const normalized = value.replace(/"/g, '""')
  return `"${normalized}"`
}

function exportDecisionsCsv() {
  if (!bundle.value) return
  const header = [
    'AI_item_id',
    'source_seed_id',
    'query',
    '目标标题',
    '最终_expected_otid',
    'otid复核状态',
    '最终_expected_kpid',
    '最终_support_kpids',
    'kpid复核状态',
    '最终_query_scope',
    '最终_preferred_granularity',
    '最终_support_pattern',
    '最终_granularity_sensitive',
    '是否纳入正式集',
    '复核人',
    '复核备注',
  ]
  const lines = [
    header.join(','),
    ...bundle.value.items.map(item => {
      const decision = decisions.value[item.id]
      const row = [
        item.id,
        item.sourceSeedId,
        item.query,
        item.targetTitle,
        decision.finalExpectedOtid,
        decision.otidStatus,
        decision.finalExpectedKpid,
        decision.finalSupportKpids.join('|'),
        decision.kpidStatus,
        decision.finalQueryScope,
        decision.finalPreferredGranularity,
        decision.finalSupportPattern,
        decision.finalGranularitySensitive === null
          ? ''
          : String(decision.finalGranularitySensitive),
        decision.includeInFormalSet,
        decision.reviewer || reviewerDefault.value,
        decision.notes,
      ]
      return row.map(value => escapeCsvCell(String(value ?? ''))).join(',')
    }),
  ]

  downloadFile(
    `${bundle.value.bundleId}_decisions.csv`,
    lines.join('\n'),
    'text/csv;charset=utf-8',
  )
}

function suggestedValueChip(current: string, suggested: string) {
  if (!suggested) return '未提供建议'
  if (current && current !== suggested) {
    return `建议值：${suggested} | 当前已改`
  }
  return `建议值：${suggested}`
}

const currentRiskMessages = computed(() => {
  const item = currentItem.value
  if (!item) return []
  return getAiRiskMessages(item)
})

const warningMessages = computed(() => [
  ...currentRiskMessages.value,
  ...currentConflictMessages.value,
])

function shouldIgnoreShortcut(event: KeyboardEvent) {
  const target = event.target as HTMLElement | null
  if (!target) return false
  const tagName = target.tagName
  return (
    target.isContentEditable
    || tagName === 'INPUT'
    || tagName === 'TEXTAREA'
    || tagName === 'SELECT'
  )
}

function handleGlobalKeydown(event: KeyboardEvent) {
  if (!bundle.value || !filteredItems.value.length) return
  if (shouldIgnoreShortcut(event)) return
  if (event.ctrlKey || event.metaKey || event.altKey) return

  if (event.key === 'j' || event.key === 'J' || event.key === 'ArrowDown') {
    event.preventDefault()
    jumpToNeighbor('next')
    return
  }

  if (event.key === 'k' || event.key === 'K' || event.key === 'ArrowUp') {
    event.preventDefault()
    jumpToNeighbor('prev')
    return
  }

  if (event.key === 'n' || event.key === 'N') {
    event.preventDefault()
    jumpToNextPending()
    return
  }

  if (event.key === 'Enter' && event.shiftKey) {
    event.preventDefault()
    markConfirmed()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleGlobalKeydown)
  void tryLoadDefaultBundle()
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeydown)
})
</script>

<template>
  <div class="flex h-full min-h-[78vh] flex-col gap-4">
    <section
      class="rounded-[28px] border border-slate-700/60 bg-slate-900/80 p-4 shadow-[0_24px_70px_rgba(15,23,42,0.45)] backdrop-blur"
    >
      <div class="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <div class="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-300/75">
            Review Workbench
          </div>
          <h2 class="mt-2 text-2xl font-black tracking-tight text-white">
            Granularity 标注工作台
          </h2>
          <p class="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            同一屏完成 query、证据、标签定义和复核决定。先看文档是否正确，再看主支撑点，再决定是否纳入正式集。
          </p>
        </div>

        <div class="flex flex-wrap items-center gap-3">
          <button
            class="inline-flex items-center gap-2 rounded-full border border-violet-500/50 bg-violet-500/10 px-4 py-2 text-sm font-semibold text-violet-100 hover:bg-violet-500/18"
            type="button"
            :disabled="!bundle || savingDraftToServer"
            @click="saveDraftOnServer"
          >
            <RefreshCcw class="h-4 w-4" />
            <span>{{ savingDraftToServer ? '保存中...' : '保存到当前草稿' }}</span>
          </button>
          <button
            class="inline-flex items-center gap-2 rounded-full border border-emerald-400/60 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-500/18"
            type="button"
            :disabled="!bundle || applyingToServer"
            @click="applyDecisionsOnServer"
          >
            <CheckCircle2 class="h-4 w-4" />
            <span>{{ applyingToServer ? '回写中...' : '直接回写当前复核表' }}</span>
          </button>
          <label class="inline-flex cursor-pointer items-center gap-2 rounded-full border border-slate-600/80 bg-slate-950/70 px-4 py-2 text-sm font-medium text-slate-100 hover:border-cyan-400/60">
            <FileUp class="h-4 w-4" />
            <span>上传 Bundle</span>
            <input class="hidden" type="file" accept=".json" @change="handleBundleFileChange" />
          </label>
          <button
            class="inline-flex items-center gap-2 rounded-full border border-slate-600/80 bg-slate-950/70 px-4 py-2 text-sm font-medium text-slate-100 hover:border-cyan-400/60"
            type="button"
            @click="tryLoadDefaultBundle"
          >
            <RefreshCcw class="h-4 w-4" />
            <span>{{ loadingDefaultBundle ? '加载中...' : '重新加载 Bundle' }}</span>
          </button>
          <button
            class="inline-flex items-center gap-2 rounded-full border border-cyan-500/60 bg-cyan-500/10 px-4 py-2 text-sm font-semibold text-cyan-100 hover:bg-cyan-500/18"
            type="button"
            @click="exportDecisionsJson"
            :disabled="!bundle"
          >
            <Download class="h-4 w-4" />
            <span>导出 JSON</span>
          </button>
          <button
            class="inline-flex items-center gap-2 rounded-full border border-emerald-500/50 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-500/18"
            type="button"
            @click="exportDecisionsCsv"
            :disabled="!bundle"
          >
            <Download class="h-4 w-4" />
            <span>导出 CSV</span>
          </button>
        </div>
      </div>

      <div class="mt-4 flex flex-col gap-3 rounded-[24px] border border-slate-800/90 bg-slate-950/70 p-4 lg:flex-row lg:items-end lg:justify-between">
        <div class="flex flex-wrap items-end gap-3">
          <label class="flex flex-col gap-2">
            <span class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">搜索</span>
            <input
              v-model="searchText"
              class="min-w-[220px] rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none ring-0 placeholder:text-slate-500 focus:border-cyan-400/70"
              placeholder="按 query / 标题 / 主题筛选"
              type="text"
            />
          </label>

          <label class="flex flex-col gap-2">
            <span class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">优先级</span>
            <select
              v-model="priorityFilter"
              class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
            >
              <option value="all">全部</option>
              <option value="high">high</option>
              <option value="review_carefully">review_carefully</option>
              <option value="medium">medium</option>
            </select>
          </label>

          <label class="flex flex-col gap-2">
            <span class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">完成状态</span>
            <select
              v-model="completionFilter"
              class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
            >
              <option value="pending">只看未完成</option>
              <option value="completed">只看已完成</option>
              <option value="all">全部</option>
            </select>
          </label>

          <label class="flex flex-col gap-2">
            <span class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">风险视图</span>
            <select
              v-model="riskFilter"
              class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
            >
              <option value="all">全部</option>
              <option value="conflict_or_risk">只看冲突/高风险</option>
              <option value="conflict_only">只看当前冲突</option>
              <option value="ai_risk_only">只看 AI 风险</option>
            </select>
          </label>

          <label class="flex items-center gap-2 rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-slate-200">
            <input v-model="p0Only" class="accent-cyan-400" type="checkbox" />
            <span>只看 P0</span>
          </label>

          <label class="flex flex-col gap-2">
            <span class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">默认复核人</span>
            <input
              v-model="reviewerDefault"
              class="min-w-[160px] rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none placeholder:text-slate-500 focus:border-cyan-400/70"
              placeholder="例如 DreamingLri"
              type="text"
            />
          </label>
        </div>

        <div class="grid gap-3 text-sm text-slate-200 sm:grid-cols-2 xl:grid-cols-4">
          <div class="rounded-2xl border border-slate-800 bg-slate-900/90 px-4 py-3">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">当前过滤结果</div>
            <div class="mt-2 text-xl font-black text-white">{{ progress.total }}</div>
          </div>
          <div class="rounded-2xl border border-slate-800 bg-slate-900/90 px-4 py-3">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">已完成</div>
            <div class="mt-2 text-xl font-black text-emerald-300">{{ progress.completed }}</div>
          </div>
          <div class="rounded-2xl border border-slate-800 bg-slate-900/90 px-4 py-3">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">待处理</div>
            <div class="mt-2 text-xl font-black text-amber-300">{{ progress.pending }}</div>
          </div>
          <div class="rounded-2xl border border-slate-800 bg-slate-900/90 px-4 py-3">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">冲突 / 高风险</div>
            <div class="mt-2 text-xl font-black text-rose-300">{{ progress.flagged }}</div>
          </div>
        </div>
      </div>

      <p class="mt-3 text-xs leading-6 text-slate-400">
        快捷键：`J / ↓` 下一条，`K / ↑` 上一条，`N` 跳到下一个未完成，`Shift + Enter` 快速标记确认。输入框聚焦时不会触发。
      </p>

      <p class="mt-2 text-xs leading-6 text-slate-500">
        当前 bundle：`{{ bundle?.bundleId || '未加载' }}`
      </p>

      <p v-if="bundle" class="mt-2 text-xs leading-6 text-slate-500">
        当前草稿文件：`{{ currentDraftPath || '未提供' }}`；
        当前复核表：`{{ currentReviewSheetPath || '未提供' }}`
      </p>

      <p v-if="loadingServerDraft" class="mt-2 text-xs leading-6 text-cyan-200/80">
        正在尝试恢复固定草稿...
      </p>

      <p
        v-if="serverActionMessage"
        class="mt-3 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100"
      >
        {{ serverActionMessage }}
      </p>

      <p
        v-if="serverActionError"
        class="mt-3 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100"
      >
        {{ serverActionError }}
      </p>

      <p v-if="bundleLoadError" class="mt-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
        {{ bundleLoadError }}
      </p>
    </section>

    <section
      v-if="bundle"
      class="grid min-h-0 flex-1 gap-4 xl:grid-cols-[320px_minmax(0,1.15fr)_minmax(340px,0.95fr)]"
    >
      <aside class="min-h-0 overflow-hidden rounded-[28px] border border-slate-700/60 bg-slate-900/80 p-4">
        <div class="flex items-center justify-between">
          <div>
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">样本列表</div>
            <div class="mt-1 text-lg font-bold text-white">Review Queue</div>
          </div>
          <button
            class="rounded-full border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-cyan-400/60"
            type="button"
            @click="jumpToNextPending"
          >
            跳到下一个未完成
          </button>
        </div>

        <div class="mt-4 flex h-[calc(100%-72px)] flex-col gap-3 overflow-y-auto pr-1">
          <button
            v-for="item in filteredItems"
            :key="item.id"
            type="button"
            class="rounded-[22px] border px-4 py-3 text-left transition-all"
            :class="item.id === currentItem?.id
              ? 'border-cyan-400/70 bg-cyan-400/10 shadow-[0_14px_40px_rgba(34,211,238,0.12)]'
              : hasConflictOrRisk(item)
                ? 'border-amber-500/35 bg-amber-500/6 hover:border-amber-400/60'
              : 'border-slate-800 bg-slate-950/65 hover:border-slate-600'"
            @click="activeItemId = item.id"
          >
            <div class="flex items-start justify-between gap-3">
              <div class="flex-1">
                <div class="line-clamp-2 text-sm font-semibold leading-6 text-white">
                  {{ item.query }}
                </div>
                <div class="mt-1 line-clamp-2 text-xs leading-5 text-slate-400">
                  {{ item.targetTitle }}
                </div>
              </div>
              <div class="flex flex-col items-end gap-2">
                <span
                  class="rounded-full px-2 py-1 text-[11px] font-bold uppercase tracking-[0.14em]"
                  :class="item.ai.priority === 'high'
                    ? 'bg-emerald-500/15 text-emerald-200'
                    : item.ai.priority === 'review_carefully'
                      ? 'bg-amber-500/15 text-amber-200'
                      : 'bg-slate-700/70 text-slate-200'"
                >
                  {{ item.ai.priority }}
                </span>
                <span
                  class="inline-flex items-center gap-1 text-[11px] font-semibold"
                  :class="isCompleted(item.id) ? 'text-emerald-300' : 'text-amber-300'"
                >
                  <CheckCircle2 v-if="isCompleted(item.id)" class="h-3.5 w-3.5" />
                  <AlertTriangle v-else class="h-3.5 w-3.5" />
                  {{ isCompleted(item.id) ? '已完成' : '待处理' }}
                </span>
              </div>
            </div>

            <div class="mt-3 flex flex-wrap gap-2 text-[11px]">
              <span class="rounded-full bg-slate-800 px-2 py-1 text-slate-300">
                {{ item.queryType }}
              </span>
              <span
                v-if="item.p0.included"
                class="rounded-full bg-cyan-500/15 px-2 py-1 text-cyan-200"
              >
                P0
              </span>
              <span class="rounded-full bg-slate-800 px-2 py-1 text-slate-300">
                {{ item.themeFamily || '未分类' }}
              </span>
              <span
                v-if="getAiRiskCount(item)"
                class="rounded-full bg-rose-500/15 px-2 py-1 text-rose-200"
              >
                风险 {{ getAiRiskCount(item) }}
              </span>
              <span
                v-if="getConflictCount(item)"
                class="rounded-full bg-amber-500/15 px-2 py-1 text-amber-200"
              >
                冲突 {{ getConflictCount(item) }}
              </span>
            </div>
          </button>
        </div>
      </aside>

      <main
        v-if="currentItem && currentDecision()"
        class="min-h-0 overflow-y-auto rounded-[28px] border border-slate-700/60 bg-slate-900/80 p-5"
      >
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">
              {{ currentItem.id }}
            </div>
            <h3 class="mt-2 text-2xl font-black leading-9 text-white">
              {{ currentItem.query }}
            </h3>
            <p class="mt-2 text-sm leading-6 text-slate-300">
              {{ currentItem.targetTitle }}
            </p>
          </div>

          <div class="flex flex-wrap gap-2">
            <button
              class="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/70 px-4 py-2 text-sm font-medium text-slate-100 hover:border-cyan-400/60"
              type="button"
              @click="useSuggestedDraft"
            >
              <RefreshCcw class="h-4 w-4" />
              恢复建议稿
            </button>
            <button
              class="inline-flex items-center gap-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-500/18"
              type="button"
              @click="markConfirmed"
            >
              <CheckCircle2 class="h-4 w-4" />
              快速标记确认
            </button>
          </div>
        </div>

        <div class="mt-4 flex flex-wrap gap-2 text-xs font-semibold">
          <span class="rounded-full bg-slate-800 px-3 py-1.5 text-slate-200">
            query_type: {{ currentItem.queryType }}
          </span>
          <span class="rounded-full bg-slate-800 px-3 py-1.5 text-slate-200">
            AI priority: {{ currentItem.ai.priority }}
          </span>
          <span class="rounded-full bg-slate-800 px-3 py-1.5 text-slate-200">
            uniqueness_risk: {{ currentItem.ai.uniquenessRisk || 'unknown' }}
          </span>
          <span class="rounded-full bg-slate-800 px-3 py-1.5 text-slate-200">
            P0: {{ currentItem.p0.included ? '是' : '否' }}
          </span>
          <span class="rounded-full bg-slate-800 px-3 py-1.5 text-slate-200">
            publish_time: {{ currentItem.article.publishTime || '未知' }}
          </span>
        </div>

        <div
          v-if="warningMessages.length"
          class="mt-5 rounded-[24px] border border-amber-400/30 bg-amber-500/10 p-4"
        >
          <div class="flex items-center gap-2 text-sm font-bold text-amber-100">
            <AlertTriangle class="h-4 w-4" />
            复核提醒
          </div>

          <div v-if="currentRiskMessages.length" class="mt-3">
            <div class="text-xs font-semibold uppercase tracking-[0.16em] text-amber-200/75">
              AI 风险信号
            </div>
            <ul class="mt-2 space-y-2 text-sm leading-6 text-amber-50/90">
              <li v-for="message in currentRiskMessages" :key="`risk-${message}`">
                {{ message }}
              </li>
            </ul>
          </div>

          <div v-if="currentConflictMessages.length" class="mt-4">
            <div class="text-xs font-semibold uppercase tracking-[0.16em] text-amber-200/75">
              当前标签冲突
            </div>
            <ul class="mt-2 space-y-2 text-sm leading-6 text-amber-50/90">
              <li v-for="message in currentConflictMessages" :key="`conflict-${message}`">
                {{ message }}
              </li>
            </ul>
          </div>
        </div>

        <div class="mt-6 grid gap-4 xl:grid-cols-2">
          <div class="rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">文档判断</div>
            <div class="mt-4 space-y-4">
              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">最终 expected_otid</span>
                <input
                  :value="currentDecision()?.finalExpectedOtid"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  type="text"
                  @input="updateDecision('finalExpectedOtid', ($event.target as HTMLInputElement).value)"
                />
              </label>

              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">otid复核状态</span>
                <select
                  :value="currentDecision()?.otidStatus"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @change="updateDecision('otidStatus', ($event.target as HTMLSelectElement).value)"
                >
                  <option value="">未设置</option>
                  <option
                    v-for="option in REVIEW_STATUS_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.label }}
                  </option>
                </select>
              </label>
            </div>
          </div>

          <div class="rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">主支撑点判断</div>
            <div class="mt-4 space-y-4">
              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">最终 expected_kpid</span>
                <select
                  :value="currentDecision()?.finalExpectedKpid"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @change="handleExpectedKpidChange(($event.target as HTMLSelectElement).value)"
                >
                  <option value="">暂不填写</option>
                  <option
                    v-for="option in currentItem.candidateKpidOptions"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.isEvidence ? 'AI证据 | ' : '' }}{{ option.label }}
                  </option>
                </select>
              </label>

              <div class="rounded-2xl border border-slate-800 bg-slate-900/70 p-3">
                <div class="flex items-center justify-between gap-3">
                  <span class="text-sm font-semibold text-slate-200">联合支撑 kpids</span>
                  <span class="text-xs font-medium text-slate-500">
                    已选 {{ currentDecision()?.finalSupportKpids.length || 0 }} 条
                  </span>
                </div>
                <p class="mt-2 text-xs leading-5 text-slate-400">
                  `multi_kp` 建议至少勾选 2 条；`single_kp` 通常只保留主支撑即可。
                </p>
                <div class="mt-3 flex flex-wrap gap-2">
                  <button
                    class="rounded-full border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-100 hover:bg-cyan-500/16"
                    type="button"
                    @click="useAiSupportKpids"
                  >
                    填充 AI 证据
                  </button>
                  <button
                    class="rounded-full border border-slate-700 bg-slate-950/70 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-cyan-400/50"
                    type="button"
                    @click="clearSupportKpids"
                  >
                    清空
                  </button>
                </div>
                <div class="mt-3 max-h-56 space-y-2 overflow-y-auto pr-1">
                  <label
                    v-for="option in currentItem.candidateKpidOptions"
                    :key="`support-${option.value}`"
                    class="flex items-start gap-3 rounded-2xl border border-slate-800 bg-slate-950/70 px-3 py-2.5 text-sm text-slate-200"
                  >
                    <input
                      class="mt-1 accent-cyan-400"
                      type="checkbox"
                      :checked="currentDecision()?.finalSupportKpids.includes(option.value)"
                      @change="toggleSupportKpid(option.value, ($event.target as HTMLInputElement).checked)"
                    />
                    <span class="leading-6">
                      {{ option.isEvidence ? 'AI证据 | ' : '' }}{{ option.label }}
                    </span>
                  </label>
                </div>
              </div>

              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">kpid复核状态</span>
                <select
                  :value="currentDecision()?.kpidStatus"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @change="updateDecision('kpidStatus', ($event.target as HTMLSelectElement).value)"
                >
                  <option value="">未设置</option>
                  <option
                    v-for="option in REVIEW_STATUS_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.label }}
                  </option>
                </select>
              </label>
            </div>
          </div>
        </div>

        <div class="mt-4 grid gap-4 xl:grid-cols-2">
          <div class="rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">标签判断</div>
            <div class="mt-4 space-y-4">
              <label class="flex flex-col gap-2">
                <span class="flex items-center justify-between text-sm font-semibold text-slate-200">
                  <span>最终 query_scope</span>
                  <span class="text-xs font-medium text-slate-500">
                    {{ suggestedValueChip(currentDecision()?.finalQueryScope || '', currentItem.suggested.queryScope) }}
                  </span>
                </span>
                <select
                  :value="currentDecision()?.finalQueryScope"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @focus="activeGuideKey = 'query_scope'"
                  @change="updateDecision('finalQueryScope', ($event.target as HTMLSelectElement).value)"
                >
                  <option value="">未设置</option>
                  <option
                    v-for="option in QUERY_SCOPE_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.label }}
                  </option>
                </select>
              </label>

              <label class="flex flex-col gap-2">
                <span class="flex items-center justify-between text-sm font-semibold text-slate-200">
                  <span>最终 preferred_granularity</span>
                  <span class="text-xs font-medium text-slate-500">
                    {{ suggestedValueChip(currentDecision()?.finalPreferredGranularity || '', currentItem.suggested.preferredGranularity) }}
                  </span>
                </span>
                <select
                  :value="currentDecision()?.finalPreferredGranularity"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @focus="activeGuideKey = 'preferred_granularity'"
                  @change="updateDecision('finalPreferredGranularity', ($event.target as HTMLSelectElement).value)"
                >
                  <option value="">未设置</option>
                  <option
                    v-for="option in PREFERRED_GRANULARITY_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.label }}
                  </option>
                </select>
              </label>

              <label class="flex flex-col gap-2">
                <span class="flex items-center justify-between text-sm font-semibold text-slate-200">
                  <span>最终 support_pattern</span>
                  <span class="text-xs font-medium text-slate-500">
                    {{ suggestedValueChip(currentDecision()?.finalSupportPattern || '', currentItem.suggested.supportPattern) }}
                  </span>
                </span>
                <select
                  :value="currentDecision()?.finalSupportPattern"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @focus="activeGuideKey = 'support_pattern'"
                  @change="handleSupportPatternChange(($event.target as HTMLSelectElement).value)"
                >
                  <option value="">未设置</option>
                  <option
                    v-for="option in SUPPORT_PATTERN_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.label }}
                  </option>
                </select>
              </label>

              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">最终 granularity_sensitive</span>
                <select
                  :value="boolToFormValue(currentDecision()?.finalGranularitySensitive)"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @change="updateDecision('finalGranularitySensitive', formValueToBool(($event.target as HTMLSelectElement).value))"
                >
                  <option value="">未设置</option>
                  <option
                    v-for="option in BOOLEAN_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.label }}
                  </option>
                </select>
              </label>
            </div>
          </div>

          <div class="rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">收口判断</div>
            <div class="mt-4 space-y-4">
              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">是否纳入正式集</span>
                <select
                  :value="currentDecision()?.includeInFormalSet"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none focus:border-cyan-400/70"
                  @change="updateDecision('includeInFormalSet', ($event.target as HTMLSelectElement).value)"
                >
                  <option value="">未设置</option>
                  <option
                    v-for="option in INCLUDE_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                  >
                    {{ option.label }}
                  </option>
                </select>
              </label>

              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">复核人</span>
                <input
                  :value="currentDecision()?.reviewer"
                  class="rounded-2xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-white outline-none placeholder:text-slate-500 focus:border-cyan-400/70"
                  placeholder="默认会使用上方的默认复核人"
                  type="text"
                  @input="updateDecision('reviewer', ($event.target as HTMLInputElement).value)"
                />
              </label>

              <label class="flex flex-col gap-2">
                <span class="text-sm font-semibold text-slate-200">复核备注</span>
                <textarea
                  :value="currentDecision()?.notes"
                  class="min-h-[132px] rounded-2xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm leading-6 text-white outline-none placeholder:text-slate-500 focus:border-cyan-400/70"
                  placeholder="可直接改写建议稿，不用从头写。"
                  @input="updateDecision('notes', ($event.target as HTMLTextAreaElement).value)"
                />
              </label>
            </div>
          </div>
        </div>

        <div class="mt-6 flex items-center justify-between">
          <button
            class="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/70 px-4 py-2 text-sm font-medium text-slate-100 hover:border-cyan-400/60"
            type="button"
            @click="jumpToNeighbor('prev')"
          >
            <ChevronLeft class="h-4 w-4" />
            上一条
          </button>

          <button
            class="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/70 px-4 py-2 text-sm font-medium text-slate-100 hover:border-cyan-400/60"
            type="button"
            @click="jumpToNeighbor('next')"
          >
            下一条
            <ChevronRight class="h-4 w-4" />
          </button>
        </div>
      </main>

      <aside
        v-if="currentItem"
        class="min-h-0 overflow-y-auto rounded-[28px] border border-slate-700/60 bg-slate-900/80 p-5"
      >
        <div class="rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
          <div class="text-xs uppercase tracking-[0.18em] text-slate-500">AI 证据摘要</div>
          <div class="mt-3 text-sm leading-7 text-slate-100">
            {{ currentItem.ai.evidenceText || '未提供' }}
          </div>
          <div v-if="currentItem.p0.reason" class="mt-4 rounded-2xl border border-cyan-500/20 bg-cyan-500/8 px-4 py-3 text-sm leading-6 text-cyan-100">
            <div class="text-xs uppercase tracking-[0.16em] text-cyan-300/70">P0 入选理由</div>
            <div class="mt-2">{{ currentItem.p0.reason }}</div>
          </div>
        </div>

        <div class="mt-4 rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
          <div class="flex items-center justify-between">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">候选 KP 原文</div>
            <div class="text-xs text-slate-500">
              共 {{ currentDocKps.length }} 条
            </div>
          </div>

          <div class="mt-4 space-y-3">
            <article
              v-for="kp in currentDocKps"
              :key="kp.kpid"
              class="rounded-[22px] border px-4 py-3"
              :class="kp.isSelectedMain
                ? 'border-emerald-400/60 bg-emerald-500/10'
                : kp.isSelectedSupport
                  ? 'border-amber-400/45 bg-amber-500/10'
                : kp.isEvidence
                  ? 'border-cyan-400/40 bg-cyan-500/8'
                  : 'border-slate-800 bg-slate-900/80'"
            >
              <div class="flex flex-wrap items-center gap-2 text-[11px] font-semibold">
                <span class="rounded-full bg-slate-950/70 px-2 py-1 text-slate-300">
                  {{ kp.kpid }}
                </span>
                <span
                  v-if="kp.isEvidence"
                  class="rounded-full bg-cyan-500/15 px-2 py-1 text-cyan-200"
                >
                  AI evidence
                </span>
                <span
                  v-if="kp.isSuggestedMain"
                  class="rounded-full bg-violet-500/15 px-2 py-1 text-violet-200"
                >
                  AI 主支撑建议
                </span>
                <span
                  v-if="kp.isSelectedMain"
                  class="rounded-full bg-emerald-500/15 px-2 py-1 text-emerald-200"
                >
                  当前选中主支撑
                </span>
                <span
                  v-if="kp.isSelectedSupport"
                  class="rounded-full bg-amber-500/15 px-2 py-1 text-amber-100"
                >
                  当前联合支撑
                </span>
              </div>
              <p class="mt-3 text-sm leading-7 text-slate-100">
                {{ kp.text }}
              </p>
            </article>
          </div>
        </div>
        <div class="mt-4 rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
          <div class="text-xs uppercase tracking-[0.18em] text-slate-500">原文摘要</div>
          <p class="mt-3 text-sm leading-7 text-slate-100">
            {{ currentItem.article.excerpt }}
          </p>
          <details class="mt-4 rounded-2xl border border-slate-800 bg-slate-900/80 px-4 py-3">
            <summary class="cursor-pointer text-sm font-semibold text-cyan-100">
              展开完整原文
            </summary>
            <div class="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-200">
              {{ currentItem.article.fullText }}
            </div>
          </details>
        </div>

        <div class="mt-4 rounded-[24px] border border-slate-800 bg-slate-950/65 p-4">
          <div class="flex items-center justify-between">
            <div class="text-xs uppercase tracking-[0.18em] text-slate-500">标签说明</div>
            <div class="text-xs text-slate-500">
              当前聚焦：{{ GUIDE_SECTION_MAP[activeGuideKey]?.label || 'Query Scope' }}
            </div>
          </div>

          <div class="mt-4 flex flex-wrap gap-2">
            <button
              v-for="section in GUIDE_SECTIONS"
              :key="section.key"
              class="rounded-full border px-3 py-1.5 text-xs font-semibold"
              :class="activeGuideKey === section.key
                ? 'border-cyan-400/70 bg-cyan-400/10 text-cyan-100'
                : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-cyan-400/40'"
              type="button"
              @click="activeGuideKey = section.key"
            >
              {{ section.label }}
            </button>
          </div>

          <div class="mt-4 space-y-3">
            <div
              v-for="option in GUIDE_SECTION_MAP[activeGuideKey]?.options || []"
              :key="option.value"
              class="rounded-[20px] border border-slate-800 bg-slate-900/80 p-4"
            >
              <div class="flex items-center justify-between gap-3">
                <div class="text-sm font-bold text-white">{{ option.label }}</div>
                <div class="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                  {{ option.value }}
                </div>
              </div>
              <p class="mt-2 text-sm leading-6 text-cyan-100/90">{{ option.summary }}</p>
              <p class="mt-2 text-sm leading-6 text-slate-300">{{ option.detail }}</p>
              <div v-if="option.examples?.length" class="mt-3">
                <div class="text-[11px] uppercase tracking-[0.16em] text-slate-500">例子</div>
                <div class="mt-2 flex flex-wrap gap-2">
                  <span
                    v-for="example in option.examples"
                    :key="example"
                    class="rounded-full bg-slate-950 px-3 py-1.5 text-xs text-slate-300"
                  >
                    {{ example }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </section>

    <section
      v-else
      class="rounded-[28px] border border-dashed border-slate-700/80 bg-slate-900/70 p-10 text-center text-slate-300"
    >
      <div class="mx-auto max-w-2xl">
        <div class="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
          Review Bundle
        </div>
        <h3 class="mt-3 text-2xl font-black text-white">还没有加载复核数据</h3>
        <p class="mt-3 text-sm leading-7 text-slate-400">
          先运行 `export_review_bundle.py` 生成 bundle，然后点“上传 Bundle”，或者把 bundle 写到
          `FrontEnd/public/review_bundle_latest.json` 后直接刷新页面。
        </p>
      </div>
    </section>
  </div>
</template>
