<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import {
  AlertCircle,
  Clock,
  Cpu,
  ExternalLink,
  Info,
  Loader2,
  Search
} from 'lucide-vue-next';
import localforage from 'localforage';
import VectorWorker from '../worker/embedding.worker.ts?worker';
import {
  FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
  type PipelineDecision,
  type PipelineDocumentRecord,
  type PipelineTrace,
  type SearchPipelineResult,
} from '../worker/search_pipeline.ts';
import type { SearchRejection } from '../worker/vector_engine.ts';
import {
  formatPercent,
  formatRetrievalScore,
  getDisplayScore,
  getPreviewText,
  isOriginalSnippet,
} from '../utils/searchPresentation';

type SearchResultDoc = PipelineDocumentRecord;
type WorkerDecision = PipelineDecision;
type WorkerTrace = PipelineTrace;
type WorkerSearchResult = SearchPipelineResult;


const emit = defineEmits(['trace-updated']);

const searchQuery = ref('');
const results = ref<SearchResultDoc[]>([]);
const weakResults = ref<SearchResultDoc[]>([]);
const rejectionInfo = ref<SearchRejection | null>(null);
const decisionInfo = ref<WorkerDecision | null>(null);
const showWeakResults = ref(false);
const isProcessing = ref(false);
const statusMsg = ref('正在唤起 Web Worker...');
const errorMsg = ref<string | null>(null);
const diagnosticLogs = ref<string[]>([]);
const isWorkerReady = ref(false);

const REJECTION_THRESHOLD = FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.rejectThreshold;
const ORIGINAL_SNIPPET_THRESHOLD =
  FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.bestSentenceThreshold;
const INDEX_CACHE_VERSION = '20260324-v3';

const heroResults = computed(() => results.value.slice(0, 3));
const compactResults = computed(() => results.value.slice(3, 10));
const hasCoverageRejection = computed(
  () => rejectionInfo.value?.reason === 'low_topic_coverage'
);
const hasConsistencyRejection = computed(
  () => rejectionInfo.value?.reason === 'low_consistency'
);
const hasDisplayThresholdReject = computed(
  () => decisionInfo.value?.rejectionReason === 'display_threshold'
);
const isRejectDecision = computed(
  () => decisionInfo.value?.behavior === 'reject'
);
const weakToggleLabel = computed(() =>
  showWeakResults.value ? '收起弱相关结果' : '查看弱相关结果'
);
const weakResultsTitle = computed(() => '弱相关结果');
const rejectReasonLabel = computed(() => {
  switch (decisionInfo.value?.rejectionReason ?? rejectionInfo.value?.reason) {
    case 'display_threshold':
      return '展示阈值不足';
    case 'low_topic_coverage':
      return '主题覆盖不足';
    case 'low_consistency':
      return '主题一致性不足';
    case 'invalid_input':
      return '输入无效';
    default:
      return '当前不适合直接回答';
  }
});
const rejectTierLabel = computed(() => {
  switch (decisionInfo.value?.rejectTier) {
    case 'hard_reject':
      return '硬拒答';
    case 'boundary_uncertain':
      return '边界不确定';
    case 'invalid_input':
      return '无效输入';
    default:
      return '';
  }
});
const rejectDescription = computed(() => {
  if (hasCoverageRejection.value) {
    return '当前知识库内没有形成同主题、可直接回答的稳定锚点，因此系统只保留弱相关入口，不直接给出答案。';
  }
  if (hasConsistencyRejection.value) {
    return '当前候选结果主题分散，或者证据只停留在弱相似层，系统拒绝输出不稳定答案。';
  }
  if (hasDisplayThresholdReject.value) {
    return `虽然召回到了候选文档，但 Top 1 原话匹配度没有达到 ${(REJECTION_THRESHOLD * 100).toFixed(0)}% 的展示阈值。`;
  }
  return '当前问题没有达到可安全展示的直接回答条件，系统进入拒答模式。';
});
const emptyTitle = computed(() => {
  if (hasCoverageRejection.value) {
    return '当前知识库暂无该主题的直接内容，暂不展示弱相关结果。';
  }
  if (hasConsistencyRejection.value) {
    return '当前查询未能形成稳定的主题结果，暂不展示不可靠答案。';
  }
  if (hasDisplayThresholdReject.value) {
    return '当前问题未达到可信展示阈值。';
  }
  return searchQuery.value.trim() ? '当前问题未达到可信展示阈值' : '输入关键词开始检索';
});
const emptySubtitle = computed(() => {
  if (hasCoverageRejection.value) {
    return '可展开查看系统筛出的弱相关结果，但这些结果不属于同主题直接答案。';
  }
  if (hasConsistencyRejection.value) {
    return '当前结果主题分散或仅靠弱语义相似命中，因此系统选择拒答。';
  }
  if (hasDisplayThresholdReject.value) {
    return `Top 1 原话匹配度需不低于 ${(REJECTION_THRESHOLD * 100).toFixed(0)}%`;
  }
  return searchQuery.value.trim()
    ? `Top 1 原话匹配度需不低于 ${(REJECTION_THRESHOLD * 100).toFixed(0)}%`
    : 'Top 1-3 展示原话，Top 4-10 展示紧凑列表';
});

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const getHighlightTerms = (query: string) =>
  Array.from(
    new Set(
      query
        .split(/[\s,，。；;、]+/)
        .map((item) => item.trim())
        .filter((item) => item.length >= 2)
    )
  ).sort((a, b) => b.length - a.length);

const highlightSnippet = (text: string, query: string) => {
  const safeText = escapeHtml(text || '');
  const terms = getHighlightTerms(query);
  if (!safeText || terms.length === 0) return safeText;

  const pattern = terms.map(escapeRegExp).join('|');
  return safeText.replace(
    new RegExp(`(${pattern})`, 'gi'),
    '<mark class="rounded bg-blue-500/20 px-0.5 text-blue-100">$1</mark>'
  );
};

const logDiagnostic = (msg: string) => {
  const time = new Date().toLocaleTimeString();
  const formatted = `[${time}] ${msg}`;
  diagnosticLogs.value.unshift(formatted);
  if (diagnosticLogs.value.length > 5) diagnosticLogs.value.pop();
};

const emitTrace = (
  query: string,
  traceResults: SearchResultDoc[],
  opts: {
    totalMs: string;
    searchMs: string;
    fetchMs?: string;
    rerankMs?: string;
    rerankedDocCount?: number;
    chunksScored?: number;
    rerankWindowReason?: string;
    maxChunksPerDoc?: number;
    chunkPlanReason?: string;
    initialTopConfidence?: number | null;
    topConfidence?: number | null;
    rejected?: boolean;
    rejection?: SearchRejection | null;
    weakResultsCount?: number;
    retrievalDecision?: WorkerDecision | null;
    decision?: WorkerDecision | null;
    directAnswerRescue?: WorkerTrace['directAnswerRescue'];
    querySignals?: WorkerTrace['querySignals'];
    retrievalSignals?: WorkerTrace['retrievalSignals'];
  }
) => {
  emit('trace-updated', {
    query,
    results: traceResults,
    retrievalDecision: opts.retrievalDecision ?? null,
    stats: {
      totalMs: opts.totalMs,
      searchMs: opts.searchMs,
      fetchMs: opts.fetchMs ?? '0.0',
      rerankMs: opts.rerankMs ?? '0.0',
      rerankedDocCount: opts.rerankedDocCount ?? 0,
      chunksScored: opts.chunksScored ?? 0,
      rerankWindowReason: opts.rerankWindowReason ?? '',
      maxChunksPerDoc: opts.maxChunksPerDoc ?? 0,
      chunkPlanReason: opts.chunkPlanReason ?? '',
      initialTopConfidence: opts.initialTopConfidence ?? null,
      rejectionThreshold: REJECTION_THRESHOLD,
      topConfidence: opts.topConfidence ?? null,
      rejected: opts.rejected ?? false,
      rejection: opts.rejection ?? null,
      weakResultsCount: opts.weakResultsCount ?? 0,
    },
    decision: opts.decision ?? null,
    rejection: opts.rejection ?? null,
    weakResultsCount: opts.weakResultsCount ?? 0,
    directAnswerRescue: opts.directAnswerRescue ?? null,
    querySignals: opts.querySignals ?? null,
    retrievalSignals: opts.retrievalSignals ?? null,
  });
};

const pendingTasks = new Map<
  string,
  { resolve: (value: any) => void; reject: (error: Error) => void; type: string }
>();

const myWorker = new VectorWorker();

const dispatchToWorker = (type: string, payload: any, transfer: Transferable[] = []): Promise<any> =>
  new Promise((resolve, reject) => {
    const taskId = crypto.randomUUID();
    pendingTasks.set(taskId, { resolve, reject, type });
    myWorker.postMessage({ type, payload, taskId }, transfer);
  });

myWorker.onmessage = (event: MessageEvent) => {
  const { status, message, result, error, stats, taskId } = event.data;

  if (status === 'loading' || status === 'progress') {
    statusMsg.value = message;
    logDiagnostic(`[引擎通知] ${message}`);
    return;
  }

  if (status === 'info') {
    logDiagnostic(`[系统提示] ${message}`);
    return;
  }

  if (status === 'ready') {
    isWorkerReady.value = true;
    statusMsg.value = 'AI 引擎就绪';
    logDiagnostic(message);
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.resolve(true);
      pendingTasks.delete(taskId);
    }
    return;
  }

  if (status === 'error') {
    isProcessing.value = false;
    errorMsg.value = `引擎启动失败: ${error}`;
    logDiagnostic(`致命错误: ${error}`);
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.reject(new Error(error));
      pendingTasks.delete(taskId);
    }
    return;
  }

  if (status === 'search_complete') {
    logDiagnostic(`统一链路完成主搜索阶段，扫描 ${stats.itemsScanned} 条，耗时 ${stats.elapsedMs}ms`);
    if (stats?.partitionUsed) {
      logDiagnostic(`[Partition] candidate=${stats.partitionCandidateCount ?? '-'}`);
    }
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.resolve(result);
      pendingTasks.delete(taskId);
    }
    return;
  }
};

const initWorkers = async () => {
  logDiagnostic('开始检查核心数据缓存...');
  statusMsg.value = '加载核心数据中...';

  try {
    const CACHE_KEY_MATRIX = `rag_vector_matrix_dmeta_${INDEX_CACHE_VERSION}`;
    const CACHE_KEY_METADATA = `rag_metadata_dmeta_${INDEX_CACHE_VERSION}`;

    let matrixBuffer = await localforage.getItem<ArrayBuffer>(CACHE_KEY_MATRIX);
    let metadataJson = await localforage.getItem<any>(CACHE_KEY_METADATA);

    if (!matrixBuffer || !metadataJson) {
      logDiagnostic('本地缓存未命中，开始同步数据...');
      const [matrixRes, metaRes] = await Promise.all([
        fetch('/data/frontend_vectors_dmeta_small.bin'),
        fetch('/data/frontend_metadata_dmeta_small.json')
      ]);

      if (!matrixRes.ok || !metaRes.ok) {
        throw new Error('网络请求核心数据失败');
      }

      matrixBuffer = await matrixRes.arrayBuffer();
      metadataJson = await metaRes.json();

      logDiagnostic('同步完成，正在写入 IndexedDB...');
      await localforage.setItem(CACHE_KEY_MATRIX, matrixBuffer);
      await localforage.setItem(CACHE_KEY_METADATA, metadataJson);
    } else {
      logDiagnostic('已从 IndexedDB 命中本地缓存');
    }

    const transferBuffer = matrixBuffer.slice(0) as ArrayBuffer;
    await dispatchToWorker(
      'INIT',
      {
        metadata: metadataJson,
        vectorMatrix: new Int8Array(transferBuffer)
      },
      [transferBuffer]
    );
  } catch (e: any) {
    errorMsg.value = `加载失败: ${e.message}`;
    logDiagnostic(`初始化失败: ${e.message}`);
  }
};

onMounted(() => {
  initWorkers();
});

onUnmounted(() => {
  myWorker.terminate();
});

const handleSearch = async () => {
  const query = searchQuery.value.trim();
  if (!query || isProcessing.value || !isWorkerReady.value) return;

  errorMsg.value = null;
  isProcessing.value = true;
  results.value = [];
  weakResults.value = [];
  rejectionInfo.value = null;
  decisionInfo.value = null;
  showWeakResults.value = false;
  statusMsg.value = '正在分词与向量化...';

  try {
    logDiagnostic('开始提交检索请求...');

    const searchResult = (await dispatchToWorker('SEARCH', query)) as WorkerSearchResult;
    const finalRender = searchResult?.results || [];
    const localWeakResults = searchResult?.weakResults || [];
    const rejection = searchResult?.rejection || null;
    const retrievalDecision = searchResult?.retrievalDecision || null;
    const finalDecision = searchResult?.finalDecision || null;
    const trace = searchResult?.trace;

    results.value = finalRender;
    weakResults.value = localWeakResults;
    rejectionInfo.value = rejection;
    decisionInfo.value = finalDecision;

    emitTrace(query, finalRender, {
      totalMs: Number(trace?.totalMs ?? 0).toFixed(1),
      searchMs: Number(trace?.searchMs ?? 0).toFixed(1),
      fetchMs: Number(trace?.fetchMs ?? 0).toFixed(1),
      rerankMs: Number(trace?.rerankMs ?? 0).toFixed(1),
      rerankedDocCount: trace?.rerankedDocCount,
      chunksScored: trace?.chunksScored,
      rerankWindowReason: trace?.rerankWindowReason,
      maxChunksPerDoc: trace?.maxChunksPerDoc,
      chunkPlanReason: trace?.chunkPlanReason,
      initialTopConfidence: trace?.initialTopConfidence ?? null,
      topConfidence: trace?.topConfidence ?? null,
      rejected: finalDecision?.behavior === 'reject',
      rejection,
      weakResultsCount: localWeakResults.length,
      retrievalDecision,
      decision: finalDecision,
      directAnswerRescue: trace?.directAnswerRescue,
      querySignals: trace?.querySignals,
      retrievalSignals: trace?.retrievalSignals,
    });

    if (!finalDecision) {
      statusMsg.value = '未能解析当前查询结果';
      logDiagnostic('未收到统一链路的最终决策');
      return;
    }

    if (retrievalDecision && retrievalDecision.behavior !== finalDecision.behavior) {
      logDiagnostic(
        `展示阶段改写行为：${retrievalDecision.behavior} -> ${finalDecision.behavior}`
      );
    }

    if (finalDecision.behavior === 'reject') {
      if (finalDecision.rejectionReason === 'display_threshold') {
        statusMsg.value = `未达到展示阈值 ${(REJECTION_THRESHOLD * 100).toFixed(0)}%，已拒答`;
        logDiagnostic(
          `展示阶段拒答：初始 ${formatPercent(trace?.initialTopConfidence ?? 0)} / 最终 ${formatPercent(trace?.topConfidence ?? 0)}`
        );
        if (trace?.directAnswerRescue?.attempted) {
          logDiagnostic(
            `直答补救重排未保留：${trace.directAnswerRescue.reason || '补救后仍未通过阈值'}`
          );
        }
      } else if (rejection?.reason === 'low_topic_coverage') {
        statusMsg.value = '当前知识库暂无该主题的直接内容，已给出弱相关入口。';
        logDiagnostic('主题覆盖不足，已拒绝直接回答并保留弱相关入口');
      } else {
        statusMsg.value = '当前查询未能形成稳定的可信结果，系统已拒答。';
        logDiagnostic('系统判定当前查询应拒答');
      }
      return;
    }

    if (trace?.directAnswerRescue?.attempted) {
      if (trace.directAnswerRescue.succeeded) {
        logDiagnostic(
          `直答补救重排成功：${formatPercent(trace.directAnswerRescue.initialTopConfidence ?? 0)} -> ${formatPercent(trace.directAnswerRescue.rescueTopConfidence ?? 0)}`
        );
      } else {
        logDiagnostic(
          `直答补救重排未生效：${trace.directAnswerRescue.reason || '未满足保留条件'}`
        );
      }
    }

    statusMsg.value = `找到 ${finalRender.length} 篇相关结果（总耗时 ${Number(trace?.totalMs ?? 0).toFixed(1)}ms）`;
    if (trace?.directAnswerRescue?.succeeded) {
      statusMsg.value = `找到 ${finalRender.length} 篇相关结果（经补救重排保留，耗时 ${Number(trace?.totalMs ?? 0).toFixed(1)}ms）`;
    }
    logDiagnostic('统一链路已完成结果展示');
  } catch (e: any) {
    console.error(e);
    errorMsg.value = `检索发生错误: ${e.message}`;
    logDiagnostic(`检索异常: ${e.message}`);
  } finally {
    isProcessing.value = false;
  }
};
</script>

<template>
  <div class="flex h-full flex-col overflow-hidden rounded-[24px] border border-white/6 bg-white/[0.04] shadow-[0_24px_80px_rgba(0,0,0,0.28)] backdrop-blur-xl">
    <div class="flex items-center justify-between border-b border-white/8 bg-white/[0.03] px-5 py-3">
      <div class="flex items-center gap-2">
        <span class="rounded-lg border border-white/6 bg-black/20 px-2.5 py-1 text-[9px] font-bold uppercase text-slate-400">DMeta Soul · WASM</span>
      </div>

      <div class="flex items-center gap-2 pl-4 text-right">
        <div class="h-1.5 w-1.5 rounded-full" :class="isProcessing ? 'animate-pulse bg-amber-400' : 'bg-emerald-400'" />
        <span class="line-clamp-1 text-[10px] font-semibold tracking-[0.16em] text-slate-500">{{ statusMsg }}</span>
      </div>
    </div>

    <div
      v-if="errorMsg"
      class="flex items-center gap-2 border-b border-red-500/20 bg-red-500/10 px-5 py-3 text-xs text-red-400"
    >
      <AlertCircle class="h-4 w-4" />
      {{ errorMsg }}
    </div>

    <div class="border-b border-white/6 bg-white/[0.03] p-5">
      <div class="group relative">
        <input
          v-model="searchQuery"
          @keydown.enter="handleSearch"
          placeholder="输入校园政策关键词或具体问题"
          class="w-full rounded-2xl border border-white/8 bg-slate-950/40 px-12 py-4 font-medium text-slate-100 outline-none transition-all placeholder:text-slate-500 focus:border-blue-400/30 focus:ring-2 focus:ring-blue-500/30"
        />
        <Search class="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-500" />
        <button
          @click="handleSearch"
          :disabled="!searchQuery.trim() || isProcessing || !isWorkerReady"
          class="absolute right-2 top-1/2 -translate-y-1/2 rounded-xl bg-blue-600 px-4 py-2 text-xs font-bold transition-all hover:bg-blue-500 active:scale-95 disabled:opacity-50"
        >
          <span v-if="!isProcessing">搜索</span>
          <Loader2 v-else class="h-4 w-4 animate-spin" />
        </button>
      </div>

      <div class="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
        <div
          v-for="(log, i) in diagnosticLogs"
          :key="i"
          class="flex items-center gap-1 font-mono text-[9px] text-slate-500"
        >
          <Info v-if="i === 0" class="h-2.5 w-2.5 text-blue-400" />
          {{ log }}
        </div>
      </div>
    </div>

    <div class="custom-scrollbar flex-1 overflow-y-auto px-5 py-4">
      <div
        v-if="results.length === 0 && !isProcessing && !isRejectDecision"
        class="flex h-full flex-col items-center justify-center text-center text-slate-500"
      >
        <div class="mb-4 flex h-14 w-14 items-center justify-center rounded-full border border-dashed border-slate-700">
          <Search class="h-5 w-5" />
        </div>
        <p class="text-sm font-medium">{{ emptyTitle }}</p>
        <p class="mt-1 text-[11px] text-slate-600">{{ emptySubtitle }}</p>
      </div>

      <section
        v-if="results.length === 0 && !isProcessing && isRejectDecision"
        class="mt-1 rounded-[22px] border border-rose-400/18 bg-rose-400/8 p-4"
      >
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="mb-2 flex flex-wrap items-center gap-2">
              <span class="rounded-full border border-rose-400/20 bg-rose-400/12 px-2.5 py-1 text-[10px] font-semibold text-rose-200">
                拒答
              </span>
              <span class="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-[10px] text-slate-300">
                {{ rejectReasonLabel }}
              </span>
              <span
                v-if="rejectTierLabel"
                class="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-[10px] text-slate-400"
              >
                {{ rejectTierLabel }}
              </span>
            </div>
            <p class="text-sm font-medium text-rose-100">当前查询不返回直接答案</p>
            <p class="mt-2 text-[12px] leading-6 text-rose-100/80">
              {{ rejectDescription }}
            </p>
          </div>

          <div class="min-w-[120px] rounded-2xl border border-white/8 bg-black/20 px-3 py-2 text-right">
            <div class="text-[10px] uppercase tracking-[0.16em] text-slate-500">阈值</div>
            <div class="mt-1 font-mono text-[13px] text-slate-100">
              {{ (REJECTION_THRESHOLD * 100).toFixed(0) }}%
            </div>
            <div
              v-if="weakResults.length > 0"
              class="mt-2 text-[10px] text-slate-400"
            >
              弱相关 {{ weakResults.length }} 条
            </div>
          </div>
        </div>
      </section>

      <section
        v-if="results.length === 0 && !isProcessing && hasCoverageRejection && weakResults.length > 0"
        class="mt-6 space-y-3"
      >
        <div class="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-500">
          <div class="flex items-center gap-3">
            <span>{{ weakResultsTitle }}</span>
          </div>
          <button
            @click="showWeakResults = !showWeakResults"
            class="rounded-xl border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[10px] font-semibold tracking-normal text-slate-300 transition-all hover:border-white/20 hover:bg-white/[0.08]"
          >
            {{ weakToggleLabel }}
          </button>
        </div>

        <div v-if="showWeakResults" class="space-y-1.5">
          <details
            v-for="(res, index) in weakResults"
            :key="res.otid || res.id || `weak-${index}`"
            class="group rounded-2xl border border-white/5 bg-white/[0.025] px-4 py-2.5"
          >
            <summary class="flex list-none cursor-pointer items-center justify-between gap-3">
              <div class="min-w-0 flex items-center gap-3">
                <span class="text-[10px] font-semibold text-slate-500">#{{ index + 1 }}</span>
                <span class="truncate text-[13px] text-slate-200">{{ res.ot_title || '未命名文档' }}</span>
              </div>
              <div class="flex shrink-0 items-center gap-3">
                <span class="font-mono text-[11px] text-slate-500">{{ formatRetrievalScore(getDisplayScore(res)) }}</span>
                <span class="text-[10px] text-slate-600 group-open:hidden">展开</span>
              </div>
            </summary>

            <div
              v-if="res.bestPoint || getPreviewText(res)"
              class="mt-2 rounded-xl bg-white/[0.03] px-3 py-2.5 text-[13px] leading-6 text-slate-400"
            >
              <div class="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-600">弱相关片段</div>
              <p>{{ res.bestPoint || getPreviewText(res) }}</p>
            </div>
          </details>
        </div>
      </section>

      <div v-if="results.length > 0" class="space-y-5">
        <section v-if="heroResults.length > 0" class="space-y-3">
          <div class="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-500">
            <span>Top 1-3</span>
            <span>原话优先</span>
          </div>

          <article
            v-for="(res, i) in heroResults"
            :key="res.otid || res.id || i"
            class="rounded-[20px] border border-white/6 bg-white/[0.045] px-4 py-3.5"
          >
            <div class="flex items-start justify-between gap-3">
              <div class="min-w-0">
                <div class="mb-2 flex items-center gap-2">
                  <span class="text-[10px] font-semibold text-slate-500">#{{ i + 1 }}</span>
                  <span class="rounded-full border border-white/8 px-2 py-0.5 text-[10px] text-slate-400">
                    {{ isOriginalSnippet(res, ORIGINAL_SNIPPET_THRESHOLD) ? '官方原话' : '相关要点' }}
                  </span>
                </div>
                <h3 class="text-[15px] font-semibold leading-6 text-slate-100">
                  {{ res.ot_title || '未命名政策文档' }}
                </h3>
              </div>

              <div class="shrink-0 text-right">
                <div v-if="isOriginalSnippet(res, ORIGINAL_SNIPPET_THRESHOLD)" class="font-mono text-[11px] text-slate-300">
                  {{ formatPercent(res.snippetScore ?? 0) }}
                </div>
                <div v-else class="font-mono text-[11px] text-slate-400">
                  {{ formatRetrievalScore(getDisplayScore(res)) }}
                </div>
                <div class="mt-1 text-[10px] text-slate-500">
                  {{ isOriginalSnippet(res, ORIGINAL_SNIPPET_THRESHOLD) ? '原话匹配度' : '检索分' }}
                </div>
              </div>
            </div>

            <div
              v-if="getPreviewText(res)"
              class="mt-3 text-sm leading-7 text-slate-200"
              v-html="highlightSnippet(getPreviewText(res), searchQuery)"
            />

            <div class="mt-3 flex items-center gap-4 text-[11px] text-slate-500">
              <div v-if="res.publish_time" class="flex items-center gap-1.5">
                <Clock class="h-3 w-3" />
                <span>{{ res.publish_time }}</span>
              </div>
              <a
                v-if="res.link"
                :href="res.link"
                target="_blank"
                class="inline-flex items-center gap-1.5 text-slate-400 hover:text-slate-200"
              >
                <span>原文</span>
                <ExternalLink class="h-3 w-3" />
              </a>
            </div>
          </article>
        </section>

        <section v-if="compactResults.length > 0" class="space-y-1.5">
          <div class="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-500">
            <span>Top 4-10</span>
            <span>轻列表</span>
          </div>

          <details
            v-for="(res, index) in compactResults"
            :key="res.otid || res.id || `compact-${index}`"
            class="group rounded-2xl border border-white/5 bg-white/[0.025] px-4 py-2.5"
          >
            <summary class="flex list-none cursor-pointer items-center justify-between gap-3">
              <div class="min-w-0 flex items-center gap-3">
                <span class="text-[10px] font-semibold text-slate-500">#{{ index + 4 }}</span>
                <span class="truncate text-[13px] text-slate-200">{{ res.ot_title || '未命名政策文档' }}</span>
              </div>
              <div class="flex shrink-0 items-center gap-3">
                <span class="font-mono text-[11px] text-slate-500">{{ formatRetrievalScore(getDisplayScore(res)) }}</span>
                <span class="text-[10px] text-slate-600 group-open:hidden">展开</span>
              </div>
            </summary>

            <div
              v-if="res.bestPoint || getPreviewText(res)"
              class="mt-2 rounded-xl bg-white/[0.03] px-3 py-2.5 text-[13px] leading-6 text-slate-400"
            >
              <div class="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-600">相关要点</div>
              <p>{{ res.bestPoint || getPreviewText(res) }}</p>
            </div>
          </details>
        </section>
      </div>
    </div>

    <div class="flex items-center justify-between border-t border-white/5 bg-black/10 px-5 py-2.5 text-[9px] font-bold uppercase tracking-widest text-slate-600">
      <span>本地政策检索引擎</span>
      <div class="flex items-center gap-3">
        <span class="flex items-center gap-1" title="向量粗排">
          <Cpu class="h-2.5 w-2.5 text-slate-500" />
          Vector Worker
        </span>
        <span class="flex items-center gap-1" title="Top 5/10 自适应原文重排与高亮提炼">
          <svg class="h-3 w-3 text-red-500/70" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
          Snippet
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}

.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent;
}

.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
}

.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.1);
}
</style>
