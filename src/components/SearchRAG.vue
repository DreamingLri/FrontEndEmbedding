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

type SearchRejection = {
  reason: 'low_topic_coverage' | 'low_consistency';
  topicIds: string[];
};

type SearchResultDoc = {
  id?: string;
  otid?: string;
  ot_title?: string;
  ot_text?: string;
  link?: string;
  publish_time?: string;
  bestSentence?: string;
  bestPoint?: string;
  best_kpid?: string;
  kps?: Array<{ kpid?: string; kp_text?: string }>;
  score?: number;
  coarseScore?: number;
  displayScore?: number;
  rerankScore?: number;
  confidenceScore?: number;
  snippetScore?: number;
};

type WorkerSearchResult = {
  matches: Array<{ otid: string; score: number; best_kpid?: string }>;
  weakMatches?: Array<{ otid: string; score: number; best_kpid?: string }>;
  rejection?: SearchRejection;
};



const emit = defineEmits(['trace-updated']);

const searchQuery = ref('');
const results = ref<SearchResultDoc[]>([]);
const weakResults = ref<SearchResultDoc[]>([]);
const rejectionInfo = ref<SearchRejection | null>(null);
const showWeakResults = ref(false);
const isProcessing = ref(false);
const statusMsg = ref('正在唤起 Web Worker...');
const errorMsg = ref<string | null>(null);
const diagnosticLogs = ref<string[]>([]);
const isWorkerReady = ref(false);

const REJECTION_THRESHOLD = 0.4;
const INDEX_CACHE_VERSION = '20260324-v3';

const heroResults = computed(() => results.value.slice(0, 3));
const compactResults = computed(() => results.value.slice(3, 10));
const hasCoverageRejection = computed(
  () => rejectionInfo.value?.reason === 'low_topic_coverage'
);
const hasConsistencyRejection = computed(
  () => rejectionInfo.value?.reason === 'low_consistency'
);
const weakToggleLabel = computed(() =>
  showWeakResults.value ? '收起弱相关结果' : '查看弱相关结果'
);
const emptyTitle = computed(() => {
  if (hasCoverageRejection.value) {
    return '当前知识库暂无该主题的直接内容，暂不展示弱相关结果。';
  }
  if (hasConsistencyRejection.value) {
    return '当前查询未能形成稳定的主题结果，暂不展示不可靠答案。';
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

const getPreviewText = (res: SearchResultDoc) => {
  if (res.bestSentence) return res.bestSentence;
  if (res.bestPoint) return res.bestPoint;

  if (res.best_kpid && Array.isArray(res.kps)) {
    const hitKp = res.kps.find((kp) => kp.kpid === res.best_kpid);
    if (hitKp?.kp_text) return hitKp.kp_text;
  }

  return '';
};

const isOriginalSnippet = (res: SearchResultDoc) => (res.snippetScore ?? -999) > REJECTION_THRESHOLD;
const getDisplayScore = (res: SearchResultDoc) =>
  res.displayScore ?? res.coarseScore ?? res.confidenceScore ?? res.rerankScore ?? 0;
const formatRetrievalScore = (score: number) => `Score ${Number(score || 0).toFixed(2)}`;
const formatPercent = (score: number) => `${((score || 0) * 100).toFixed(1)}%`;

const mergeCoarseMatchesIntoDocuments = (
  documents: SearchResultDoc[],
  coarseMatches: Array<{ otid: string; score: number; best_kpid?: string }>
) => {
  const documentMap = new Map(
    documents.map((doc) => [doc.otid || doc.id || '', doc])
  );

  return coarseMatches
    .map((match) => {
      const doc = documentMap.get(match.otid);
      if (!doc) return null;

      return {
        ...doc,
        score: match.score ?? doc.score,
        coarseScore: match.score ?? doc.coarseScore ?? doc.score,
        displayScore: match.score ?? doc.displayScore ?? doc.score,
        best_kpid: match.best_kpid ?? doc.best_kpid
      };
    })
    .filter(Boolean) as SearchResultDoc[];
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
    topConfidence?: number | null;
    rejected?: boolean;
    rejection?: SearchRejection | null;
    weakResultsCount?: number;
  }
) => {
  emit('trace-updated', {
    query,
    results: traceResults,
    stats: {
      totalMs: opts.totalMs,
      searchMs: opts.searchMs,
      fetchMs: opts.fetchMs ?? '0.0',
      rerankMs: opts.rerankMs ?? '0.0',
      rejectionThreshold: REJECTION_THRESHOLD,
      topConfidence: opts.topConfidence ?? null,
      rejected: opts.rejected ?? false,
      rejection: opts.rejection ?? null,
      weakResultsCount: opts.weakResultsCount ?? 0,
    },
    rejection: opts.rejection ?? null,
    weakResultsCount: opts.weakResultsCount ?? 0,
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
    logDiagnostic(`粗排结束，扫描 ${stats.itemsScanned} 条，耗时 ${stats.elapsedMs}ms`);
    if (stats?.partitionUsed) {
      logDiagnostic(`[Partition] candidate=${stats.partitionCandidateCount ?? '-'}`);
    }
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.resolve(result);
      pendingTasks.delete(taskId);
    }
    return;
  }

  if (status === 'rerank_complete') {
    logDiagnostic(`重排与原话提炼完成，耗时 ${stats.elapsedMs}ms`);
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.resolve(result);
      pendingTasks.delete(taskId);
    }
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
  showWeakResults.value = false;
  statusMsg.value = '正在分词与向量化...';

  try {
    const tStart = performance.now();
    logDiagnostic('开始提交检索请求...');

    const searchResult = (await dispatchToWorker('SEARCH', query)) as WorkerSearchResult;
    const localMatches = searchResult?.matches || [];
    const localWeakMatches = searchResult?.weakMatches || [];
    const rejection = searchResult?.rejection || null;
    const tSearchEnd = performance.now();

    if (
      rejection?.reason === 'low_consistency' &&
      localMatches.length === 0 &&
      localWeakMatches.length === 0
    ) {
      results.value = [];
      weakResults.value = [];
      rejectionInfo.value = rejection;

      emitTrace(query, [], {
        totalMs: (tSearchEnd - tStart).toFixed(1),
        searchMs: (tSearchEnd - tStart).toFixed(1),
        rejected: true,
        rejection,
      });

      statusMsg.value = '当前查询未能形成稳定的主题结果，系统已拒答。';
      logDiagnostic('结果主题分散或仅靠弱语义相似命中，已拒答');
      return;
    }

    const topIds = localMatches.slice(0, 15).map((r) => r.otid);
    const weakTopIds = localWeakMatches.slice(0, 10).map((r) => r.otid);
    const fetchIds = Array.from(new Set([...topIds, ...weakTopIds]));

    logDiagnostic(`粗排返回 ${localMatches.length} 篇，取 Top ${Math.min(fetchIds.length, 15)} 进入后续处理`);

    if (fetchIds.length === 0) {
      statusMsg.value = '未找到匹配答案';
      logDiagnostic('本地检索未命中结果');
      emitTrace(query, [], {
        totalMs: (performance.now() - tStart).toFixed(1),
        searchMs: (tSearchEnd - tStart).toFixed(1),
      });
      return;
    }

    statusMsg.value = '正在请求原文数据...';
    const response = await fetch('/api/get_answers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ otids: fetchIds })
    });

    if (!response.ok) {
      throw new Error(`后端响应异常: ${response.status}`);
    }

    const result = await response.json();
    const tFetchEnd = performance.now();

    if (!result.data) {
      statusMsg.value = '未找到匹配答案';
      return;
    }

    const docsForRender = mergeCoarseMatchesIntoDocuments(result.data, localMatches.slice(0, 15));
    const weakDocsForRender = mergeCoarseMatchesIntoDocuments(result.data, localWeakMatches.slice(0, 10));

    if (rejection?.reason === 'low_topic_coverage') {
      results.value = [];
      weakResults.value = weakDocsForRender;
      rejectionInfo.value = rejection;

      emitTrace(query, [], {
        totalMs: (tFetchEnd - tStart).toFixed(1),
        searchMs: (tSearchEnd - tStart).toFixed(1),
        fetchMs: (tFetchEnd - tSearchEnd).toFixed(1),
        rejected: true,
        rejection,
        weakResultsCount: weakDocsForRender.length,
      });

      statusMsg.value = '当前知识库暂无该主题的直接内容，暂不展示弱相关结果。';
      logDiagnostic('主题覆盖不足，已提供弱相关结果入口');
      return;
    }

    statusMsg.value = '正在重排并提炼可信原话...';
    const finalRender = (await dispatchToWorker('RERANK', {
      query,
      documents: docsForRender
    })) as SearchResultDoc[];

    const tRerankEnd = performance.now();
    const topConfidence = finalRender[0]?.confidenceScore ?? finalRender[0]?.rerankScore ?? -999;
    const shouldReject = finalRender.length > 0 && topConfidence < REJECTION_THRESHOLD;

    results.value = shouldReject ? [] : finalRender;

    emitTrace(query, finalRender, {
      totalMs: (tRerankEnd - tStart).toFixed(1),
      searchMs: (tSearchEnd - tStart).toFixed(1),
      fetchMs: (tFetchEnd - tSearchEnd).toFixed(1),
      rerankMs: (tRerankEnd - tFetchEnd).toFixed(1),
      topConfidence,
      rejected: shouldReject,
    });

    if (shouldReject) {
      statusMsg.value = `未达到展示阈值 ${(REJECTION_THRESHOLD * 100).toFixed(0)}%，已拒答`;
      logDiagnostic(`拒答：Top 1 原话匹配度 ${formatPercent(topConfidence)}`);
    } else {
      statusMsg.value = `找到 ${finalRender.length} 篇相关结果（总耗时 ${(tRerankEnd - tStart).toFixed(1)}ms）`;
      logDiagnostic('结果已完成展示');
    }
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
        v-if="results.length === 0 && !isProcessing"
        class="flex h-full flex-col items-center justify-center text-center text-slate-500"
      >
        <div class="mb-4 flex h-14 w-14 items-center justify-center rounded-full border border-dashed border-slate-700">
          <Search class="h-5 w-5" />
        </div>
        <p class="text-sm font-medium">{{ emptyTitle }}</p>
        <p class="mt-1 text-[11px] text-slate-600">{{ emptySubtitle }}</p>
      </div>

      <section
        v-if="results.length === 0 && !isProcessing && hasCoverageRejection && weakResults.length > 0"
        class="mt-6 space-y-3"
      >
        <div class="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-500">
          <span>Weak Results</span>
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
                    {{ isOriginalSnippet(res) ? '官方原话' : '相关要点' }}
                  </span>
                </div>
                <h3 class="text-[15px] font-semibold leading-6 text-slate-100">
                  {{ res.ot_title || '未命名政策文档' }}
                </h3>
              </div>

              <div class="shrink-0 text-right">
                <div v-if="isOriginalSnippet(res)" class="font-mono text-[11px] text-slate-300">
                  {{ formatPercent(res.snippetScore ?? 0) }}
                </div>
                <div v-else class="font-mono text-[11px] text-slate-400">
                  {{ formatRetrievalScore(getDisplayScore(res)) }}
                </div>
                <div class="mt-1 text-[10px] text-slate-500">
                  {{ isOriginalSnippet(res) ? '原话匹配度' : '检索分' }}
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
