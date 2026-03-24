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

const selectedModelId = ref<string>('dmeta_small');
const models = [
  { id: 'dmeta_small', name: 'DMeta Soul', path: 'DMetaSoul/Dmeta-embedding-zh-small' }
];

const emit = defineEmits(['trace-updated']);

const searchQuery = ref('');
const results = ref<any[]>([]);
const isProcessing = ref(false);
const statusMsg = ref('正在唤起 Web Worker...');
const errorMsg = ref<string | null>(null);
const diagnosticLogs = ref<string[]>([]);
const isWorkerReady = ref(false);

const REJECTION_THRESHOLD = 0.4;
const heroResults = computed(() => results.value.slice(0, 3));
const compactResults = computed(() => results.value.slice(3, 10));

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

const getPreviewText = (res: any) => {
  if (res.bestSentence) return res.bestSentence;
  if (res.bestPoint) return res.bestPoint;

  if (res.best_kpid && Array.isArray(res.kps)) {
    const hitKp = res.kps.find((kp: any) => kp.kpid === res.best_kpid);
    if (hitKp?.kp_text) return hitKp.kp_text;
  }

  return '';
};

const isOriginalSnippet = (res: any) => (res.snippetScore ?? -999) > REJECTION_THRESHOLD;
const getDisplayScore = (res: any) => res.displayScore ?? res.coarseScore ?? res.confidenceScore ?? res.rerankScore ?? 0;
const formatRetrievalScore = (score: number) => `Score ${Number(score || 0).toFixed(2)}`;
const formatPercent = (score: number) => `${((score || 0) * 100).toFixed(1)}%`;

const mergeCoarseMatchesIntoDocuments = (documents: any[], coarseMatches: any[]) => {
  const coarseMap = new Map(
    coarseMatches.map((match: any) => [match.otid, match])
  );

  return documents.map((doc: any) => {
    const otid = doc.otid || doc.id;
    const coarse = coarseMap.get(otid);
    if (!coarse) return doc;

    return {
      ...doc,
      score: coarse.score ?? doc.score,
      coarseScore: coarse.score ?? doc.coarseScore ?? doc.score,
      displayScore: coarse.score ?? doc.displayScore ?? doc.score,
      best_kpid: coarse.best_kpid ?? doc.best_kpid
    };
  });
};

const logDiagnostic = (msg: string) => {
  const time = new Date().toLocaleTimeString();
  diagnosticLogs.value.unshift(`[${time}] ${msg}`);
  if (diagnosticLogs.value.length > 5) diagnosticLogs.value.pop();
};

const pendingTasks = new Map<string, { resolve: Function; reject: Function; type: string }>();

const dispatchToWorker = (type: string, payload: any, transfer: Transferable[] = []): Promise<any> =>
  new Promise((resolve, reject) => {
    const taskId = crypto.randomUUID();
    pendingTasks.set(taskId, { resolve, reject, type });
    myWorker.postMessage({ type, payload, taskId }, transfer);
  });

const myWorker = new VectorWorker();

myWorker.onmessage = (event: MessageEvent) => {
  const { status, message, result, error, stats, taskId } = event.data;

  if (status === 'loading' || status === 'progress') {
    statusMsg.value = message;
    logDiagnostic(`[引擎通知] ${message}`);
  } else if (status === 'info') {
    logDiagnostic(`[系统提示] ${message}`);
  } else if (status === 'ready') {
    isWorkerReady.value = true;
    statusMsg.value = 'AI 检索引擎已就绪';
    logDiagnostic(message);
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.resolve(true);
      pendingTasks.delete(taskId);
    }
  } else if (status === 'error') {
    isProcessing.value = false;
    errorMsg.value = `引擎启动失败: ${error}`;
    logDiagnostic(`致命错误: ${error}`);
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.reject(new Error(error));
      pendingTasks.delete(taskId);
    }
  } else if (status === 'search_complete') {
    logDiagnostic(`粗排结束，扫描 ${stats.itemsScanned} 条，耗时 ${stats.elapsedMs}ms`);
    if (taskId && pendingTasks.has(taskId)) {
      pendingTasks.get(taskId)?.resolve(result);
      pendingTasks.delete(taskId);
    }
  } else if (status === 'rerank_complete') {
    logDiagnostic(`原话提炼完成，耗时 ${stats.elapsedMs}ms`);
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
    const CACHE_KEY_MATRIX = 'rag_vector_matrix_dmeta_v2';
    const CACHE_KEY_METADATA = 'rag_metadata_dmeta_v2';

    let matrixBuffer = await localforage.getItem<ArrayBuffer>(CACHE_KEY_MATRIX);
    let metadataJson = await localforage.getItem<any>(CACHE_KEY_METADATA);

    if (!matrixBuffer || !metadataJson) {
      logDiagnostic('本地缓存未命中，开始同步数据...');
      const [matrixRes, metaRes] = await Promise.all([
        fetch('/data/frontend_vectors_dmeta_small.bin'),
        fetch('/data/frontend_metadata_dmeta_small.json')
      ]);

      if (!matrixRes.ok || !metaRes.ok) throw new Error('网络请求核心数据失败');

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
  if (myWorker) myWorker.terminate();
});

const handleSearch = async () => {
  if (!searchQuery.value.trim() || isProcessing.value || !isWorkerReady.value) return;

  errorMsg.value = null;
  isProcessing.value = true;
  results.value = [];
  statusMsg.value = '正在分词与向量化...';

  try {
    const tStart = performance.now();
    logDiagnostic('开始提交检索请求...');
    const localMatches = await dispatchToWorker('SEARCH', searchQuery.value.trim());
    const tSearchEnd = performance.now();

    const topIds = localMatches.slice(0, 15).map((r: any) => r.otid);
    logDiagnostic(`粗排返回 ${localMatches.length} 篇，取 Top ${topIds.length} 进入原话提炼`);

    if (topIds.length === 0) {
      statusMsg.value = '未找到匹配答案';
      logDiagnostic('本地检索未命中结果');
      return;
    }

    statusMsg.value = '正在请求原文数据...';
    const response = await fetch('/api/get_answers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ otids: topIds })
    });

    if (!response.ok) throw new Error(`后端响应异常: ${response.status}`);

    const result = await response.json();
    const tFetchEnd = performance.now();

    if (result.data) {
      const docsForRender = mergeCoarseMatchesIntoDocuments(result.data, localMatches.slice(0, 15));
      statusMsg.value = '正在提炼可信原话...';
      const finalRender = await dispatchToWorker('RERANK', {
        query: searchQuery.value.trim(),
        documents: docsForRender
      });

      const tRerankEnd = performance.now();
      const topConfidence = finalRender[0]?.confidenceScore ?? finalRender[0]?.rerankScore ?? -999;
      const shouldReject = finalRender.length > 0 && topConfidence < REJECTION_THRESHOLD;
      results.value = shouldReject ? [] : finalRender;

      const stats = {
        totalMs: (tRerankEnd - tStart).toFixed(1),
        searchMs: (tSearchEnd - tStart).toFixed(1),
        fetchMs: (tFetchEnd - tSearchEnd).toFixed(1),
        rerankMs: (tRerankEnd - tFetchEnd).toFixed(1),
        rejectionThreshold: REJECTION_THRESHOLD,
        topConfidence,
        rejected: shouldReject
      };

      emit('trace-updated', {
        query: searchQuery.value,
        results: finalRender,
        stats
      });

      if (shouldReject) {
        statusMsg.value = `未达到展示阈值 ${(REJECTION_THRESHOLD * 100).toFixed(0)}%，已拒识`;
        logDiagnostic(`拒识：Top 1 原话匹配度=${formatPercent(topConfidence)}`);
      } else {
        statusMsg.value = `找到 ${finalRender.length} 篇相关结果（总耗时 ${stats.totalMs}ms）`;
        logDiagnostic('结果已完成展示');
      }
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
        <div class="mr-1 flex items-center gap-1 rounded-lg border border-white/6 bg-black/20 p-1">
          <button
            v-for="m in models"
            :key="m.id"
            @click="selectedModelId = m.id as any"
            class="rounded px-2 py-1 text-[9px] font-bold uppercase transition-all"
            :class="selectedModelId === m.id ? 'bg-blue-600 text-white' : 'text-slate-500 hover:text-slate-300'"
          >
            {{ m.name }}
          </button>
        </div>
        <select disabled class="cursor-not-allowed rounded-lg border border-white/6 bg-black/20 px-2 py-1 text-[9px] font-bold uppercase text-slate-500 outline-none">
          <option value="wasm">WASM Worker</option>
        </select>
      </div>

      <div class="flex items-center gap-2 pl-4 text-right">
        <div class="h-1.5 w-1.5 rounded-full" :class="isProcessing ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'" />
        <span class="line-clamp-1 text-[10px] font-semibold tracking-[0.16em] text-slate-500">{{ statusMsg }}</span>
      </div>
    </div>

    <div v-if="errorMsg" class="flex items-center gap-2 border-b border-red-500/20 bg-red-500/10 px-5 py-3 text-xs text-red-400">
      <AlertCircle class="h-4 w-4" />
      {{ errorMsg }}
    </div>

    <div class="border-b border-white/6 bg-white/[0.03] p-5">
      <div class="relative group">
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
          class="absolute right-2 top-1/2 -translate-y-1/2 rounded-xl bg-blue-600 px-4 py-2 text-xs font-bold transition-all active:scale-95 hover:bg-blue-500 disabled:opacity-50"
        >
          <span v-if="!isProcessing">搜索</span>
          <Loader2 v-else class="h-4 w-4 animate-spin" />
        </button>
      </div>

      <div class="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
        <div v-for="(log, i) in diagnosticLogs" :key="i" class="flex items-center gap-1 font-mono text-[9px] text-slate-500">
          <Info v-if="i === 0" class="h-2.5 w-2.5 text-blue-400" />
          {{ log }}
        </div>
      </div>
    </div>

    <div class="custom-scrollbar flex-1 overflow-y-auto px-5 py-4">
      <div v-if="results.length === 0 && !isProcessing" class="flex h-full flex-col items-center justify-center text-center text-slate-500">
        <div class="mb-4 flex h-14 w-14 items-center justify-center rounded-full border border-dashed border-slate-700">
          <Search class="h-5 w-5" />
        </div>
        <p class="text-sm font-medium">{{ searchQuery.trim() ? '当前问题未达到可信展示阈值' : '输入关键词开始检索' }}</p>
        <p class="mt-1 text-[11px] text-slate-600">
          {{ searchQuery.trim() ? `Top 1 原话匹配度需不低于 ${(REJECTION_THRESHOLD * 100).toFixed(0)}%` : 'Top 3 展示原话，Top 4-10 展示标题' }}
        </p>
      </div>

      <div v-else class="space-y-5">
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
                <div
                  v-if="isOriginalSnippet(res)"
                  class="font-mono text-[11px] text-slate-300"
                >
                  {{ formatPercent(res.snippetScore ?? 0) }}
                </div>
                <div
                  v-else
                  class="font-mono text-[11px] text-slate-400"
                >
                  {{ formatRetrievalScore(getDisplayScore(res)) }}
                </div>
                <div class="mt-1 text-[10px] text-slate-500">
                  {{ isOriginalSnippet(res) ? '原话匹配度' : '检索分' }}
                </div>
              </div>
            </div>

            <div v-if="getPreviewText(res)" class="mt-3 text-sm leading-7 text-slate-200" v-html="highlightSnippet(getPreviewText(res), searchQuery)" />

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

            <div v-if="res.bestPoint || getPreviewText(res)" class="mt-2 rounded-xl bg-white/[0.03] px-3 py-2.5 text-[13px] leading-6 text-slate-400">
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
        <span class="flex items-center gap-1" title="CPU 向量粗排"><Cpu class="h-2.5 w-2.5 text-slate-500" /> Vector Worker</span>
        <span class="flex items-center gap-1" title="Top 3 原文高亮提炼">
          <svg class="h-3 w-3 text-red-500/70" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" /></svg>
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
