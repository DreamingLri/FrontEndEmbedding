<script setup lang="ts">
import { computed, ref } from 'vue';
import {
  AlertCircle,
  Clock,
  Cpu,
  ExternalLink,
  Info,
  Loader2,
  Search
} from 'lucide-vue-next';
import type { PipelineDocumentRecord } from '../worker/search_pipeline.ts';
import { useSearchRuntime } from '../services/useSearchRuntime';
import {
  formatPercent,
  formatRetrievalScore,
  getDisplayScore,
  getPreviewText,
  highlightSnippet,
  isOriginalSnippet,
} from '../utils/searchPresentation';
import {
  getEmptyStateCopy,
  getRejectDescription,
  getRejectReasonLabel,
  getRejectTierLabel,
  ORIGINAL_SNIPPET_THRESHOLD,
  splitPrimaryResults,
  type SearchTraceData,
} from '../utils/searchUi';

const emit = defineEmits<{
  (event: 'trace-updated', payload: SearchTraceData): void
}>();

const showWeakResults = ref(false);
const {
  searchQuery,
  results,
  weakResults,
  rejectionInfo,
  decisionInfo,
  isProcessing,
  statusMsg,
  errorMsg,
  diagnosticLogs,
  isWorkerReady,
  handleSearch: executeSearch,
} = useSearchRuntime({
  onTraceUpdated: (traceData) => emit('trace-updated', traceData),
});

const trimmedQuery = computed(() => searchQuery.value.trim());
const resultBuckets = computed(() => splitPrimaryResults(results.value));
const heroResults = computed(() => resultBuckets.value.heroResults);
const compactResults = computed(() => resultBuckets.value.compactResults);
const activeRejectionReason = computed(
  () => decisionInfo.value?.rejectionReason ?? rejectionInfo.value?.reason ?? null
);
const hasCoverageRejection = computed(
  () => activeRejectionReason.value === 'low_topic_coverage'
);
const isRejectDecision = computed(
  () => decisionInfo.value?.behavior === 'reject'
);
const weakToggleLabel = computed(() =>
  showWeakResults.value ? '收起弱相关结果' : '查看弱相关结果'
);
const rejectReasonLabel = computed(() => {
  return getRejectReasonLabel(activeRejectionReason.value) || '当前不适合直接回答';
});
const rejectTierLabel = computed(() =>
  getRejectTierLabel(decisionInfo.value?.rejectTier)
);
const rejectDescription = computed(() =>
  getRejectDescription(activeRejectionReason.value)
);
const emptyState = computed(() => getEmptyStateCopy(trimmedQuery.value));

const getAccordionPreview = (result: PipelineDocumentRecord) =>
  result.bestPoint || getPreviewText(result);

const handleSearch = async () => {
  showWeakResults.value = false;
  await executeSearch();
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
          :disabled="!trimmedQuery || isProcessing || !isWorkerReady"
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
        <p class="text-sm font-medium">{{ emptyState.title }}</p>
        <p class="mt-1 text-[11px] text-slate-600">{{ emptyState.subtitle }}</p>
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
            <div class="text-[10px] uppercase tracking-[0.16em] text-slate-500">弱相关</div>
            <div class="mt-1 font-mono text-[13px] text-slate-100">
              {{ weakResults.length }}
            </div>
            <div
              v-if="weakResults.length > 0"
              class="mt-2 text-[10px] text-slate-400"
            >
              可展开查看
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
            <span>弱相关结果</span>
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
              v-if="getAccordionPreview(res)"
              class="mt-2 rounded-xl bg-white/[0.03] px-3 py-2.5 text-[13px] leading-6 text-slate-400"
            >
              <div class="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-600">弱相关片段</div>
              <p>{{ getAccordionPreview(res) }}</p>
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
              v-html="highlightSnippet(getPreviewText(res), trimmedQuery)"
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
              v-if="getAccordionPreview(res)"
              class="mt-2 rounded-xl bg-white/[0.03] px-3 py-2.5 text-[13px] leading-6 text-slate-400"
            >
              <div class="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-600">相关要点</div>
              <p>{{ getAccordionPreview(res) }}</p>
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
