<script setup lang="ts">
import {
  Activity,
  Clock,
  Cpu,
  ExternalLink,
  FileText,
  Layers,
  ShieldAlert
} from 'lucide-vue-next';

const props = defineProps<{
  traceData: {
    query: string;
    results: any[];
    stats?: {
      totalMs: string;
      searchMs: string;
      fetchMs: string;
      rerankMs: string;
      rejectionThreshold?: number;
      topConfidence?: number;
      rejected?: boolean;
    };
  } | null;
}>();

const formatPercent = (score: number) => ((score || 0) * 100).toFixed(1) + '%';
const formatRetrievalScore = (score: number) => `Score ${Number(score || 0).toFixed(2)}`;
const getDisplayScore = (res: any) => res.displayScore ?? res.coarseScore ?? res.confidenceScore ?? res.rerankScore ?? res.score ?? 0;
const getPreviewText = (res: any) => res.bestSentence || res.bestPoint || '';
const isOriginalSnippet = (res: any) => (res.snippetScore ?? -999) > 0.4 && !!res.bestSentence;
const top3Results = () => props.traceData?.results?.slice(0, 3) ?? [];
const compactResults = () => props.traceData?.results?.slice(3, 10) ?? [];
</script>

<template>
  <div class="flex h-full flex-col overflow-hidden rounded-2xl border border-white/5 bg-slate-900/10 shadow-xl backdrop-blur-md">
    <div class="flex items-center gap-2 border-b border-white/10 bg-white/5 px-6 py-4">
      <Activity class="h-4 w-4 text-slate-300" />
      <h3 class="text-xs font-bold uppercase tracking-widest text-slate-300">检索 Trace</h3>
    </div>

    <div v-if="!traceData" class="flex flex-1 flex-col items-center justify-center p-8 text-center opacity-40">
      <Layers class="mb-2 h-12 w-12 text-slate-600" />
      <p class="text-xs text-slate-500">执行检索后，这里会显示召回、原话提炼和拒识信息</p>
    </div>

    <div v-else class="custom-scrollbar flex-1 overflow-y-auto p-5 space-y-4">
      <div class="rounded-xl border border-white/6 bg-white/[0.04] p-3">
        <div class="mb-1 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">查询</div>
        <p class="text-sm italic text-slate-200">"{{ traceData.query }}"</p>
      </div>

      <div v-if="traceData.stats" class="grid grid-cols-4 gap-2">
        <div class="rounded-lg border border-white/5 bg-black/20 p-2 text-center">
          <div class="text-[8px] font-bold uppercase text-slate-500">Total</div>
          <div class="text-xs font-mono font-bold text-white">{{ traceData.stats.totalMs }}ms</div>
        </div>
        <div class="rounded-lg border border-blue-500/20 bg-blue-500/10 p-2 text-center">
          <div class="text-[8px] font-bold uppercase text-blue-400">Search</div>
          <div class="text-xs font-mono font-bold text-blue-300">{{ traceData.stats.searchMs }}ms</div>
        </div>
        <div class="rounded-lg border border-emerald-500/20 bg-emerald-500/10 p-2 text-center">
          <div class="text-[8px] font-bold uppercase text-emerald-400">Fetch</div>
          <div class="text-xs font-mono font-bold text-emerald-300">{{ traceData.stats.fetchMs }}ms</div>
        </div>
        <div class="rounded-lg border border-purple-500/20 bg-purple-500/10 p-2 text-center">
          <div class="flex items-center justify-center gap-1 text-[8px] font-bold uppercase text-purple-400">
            <Cpu class="h-2.5 w-2.5" />
            Rerank
          </div>
          <div class="text-xs font-mono font-bold text-purple-300">{{ traceData.stats.rerankMs }}ms</div>
        </div>
      </div>

      <div
        v-if="traceData.stats && typeof traceData.stats.rejected === 'boolean'"
        class="rounded-xl border px-3 py-2 text-xs"
        :class="traceData.stats.rejected ? 'border-amber-500/30 bg-amber-500/10 text-amber-200' : 'border-emerald-500/20 bg-emerald-500/10 text-emerald-200'"
      >
        <div class="flex items-center gap-2">
          <ShieldAlert v-if="traceData.stats.rejected" class="h-4 w-4" />
          <FileText v-else class="h-4 w-4" />
          <span>{{ traceData.stats.rejected ? '本次查询被拒识' : '本次查询通过展示阈值' }}</span>
        </div>
        <div class="mt-1 text-[11px] opacity-80">
          Top 1 原话匹配度: {{ formatPercent(traceData.stats.topConfidence || 0) }}
          <span v-if="traceData.stats.rejectionThreshold !== undefined">
            / 阈值: {{ formatPercent(traceData.stats.rejectionThreshold) }}
          </span>
        </div>
      </div>

      <section v-if="top3Results().length > 0" class="space-y-2">
        <div class="flex items-center justify-between text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
          <span>Top 1-3</span>
          <span>原话提炼</span>
        </div>

        <div
          v-for="(res, i) in top3Results()"
          :key="res.otid || res.id || i"
          class="rounded-xl border border-white/6 bg-white/[0.04] p-3"
        >
          <div class="mb-2 flex items-start justify-between gap-3">
            <div class="min-w-0">
              <div class="mb-1 flex items-center gap-2">
                <span class="text-[10px] font-semibold text-slate-500">#{{ i + 1 }}</span>
                <span class="rounded-full border border-white/8 px-2 py-0.5 text-[10px] text-slate-400">
                  {{ isOriginalSnippet(res) ? '官方原话' : '相关要点' }}
                </span>
              </div>
              <div class="text-xs font-semibold text-slate-200">{{ res.ot_title || '未命名政策文档' }}</div>
            </div>
            <div class="text-right">
              <div v-if="isOriginalSnippet(res)" class="text-[10px] font-mono text-slate-300">
                {{ formatPercent(res.snippetScore ?? 0) }}
              </div>
              <div v-else class="text-[10px] font-mono text-slate-400">
                {{ formatRetrievalScore(getDisplayScore(res)) }}
              </div>
            </div>
          </div>

          <div v-if="getPreviewText(res)" class="mb-2 text-[11px] leading-6 text-slate-300">
            {{ getPreviewText(res) }}
          </div>

          <div class="flex items-center gap-3 text-[10px] text-slate-500">
            <div v-if="res.publish_time" class="flex items-center gap-1">
              <Clock class="h-3 w-3" />
              {{ res.publish_time }}
            </div>
            <a v-if="res.link" :href="res.link" target="_blank" class="inline-flex items-center gap-1 text-slate-400 hover:text-slate-200">
              原文
              <ExternalLink class="h-3 w-3" />
            </a>
          </div>
        </div>
      </section>

      <section v-if="compactResults().length > 0" class="space-y-2">
        <div class="flex items-center justify-between text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
          <span>Top 4-10</span>
          <span>知识点</span>
        </div>

        <div
          v-for="(res, i) in compactResults()"
          :key="res.otid || res.id || `compact-${i}`"
          class="rounded-xl border border-white/6 bg-white/[0.03] p-3"
        >
          <div class="flex items-start justify-between gap-3">
            <div class="min-w-0">
              <div class="mb-1 text-[10px] font-semibold text-slate-500">#{{ i + 4 }}</div>
              <div class="text-xs text-slate-200">{{ res.ot_title || '未命名政策文档' }}</div>
            </div>
            <div class="text-[10px] font-mono text-slate-500">{{ formatRetrievalScore(getDisplayScore(res)) }}</div>
          </div>

          <div v-if="res.bestPoint" class="mt-2 text-[11px] leading-6 text-slate-400">
            {{ res.bestPoint }}
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
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
