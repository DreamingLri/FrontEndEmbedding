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
import type {
  PipelineBehavior,
  PipelineDecision,
  PipelineDocumentRecord,
  PipelineTrace,
} from '../worker/search_pipeline.ts';
import type { SearchRejection } from '../worker/vector_engine.ts';
import {
  formatPercent,
  formatRetrievalScore,
  getDisplayScore,
  getPreviewText,
  isOriginalSnippet,
} from '../utils/searchPresentation';

type TraceDecision = PipelineDecision;
type TraceRejection = SearchRejection;
type TraceResult = PipelineDocumentRecord;

type TraceData = {
  query: string;
  results: TraceResult[];
  retrievalDecision?: TraceDecision | null;
  decision?: TraceDecision | null;
  rejection?: TraceRejection | null;
  weakResultsCount?: number;
  directAnswerRescue?: PipelineTrace['directAnswerRescue'] | null;
  querySignals?: PipelineTrace['querySignals'] | null;
  retrievalSignals?: PipelineTrace['retrievalSignals'] | null;
  stats?: {
    totalMs: string;
    searchMs: string;
    fetchMs: string;
    rerankMs: string;
    rerankedDocCount?: number;
    chunksScored?: number;
    rerankWindowReason?: string;
    maxChunksPerDoc?: number;
    chunkPlanReason?: string;
    rejectionThreshold?: number;
    initialTopConfidence?: number | null;
    topConfidence?: number | null;
    rejected?: boolean;
  };
};

const props = defineProps<{
  traceData: TraceData | null;
}>();

const top3Results = () => props.traceData?.results?.slice(0, 3) ?? [];
const compactResults = () => props.traceData?.results?.slice(3, 10) ?? [];

const behaviorLabel = (behavior?: PipelineBehavior | null) => {
  switch (behavior) {
    case 'answer':
      return '回答';
    case 'reject':
      return '拒答';
    default:
      return '未决';
  }
};

const behaviorClass = (behavior?: PipelineBehavior | null) => {
  switch (behavior) {
    case 'answer':
      return 'border-emerald-500/25 bg-emerald-500/10 text-emerald-200';
    case 'reject':
      return 'border-rose-500/25 bg-rose-500/10 text-rose-200';
    default:
      return 'border-white/10 bg-white/[0.03] text-slate-300';
  }
};

const decisionFlowLabel = () => {
  const retrievalBehavior = props.traceData?.retrievalDecision?.behavior;
  const finalBehavior = props.traceData?.decision?.behavior;
  if (!retrievalBehavior || !finalBehavior || retrievalBehavior === finalBehavior) {
    return '';
  }
  return `${behaviorLabel(retrievalBehavior)} -> ${behaviorLabel(finalBehavior)}`;
};

const rejectionReasonLabel = () => {
  const reason =
    props.traceData?.decision?.rejectionReason ??
    props.traceData?.rejection?.reason ??
    null;

  switch (reason) {
    case 'display_threshold':
      return '展示阈值不足';
    case 'low_topic_coverage':
      return '主题覆盖不足';
    case 'low_consistency':
      return '主题一致性不足';
    default:
      return '';
  }
};

const decisionSubtitle = () => {
  const decision = props.traceData?.decision;
  if (!decision) {
    return '等待统一 pipeline 返回行为决策。';
  }

  if (decision.behavior === 'reject') {
    const rejectReason = rejectionReasonLabel();
    return rejectReason
      ? `系统选择拒答，主因是：${rejectReason}。`
      : '系统选择拒答，当前结果未达到稳定可展示条件。';
  }

  return '系统已进入直接回答链路，并对候选文档完成展示前重排。';
};

const rescueSubtitle = () => {
  const rescue = props.traceData?.directAnswerRescue;
  if (!rescue?.attempted) {
    return '';
  }
  if (rescue.succeeded) {
    return '直答候选在首次展示评分偏低时，系统扩大了重排范围并成功保留答案。';
  }
  if (rescue.accepted) {
    return '系统触发了补救重排，但补救后仍未达到最终保留条件。';
  }
  return rescue.reason || '当前直答候选未满足补救重排的触发条件。';
};
</script>

<template>
  <div class="flex h-full flex-col overflow-hidden rounded-[24px] border border-white/5 bg-white/[0.025] shadow-[0_20px_60px_rgba(0,0,0,0.18)] backdrop-blur-xl">
    <div class="flex items-center gap-2 border-b border-white/8 bg-white/[0.025] px-5 py-3">
      <Activity class="h-4 w-4 text-slate-400" />
      <h3 class="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-400">检索 Trace</h3>
    </div>

    <div v-if="!traceData" class="flex flex-1 flex-col items-center justify-center p-8 text-center opacity-35">
      <Layers class="mb-2 h-10 w-10 text-slate-700" />
      <p class="text-xs text-slate-500">这里会显示检索路径和关键状态</p>
    </div>

    <div v-else class="custom-scrollbar flex-1 space-y-3 overflow-y-auto p-4">
      <div class="rounded-xl border border-white/5 bg-white/[0.03] p-3">
        <div class="mb-1 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">查询</div>
        <p class="text-sm italic text-slate-200">"{{ traceData.query }}"</p>
      </div>

      <div v-if="traceData.stats" class="grid grid-cols-3 gap-2">
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
          <div class="mt-1 text-[9px] text-purple-200/80">
            {{ traceData.stats.rerankedDocCount ?? 0 }} 篇 / {{ traceData.stats.chunksScored ?? 0 }} chunks
          </div>
          <div class="mt-1 text-[9px] text-purple-200/70">
            每篇上限 {{ traceData.stats.maxChunksPerDoc ?? 0 }} chunks
          </div>
          <div v-if="traceData.stats.rerankWindowReason" class="mt-1 text-[8px] uppercase tracking-[0.12em] text-purple-200/60">
            {{ traceData.stats.rerankWindowReason }}
          </div>
        </div>
      </div>

      <div
        v-if="traceData.decision"
        class="rounded-xl border border-white/5 bg-white/[0.03] p-3"
      >
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="mb-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">行为决策</div>
            <div class="flex flex-wrap items-center gap-2">
              <span class="rounded-full border px-2.5 py-1 text-[10px] font-semibold" :class="behaviorClass(traceData.decision.behavior)">
                {{ behaviorLabel(traceData.decision.behavior) }}
              </span>
              <span
                v-if="traceData.retrievalDecision"
                class="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-[10px] text-slate-300"
              >
                检索阶段 {{ behaviorLabel(traceData.retrievalDecision.behavior) }}
              </span>
            </div>
          </div>
          <div class="shrink-0 text-right text-[10px] text-slate-400">
            <div>置信 {{ formatPercent(traceData.decision.confidence || 0) }}</div>
            <div>弱相关 {{ traceData.weakResultsCount ?? 0 }} 条</div>
          </div>
        </div>

        <div class="mt-2 text-[11px] leading-6 text-slate-400">
          {{ decisionSubtitle() }}
        </div>
        <div v-if="decisionFlowLabel()" class="mt-2 text-[10px] text-slate-500">
          展示阶段调整: {{ decisionFlowLabel() }}
        </div>

        <div class="mt-3 flex flex-wrap gap-2 text-[10px]">
          <span
            v-if="traceData.decision.preferLatestWithinTopic"
            class="rounded-full border border-sky-400/20 bg-sky-400/10 px-2 py-0.5 text-sky-200"
          >
            latest 优先
          </span>
          <span
            v-if="traceData.decision.useWeakMatches"
            class="rounded-full border border-amber-400/20 bg-amber-400/10 px-2 py-0.5 text-amber-200"
          >
            使用弱相关候选
          </span>
          <span
            v-if="rejectionReasonLabel()"
            class="rounded-full border border-rose-400/20 bg-rose-400/10 px-2 py-0.5 text-rose-200"
          >
            {{ rejectionReasonLabel() }}
          </span>
        </div>
      </div>

      <div
        v-if="traceData.stats && traceData.stats.topConfidence !== null && traceData.stats.topConfidence !== undefined"
        class="rounded-xl border px-3 py-2 text-xs"
        :class="traceData.stats.rejected ? 'border-amber-500/30 bg-amber-500/10 text-amber-200' : 'border-emerald-500/20 bg-emerald-500/10 text-emerald-200'"
      >
        <div class="flex items-center gap-2">
          <ShieldAlert v-if="traceData.stats.rejected" class="h-4 w-4" />
          <FileText v-else class="h-4 w-4" />
          <span>{{ traceData.stats.rejected ? '本次查询未通过展示阈值' : '本次查询通过展示阈值' }}</span>
        </div>
        <div class="mt-1 text-[11px] opacity-80">
          Top 1 原话匹配度: {{ formatPercent(traceData.stats.topConfidence || 0) }}
          <span v-if="traceData.stats.rejectionThreshold !== undefined">
            / 阈值: {{ formatPercent(traceData.stats.rejectionThreshold) }}
          </span>
        </div>
        <div
          v-if="traceData.stats.initialTopConfidence !== null && traceData.stats.initialTopConfidence !== undefined"
          class="mt-1 text-[11px] opacity-70"
        >
          初始重排置信度: {{ formatPercent(traceData.stats.initialTopConfidence || 0) }}
        </div>
      </div>

      <div
        v-if="traceData.directAnswerRescue?.attempted"
        class="rounded-xl border px-3 py-2 text-xs"
        :class="traceData.directAnswerRescue.succeeded ? 'border-sky-500/25 bg-sky-500/10 text-sky-200' : 'border-white/10 bg-white/[0.03] text-slate-300'"
      >
        <div class="flex items-center justify-between gap-3">
          <div class="flex items-center gap-2">
            <Cpu class="h-4 w-4" />
            <span>直答补救重排</span>
          </div>
          <span class="rounded-full border border-white/10 px-2 py-0.5 text-[10px]">
            {{ traceData.directAnswerRescue.succeeded ? '已保留' : '未保留' }}
          </span>
        </div>
        <div class="mt-1 text-[11px] opacity-80">
          {{ rescueSubtitle() }}
        </div>
        <div class="mt-2 flex flex-wrap gap-2 text-[10px] opacity-80">
          <span>
            初始 {{ formatPercent(traceData.directAnswerRescue.initialTopConfidence || 0) }}
          </span>
          <span v-if="traceData.directAnswerRescue.rescueTopConfidence !== undefined">
            补救后 {{ formatPercent(traceData.directAnswerRescue.rescueTopConfidence || 0) }}
          </span>
          <span v-if="traceData.directAnswerRescue.initialRerankDocCount !== undefined">
            文档窗口 {{ traceData.directAnswerRescue.initialRerankDocCount }} -> {{ traceData.directAnswerRescue.rescueRerankDocCount ?? traceData.directAnswerRescue.initialRerankDocCount }}
          </span>
          <span v-if="traceData.directAnswerRescue.initialMaxChunksPerDoc !== undefined">
            chunks/文档 {{ traceData.directAnswerRescue.initialMaxChunksPerDoc }} -> {{ traceData.directAnswerRescue.rescueMaxChunksPerDoc ?? traceData.directAnswerRescue.initialMaxChunksPerDoc }}
          </span>
        </div>
      </div>

      <div
        v-if="traceData.querySignals || traceData.retrievalSignals"
        class="rounded-xl border border-white/5 bg-white/[0.03] p-3"
      >
        <div class="mb-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">诊断信号</div>
        <div class="flex flex-wrap gap-2 text-[10px]">
          <span
            v-if="traceData.querySignals?.hasExplicitTopicOrIntent"
            class="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 text-emerald-200"
          >
            显式主题/意图
          </span>
          <span
            v-if="traceData.querySignals?.hasStrongDetailAnchor"
            class="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 text-emerald-200"
          >
            强细节锚点
          </span>
          <span
            v-if="traceData.querySignals?.hasExplicitYear"
            class="rounded-full border border-sky-400/20 bg-sky-400/10 px-2 py-0.5 text-sky-200"
          >
            显式年份
          </span>
          <span
            v-if="traceData.querySignals?.hasLatestPolicyState"
            class="rounded-full border border-sky-400/20 bg-sky-400/10 px-2 py-0.5 text-sky-200"
          >
            latest 诉求
          </span>
          <span
            v-if="traceData.querySignals?.hasEntryLikeAnchor"
            class="rounded-full border border-amber-400/20 bg-amber-400/10 px-2 py-0.5 text-amber-200"
          >
            入口型问法
          </span>
        </div>
        <div class="mt-3 grid grid-cols-2 gap-2 text-[10px] text-slate-400">
          <div v-if="traceData.querySignals">
            查询长度 {{ traceData.querySignals.queryLength }}
            <span v-if="traceData.querySignals.tokenCount"> / 分词 {{ traceData.querySignals.tokenCount }}</span>
          </div>
          <div v-if="traceData.retrievalSignals">
            Top1-Top2 间隔 {{ formatPercent(traceData.retrievalSignals.top1Top2Gap || 0) }}
          </div>
          <div v-if="traceData.retrievalSignals">
            主导主题占比 {{ formatPercent(traceData.retrievalSignals.dominantTopicRatio || 0) }}
          </div>
          <div v-if="traceData.retrievalSignals">
            主题数 {{ traceData.retrievalSignals.distinctTopicCount }} / 候选 {{ traceData.retrievalSignals.candidateCount }}
          </div>
        </div>
      </div>

      <section v-if="top3Results().length > 0" class="space-y-2">
        <div class="flex items-center justify-between text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
          <span>Top 1-3</span>
          <span>摘要</span>
        </div>

        <div
          v-for="(res, i) in top3Results()"
          :key="res.otid || res.id || i"
          class="rounded-xl border border-white/5 bg-white/[0.03] p-3"
        >
          <div class="mb-2 flex items-start justify-between gap-3">
            <div class="min-w-0">
              <div class="mb-1 flex items-center gap-2">
                <span class="text-[10px] font-semibold text-slate-500">#{{ i + 1 }}</span>
                <span class="rounded-full border border-white/8 px-2 py-0.5 text-[10px] text-slate-400">
                  {{ isOriginalSnippet(res, traceData.stats?.rejectionThreshold ?? 0.4) ? '官方原话' : '相关要点' }}
                </span>
              </div>
              <div class="text-xs font-semibold text-slate-200">{{ res.ot_title || '未命名政策文档' }}</div>
            </div>
            <div class="text-right">
              <div v-if="isOriginalSnippet(res, traceData.stats?.rejectionThreshold ?? 0.4)" class="text-[10px] font-mono text-slate-300">
                {{ formatPercent(res.snippetScore ?? 0) }}
              </div>
              <div v-else class="text-[10px] font-mono text-slate-400">
                {{ formatRetrievalScore(getDisplayScore(res)) }}
              </div>
            </div>
          </div>

          <div v-if="getPreviewText(res)" class="mb-2 text-[11px] leading-6 text-slate-400">
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

      <section v-if="compactResults().length > 0" class="space-y-1.5">
        <div class="flex items-center justify-between text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
          <span>Top 4-10</span>
          <span>列表</span>
        </div>

        <div
          v-for="(res, i) in compactResults()"
          :key="res.otid || res.id || `compact-${i}`"
          class="rounded-xl border border-white/5 bg-white/[0.02] p-3"
        >
          <div class="flex items-start justify-between gap-3">
            <div class="min-w-0">
              <div class="mb-1 text-[10px] font-semibold text-slate-500">#{{ i + 4 }}</div>
              <div class="text-xs text-slate-200">{{ res.ot_title || '未命名政策文档' }}</div>
            </div>
            <div class="text-[10px] font-mono text-slate-500">{{ formatRetrievalScore(getDisplayScore(res)) }}</div>
          </div>

          <div v-if="res.bestPoint" class="mt-2 line-clamp-2 text-[11px] leading-6 text-slate-500">
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
