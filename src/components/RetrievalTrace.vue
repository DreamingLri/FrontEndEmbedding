<script setup lang="ts">
import { 
  Activity, 
  Layers, 
  MapPin, 
  Clock,
  ExternalLink,
  Zap,
  Cpu
} from 'lucide-vue-next';

defineProps<{
  traceData: {
    query: string;
    results: any[];
    stats?: {
      totalMs: string;
      searchMs: string;
      fetchMs: string;
      rerankMs: string;
    };
  } | null;
}>();

const formatScore = (score: number) => (score * 100).toFixed(1) + '%';
</script>

<template>
  <div class="h-full flex flex-col bg-slate-900/10 backdrop-blur-md rounded-2xl border border-white/5 overflow-hidden shadow-xl">
    <div class="px-6 py-4 border-b border-white/10 bg-white/5 flex items-center gap-2">
        <Activity class="w-4 h-4 text-purple-400" />
        <h3 class="text-xs font-bold uppercase tracking-widest text-slate-300">检索分析 Trace</h3>
    </div>

    <div v-if="!traceData" class="flex-1 flex flex-col items-center justify-center p-8 text-center opacity-40">
        <Layers class="w-12 h-12 text-slate-600 mb-2" />
        <p class="text-xs text-slate-500">发送消息后，这里将实时分析向量匹配链路</p>
    </div>

    <div v-else class="flex-1 overflow-y-auto p-5 space-y-4 custom-scrollbar">
        <div class="p-3 rounded-xl bg-purple-500/10 border border-purple-500/20">
            <h4 class="text-[10px] font-bold text-purple-300 uppercase mb-1">查询语义</h4>
            <p class="text-xs text-slate-300 italic">"{{ traceData.query }}"</p>
        </div>

        <div class="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-6 mb-2 flex items-center gap-2">
            <span>引擎运算时延分析 (Telemetry)</span>
            <div class="h-[1px] flex-1 bg-white/5"></div>
        </div>

        <div v-if="traceData.stats" class="grid grid-cols-4 gap-2 mb-6">
            <div class="bg-black/20 border border-white/5 rounded-lg p-2 flex flex-col items-center justify-center text-center">
                <span class="text-[8px] text-slate-500 font-bold uppercase mb-1 drop-shadow-sm">Total</span>
                <span class="text-xs font-mono font-bold text-white">{{ traceData.stats.totalMs }}ms</span>
            </div>
            <div class="bg-blue-500/10 border border-blue-500/20 rounded-lg p-2 flex flex-col items-center justify-center text-center">
                <span class="text-[8px] text-blue-400 font-bold uppercase mb-1 drop-shadow-sm">WASM 粗排</span>
                <span class="text-xs font-mono font-bold text-blue-300">{{ traceData.stats.searchMs }}ms</span>
            </div>
            <div class="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-2 flex flex-col items-center justify-center text-center">
                <span class="text-[8px] text-emerald-400 font-bold uppercase mb-1 drop-shadow-sm">SQLite 拉取</span>
                <span class="text-xs font-mono font-bold text-emerald-300">{{ traceData.stats.fetchMs }}ms</span>
            </div>
            <div class="bg-purple-500/10 border border-purple-500/20 rounded-lg p-2 flex flex-col items-center justify-center text-center">
                <span class="text-[8px] text-purple-400 font-bold uppercase mb-1 flex items-center gap-1 drop-shadow-sm"><Cpu class="w-2.5 h-2.5"/> GPU 精排</span>
                <span class="text-xs font-mono font-bold text-purple-300">{{ traceData.stats.rerankMs }}ms</span>
            </div>
        </div>

        <div class="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-6 mb-2 flex items-center gap-2">
            <span>命中 Top {{ traceData.results.length }} 份政策原文</span>
            <div class="h-[1px] flex-1 bg-white/5"></div>
        </div>

        <div v-for="(res, i) in traceData.results" :key="res.otid || res.id || i" 
             class="group p-4 rounded-xl bg-white/5 border border-white/5 hover:border-blue-500/30 transition-all">
            <div class="flex items-start justify-between mb-3">
                <div class="flex items-center gap-2">
                    <span class="text-[10px] font-bold w-5 h-5 rounded-full bg-slate-800 text-slate-500 flex items-center justify-center">
                        {{ i + 1 }}
                    </span>
                    <span class="text-[10px] font-mono text-blue-400">ID: {{ String(res.otid || res.pkid || res.id || 'unknown').slice(0, 8) }}...</span>
                </div>
                <div class="flex items-center gap-1 bg-blue-500/20 px-2 py-0.5 rounded text-blue-400">
                    <Zap class="w-3 h-3" />
                    <span class="text-[10px] font-bold">{{ formatScore(res.rerankScore || res.score || 0) }}</span>
                </div>
            </div>

            <h5 class="text-xs font-bold text-slate-200 mb-2 group-hover:text-blue-400 transition-colors">
                {{ res.ot_title || '未命名政策文档' }}
            </h5>

            <div class="flex flex-wrap gap-2 mb-3">
                <div v-if="res.publish_time" class="flex items-center gap-1 text-[9px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
                    <Clock class="w-2.5 h-2.5" />
                    {{ res.publish_time }}
                </div>
                <div v-if="res.source" class="flex items-center gap-1 text-[9px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
                    <MapPin class="w-2.5 h-2.5" />
                    {{ res.source }}
                </div>
            </div>

            <div class="text-[10px] text-slate-400 line-clamp-2 leading-relaxed mb-3">
                {{ res.ot_text || res.content_json || res.text }}
            </div>

            <a v-if="res.metadata?.url" :href="res.metadata.url" target="_blank" 
               class="flex items-center gap-1 text-[10px] font-bold text-blue-500 hover:text-blue-400">
                查看政策原文
                <ExternalLink class="w-2.5 h-2.5" />
            </a>
        </div>
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
