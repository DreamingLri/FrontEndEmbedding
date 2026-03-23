<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import { 
  Cpu, 
  Loader2, 
  Search,
  AlertCircle,
  Clock,
  ExternalLink,
  Zap,
  Info,
  Layers
} from 'lucide-vue-next';

// --- Vite 专属：内联导入 Worker ---
import localforage from 'localforage';
import VectorWorker from '../worker/embedding.worker.ts?worker';

// --- Config / UI State ---
const selectedModelId = ref<string>('dmeta_small');
const models = [
  { id: 'dmeta_small', name: 'DMeta Soul', path: 'DMetaSoul/Dmeta-embedding-zh-small' }
];

const emit = defineEmits(['trace-updated']);

// --- State ---
const searchQuery = ref('');
const results = ref<any[]>([]);
const isProcessing = ref(false);
const statusMsg = ref('唤起 Web Worker...');
const errorMsg = ref<string | null>(null);
const diagnosticLogs = ref<string[]>([]);
const isWorkerReady = ref(false);

const logDiagnostic = (msg: string) => {
    const time = new Date().toLocaleTimeString();
    diagnosticLogs.value.unshift(`[${time}] ${msg}`);
    if (diagnosticLogs.value.length > 5) diagnosticLogs.value.pop();
};

// --- Worker Manager ---
const pendingTasks = new Map<string, { resolve: Function, reject: Function, type: string }>();

const dispatchToWorker = (type: string, payload: any, transfer: Transferable[] = []): Promise<any> => {
    return new Promise((resolve, reject) => {
        const taskId = crypto.randomUUID();
        pendingTasks.set(taskId, { resolve, reject, type });
        myWorker.postMessage({ type, payload, taskId }, transfer);
    });
};

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
        statusMsg.value = 'AI 聚合引擎就绪';
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
        logDiagnostic(`粗排结束，扫描 ${stats.itemsScanned} 词条，耗时 ${stats.elapsedMs}ms`);
        if (taskId && pendingTasks.has(taskId)) {
            pendingTasks.get(taskId)?.resolve(result);
            pendingTasks.delete(taskId);
        }
    } else if (status === 'rerank_complete') {
        logDiagnostic(`WebGPU 交叉重排计算完成！(耗时 ${stats.elapsedMs}ms)`);
        if (taskId && pendingTasks.has(taskId)) {
            pendingTasks.get(taskId)?.resolve(result);
            pendingTasks.delete(taskId);
        }
    }
};

const initWorkers = async () => {
    logDiagnostic('开始检索核心数据库缓存...');
    statusMsg.value = '加载核心数据...';
    
    try {
        const CACHE_KEY_MATRIX = 'rag_vector_matrix_dmeta_v1';
        const CACHE_KEY_METADATA = 'rag_metadata_dmeta_v1';
        
        let matrixBuffer = await localforage.getItem<ArrayBuffer>(CACHE_KEY_MATRIX);
        let metadataJson = await localforage.getItem<any>(CACHE_KEY_METADATA);
        
        if (!matrixBuffer || !metadataJson) {
            logDiagnostic('未命中本地数据库，开始网络同步...');
            const [matrixRes, metaRes] = await Promise.all([
                fetch('/data/frontend_vectors_dmeta_small.bin'),
                fetch('/data/frontend_metadata_dmeta_small.json')
            ]);
            
            if (!matrixRes.ok || !metaRes.ok) throw new Error('网络请求资源失败');
            
            matrixBuffer = await matrixRes.arrayBuffer();
            metadataJson = await metaRes.json();
            
            logDiagnostic('数据同步完成，持久化到 IndexedDB 以备后续秒开...');
            await localforage.setItem(CACHE_KEY_MATRIX, matrixBuffer);
            await localforage.setItem(CACHE_KEY_METADATA, metadataJson);
        } else {
            logDiagnostic('成功利用 IndexedDB 加载核心数据（秒开模式就绪）');
        }

        logDiagnostic('利用 Zero-Copy(Transferable Objects) 转交内存所有权给 Worker...');
        const transferBuffer = matrixBuffer.slice(0) as ArrayBuffer; // 🌟 拷贝一份内存交给 Worker，避免原 buffer 被掏空导致二次 init 崩溃
        
        await dispatchToWorker('INIT', {
            metadata: metadataJson,
            vectorMatrix: new Int8Array(transferBuffer)
        }, [transferBuffer]);

    } catch (e: any) {
        errorMsg.value = `加载失败: ${e.message}`;
        logDiagnostic(`致命错误: ${e.message}`);
    }
};

onMounted(() => {
    initWorkers();
});

onUnmounted(() => {
    if (myWorker) myWorker.terminate();
});

// --- Search Logic ---
const handleSearch = async () => {
    if (!searchQuery.value.trim() || isProcessing.value || !isWorkerReady.value) return;

    errorMsg.value = null;
    isProcessing.value = true;
    results.value = [];
    statusMsg.value = '分词与向量化中...';

    try {
        const tStart = performance.now();
        logDiagnostic('向 Worker 提交提问...');
        // 1. 发送搜索指令并等待 Worker 的回调
        const localMatches = await dispatchToWorker('SEARCH', searchQuery.value.trim());
        
        const tSearchEnd = performance.now();

        // searchAndRank 返回的已经是以 otid 聚合后的 { otid, score }[]，直接提取 Top 15
        const topIds = localMatches.slice(0, 15).map((r: any) => r.otid);

        logDiagnostic(`粗排返回 ${localMatches.length} 篇，取 Top ${topIds.length} 进入精排`);
        console.log('[DEBUG] LocalMatches:', localMatches);
        console.log('[DEBUG] Final TopIds:', topIds);

        if (topIds.length === 0) {
            statusMsg.value = '未找到匹配的答案';
            logDiagnostic('本地向量苦无反馈。');
            isProcessing.value = false;
            return;
        }
        
        // 2. Fetch Backend Contents (使用 POST)
        statusMsg.value = '请求原文数据...';
        logDiagnostic(`命中 ${topIds.length} 篇相关文献，向后端 POST 拉取原文...`);
        
        const response = await fetch(`/api/get_answers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ otids: topIds })
        });

        if (!response.ok) throw new Error(`后端响应异常: ${response.status}`);
        
        const result = await response.json();
        const tFetchEnd = performance.now();
        
        if (result.data) {
            // 3. 进入同端精排阶段
            statusMsg.value = '深入研读细节并交叉比对...';
            const finalRender = await dispatchToWorker('RERANK', { query: searchQuery.value.trim(), documents: result.data });

            const tRerankEnd = performance.now();
            results.value = finalRender;
            
            const stats = {
                totalMs: (tRerankEnd - tStart).toFixed(1),
                searchMs: (tSearchEnd - tStart).toFixed(1),
                fetchMs: (tFetchEnd - tSearchEnd).toFixed(1),
                rerankMs: (tRerankEnd - tFetchEnd).toFixed(1)
            };

            emit('trace-updated', {
                query: searchQuery.value,
                results: finalRender,
                stats
            });
            statusMsg.value = `找到 ${finalRender.length} 篇高精度原文献 (总耗时 ${stats.totalMs}ms)`;
            logDiagnostic('精排展示完毕！');
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
  <div class="flex flex-col h-full bg-slate-900/10 backdrop-blur-md rounded-2xl border border-white/5 overflow-hidden shadow-2xl relative">
    
    <!-- Controls Header -->
    <div class="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-white/5">
        <div class="flex items-center gap-2">
             <div class="flex items-center gap-2 p-1 bg-black/30 rounded-lg border border-white/5 mr-2">
                <button 
                  v-for="m in models" :key="m.id"
                  @click="selectedModelId = m.id as any"
                  class="px-2 py-1 rounded text-[9px] font-bold uppercase transition-all"
                  :class="selectedModelId === m.id ? 'bg-blue-600 text-white' : 'text-slate-500 hover:text-slate-300'"
                >
                    {{ m.name }}
                </button>
             </div>
             <select disabled class="bg-black/20 border border-white/5 text-[9px] font-bold text-slate-400 uppercase outline-none px-2 py-1.5 rounded-lg cursor-not-allowed">
                <option value="wasm">WASM Worker</option>
             </select>
        </div>
        
        <div class="flex items-center gap-2">
            <div class="w-2 h-2 rounded-full" :class="isProcessing ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'"></div>
            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{{ statusMsg }}</span>
        </div>
    </div>

    <!-- Error Banner -->
    <div v-if="errorMsg" class="px-6 py-3 bg-red-500/10 border-b border-red-500/20 text-red-400 text-xs flex items-center gap-2">
        <AlertCircle class="w-4 h-4" />
        {{ errorMsg }}
    </div>

    <!-- Search Input Area -->
    <div class="p-6 border-b border-white/5 bg-white/5">
        <div class="relative group">
            <input 
                v-model="searchQuery"
                @keydown.enter="handleSearch"
                placeholder="在此输入校园政策关键词或特定问题进行检索..."
                class="w-full bg-slate-900/50 border border-white/10 rounded-xl px-12 py-4 text-slate-100 placeholder-slate-500 outline-none focus:ring-2 focus:ring-blue-500/40 transition-all font-medium"
            />
            <Search class="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
            <button 
                @click="handleSearch"
                :disabled="!searchQuery.trim() || isProcessing || !isWorkerReady"
                class="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-2 bg-blue-600 rounded-lg font-bold text-xs hover:bg-blue-500 disabled:opacity-50 transition-all shadow-lg active:scale-95"
            >
                <span v-if="!isProcessing">搜索</span>
                <Loader2 v-else class="w-4 h-4 animate-spin" />
            </button>
        </div>

        <!-- Inline Diagnostics -->
        <div class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
            <div v-for="(log, i) in diagnosticLogs" :key="i" class="text-[9px] font-mono text-slate-500 flex items-center gap-1">
                <Info v-if="i === 0" class="w-2.5 h-2.5 text-blue-400" />
                {{ log }}
            </div>
        </div>
    </div>

    <!-- Results Area -->
    <div class="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
        <div v-if="results.length === 0 && !isProcessing" class="h-full flex flex-col items-center justify-center opacity-30 text-center">
            <div class="w-16 h-16 rounded-full border-2 border-dashed border-slate-700 flex items-center justify-center mb-4">
                 <Search class="w-6 h-6" />
            </div>
            <p class="text-sm font-medium">输入关键词并按回车开始本地语义搜索</p>
            <p class="text-[10px] uppercase tracking-tighter mt-1">AI 检索已就绪 · WebGPU 加速开启</p>
        </div>

        <div v-for="(res, i) in results" :key="res.id" 
             class="group p-5 rounded-2xl bg-white/5 border border-white/5 hover:border-blue-500/30 transition-all animate-in fade-in slide-in-from-bottom-2 duration-300">
            
            <div class="flex items-start justify-between mb-2">
                 <div class="flex items-center gap-2">
                    <span class="text-[10px] font-bold bg-slate-800 text-slate-500 px-2 py-0.5 rounded">#{{ i + 1 }}</span>
                    <h3 class="font-bold text-slate-100 group-hover:text-blue-400 transition-colors">{{ res.ot_title || '未命名政策文档' }}</h3>
                 </div>
                 <div class="flex items-center gap-1.5 px-2 py-1 rounded bg-blue-500/10 text-blue-400">
                    <Zap class="w-3 h-3" />
                    <span class="text-[10px] font-mono font-bold px-1" title="这是经过显卡 FP16 精度深度重推算出的相似度" >{{ ((res.rerankScore || 0) * 100).toFixed(1) }}% Match</span>
                 </div>
            </div>

            <p v-if="res.bestSentence && res.bestSentence.length > 0" class="text-sm font-semibold text-white/90 leading-relaxed mb-2 bg-blue-600/10 p-3 rounded-lg border-l-2 border-blue-500">
                <Layers class="w-4 h-4 inline-block mb-1 text-blue-400 mr-1"/>
                {{ res.bestSentence }}
            </p>

            <p class="text-sm text-slate-400 leading-relaxed mb-4 line-clamp-3">
                {{ res.ot_text || res.content_json }}
            </p>

            <div class="flex items-center justify-between pt-4 border-t border-white/5">
                <div class="flex items-center gap-4">
                    <div v-if="res.publish_time" class="flex items-center gap-1.5 text-[10px] text-slate-500 font-bold uppercase">
                        <Clock class="w-3 h-3" />
                        {{ res.publish_time }}
                    </div>
                </div>
                <a v-if="res.link" :href="res.link" target="_blank" class="flex items-center gap-1.5 text-[10px] font-bold text-blue-500 hover:text-blue-400">
                    查看原文
                    <ExternalLink class="w-3 h-3" />
                </a>
            </div>
        </div>
    </div>

    <!-- Footnote -->
    <div class="px-6 py-3 border-t border-white/5 bg-black/20 flex justify-between items-center text-[9px] font-bold text-slate-600 uppercase tracking-widest">
        <span>基于校策通本地异步 Worker 引擎 v3.0</span>
        <div class="flex items-center gap-3">
            <span class="flex items-center gap-1" title="基于 CPU 极速提取"><Cpu class="w-2.5 h-2.5 text-slate-400" /> Vector Worker Q8</span>
            <span class="flex items-center gap-1" title="基于纯净前端显卡算力"><svg class="w-3 h-3 text-red-500/70" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg> Rerank Worker FP16</span>
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
