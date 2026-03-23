<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import { Play, Cpu, Loader2, Target } from 'lucide-vue-next';
import VectorWorker from '../worker/embedding.worker.ts?worker';

const testsetRaw = ref('[\n  {"query": "2026 保研", "expected_otid": "otid_xxxx"},\n  {"query": "留学的推荐信需要找谁写", "expected_otid": "otid_xxxx"}\n]');
const isRunning = ref(false);
const totalQueries = ref(0);
const processedCount = ref(0);

const metrics = ref({
    coarseHit1: 0,
    coarseHit5: 0,
    fineHit1: 0,
    fineHit5: 0,
});

let vectorWorker: Worker | null = null;
const isWorkersReady = ref(false);

const reportLines = ref<string[]>([]);
const hasFinished = ref(false);
const badCases = ref<any[]>([]);

// Format date like evaluate.ts
const getFormattedTime = () => {
    const d = new Date();
    return `${d.getFullYear()}${(d.getMonth()+1).toString().padStart(2,'0')}${d.getDate().toString().padStart(2,'0')}_${d.getHours().toString().padStart(2,'0')}${d.getMinutes().toString().padStart(2,'0')}`;
};

const bgLog = (msg: string) => {
    // 纯后台记录，不渲染到界面，彻底避免卡顿
    reportLines.value.push(msg);
    console.log(msg); // 也可以保留控制台输出方便调试
};

// Helper Formatters
const pRateRaw = (val: number) => totalQueries.value === 0 ? '0.0' : ((val / totalQueries.value) * 100).toFixed(1);

// 导出评估报告
const downloadReport = () => {
    let content = `==================================================\n`;
    content += `          前端端侧 (WASM + WebGPU) RAG 评测报告          \n`;
    content += `==================================================\n`;
    content += `测评时间: ${new Date().toLocaleString()}\n`;
    content += `总测试用例数: ${totalQueries.value}\n`;
    content += `--------------------------------------------------\n`;
    content += `[WASM第一跳] 粗排 Hit@1: ${pRateRaw(metrics.value.coarseHit1)}%\n`;
    content += `[WASM第一跳] 粗排 Hit@5: ${pRateRaw(metrics.value.coarseHit5)}%\n`;
    content += `[GPU第二跳] 终排 Hit@1: ${pRateRaw(metrics.value.fineHit1)}%\n`;
    content += `[GPU第二跳] 终排 Hit@5: ${pRateRaw(metrics.value.fineHit5)}%\n`;
    content += `==================================================\n\n`;

    // Bad Case 排序：未命中排最前，其次按名次数字从大到小（名次越靠后越差）
    const sortedBadCases = [...badCases.value].sort((a, b) => {
        const rankA = a.fineRank === '未命中' ? 999999 : a.fineRank;
        const rankB = b.fineRank === '未命中' ? 999999 : b.fineRank;
        return rankB - rankA;
    });

    const top10Worst = sortedBadCases.slice(0, 10);
    const remainingBadCases = sortedBadCases.slice(10);

    content += `==================================================\n`;
    content += `                Bad Case 全局深度分析 (Top 10 Worst)\n`;
    content += `==================================================\n\n`;

    if (top10Worst.length > 0) {
        content += `--- [重点关注] 终排未能 Hit@1 的前 10 个 Worst Case ---\n`;
        top10Worst.forEach((bc: any, idx: number) => {
            content += `[WORST ${idx + 1}]\n`;
            content += `    Query: "${bc.query}"\n`;
            content += `    --------------------------------\n`;
            content += `    🔴 预期 OTID: ${bc.expected_otid}\n`;
            content += `    实际排名 (WASM粗排): ${bc.coarseRank}\n`;
            content += `    实际排名 (WebGPU终排): ${bc.fineRank}\n`;
            content += `    --------------------------------\n`;
            content += `    ⚠️ WebGPU 实际检索置顶 Top 3 结果排查:\n`;
            bc.top3.forEach((t: any, tIdx: number) => {
                content += `      ${tIdx + 1}. ${t.otid} (得分: ${t.score.toFixed(4)}) - 标题: ${t.title || '无标题'}\n`;
            });
            content += `    ================================\n\n`;
        });
    } else {
        content += `    🎉 恭喜！当前评测集 100% 达成 Hit@1！\n\n`;
    }

    if (remainingBadCases.length > 0) {
        content += `--- [其它 Bad Case] 共 ${remainingBadCases.length} 个 ---\n`;
        remainingBadCases.forEach((bc: any, idx: number) => {
            content += `[BAD ${idx + 11}] Query: ${bc.query} (终排 Rank: ${bc.fineRank})\n`;
        });
        content += `\n`;
    }

    content += `==================================================\n`;
    content += `                 评测执行明细流水日志\n`;
    content += `==================================================\n`;
    content += reportLines.value.join('\n');

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `WebGPU_Rerank_Report_${getFormattedTime()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
};

// --- Promise Based Worker Caller ---
const runWorkerAsync = (worker: Worker, type: string, payload: any): Promise<any> => {
    return new Promise((resolve, reject) => {
        const handler = (event: MessageEvent) => {
            const data = event.data;
            if (data.status === 'search_complete' || data.status === 'rerank_complete') {
                worker.removeEventListener('message', handler);
                resolve(data.result);
            } else if (data.status === 'error') {
                worker.removeEventListener('message', handler);
                reject(new Error(data.error));
            }
        };
        worker.addEventListener('message', handler);
        worker.postMessage({ type, payload });
    });
};

const initWorkers = () => {
    vectorWorker = new VectorWorker();
    
    vectorWorker.onmessage = (e) => {
        if (e.data.status === 'ready') isWorkersReady.value = true;
    };
    
    vectorWorker.postMessage({
        type: 'INIT',
        payload: {
            metadataUrl: '/data/frontend_metadata_dmeta_small.json',
            vectorsUrl: '/data/frontend_vectors_dmeta_small.bin'
        }
    });
};

const loadTestset = async () => {
    try {
        const response = await fetch('/data/eval_testset.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        // Beautify the JSON into the text area
        testsetRaw.value = JSON.stringify(data, null, 2);
        bgLog(`✓ 成功从前端公有资源池加载 \`eval_testset.json\`！共装有 ${data.length} 条压测标准用例。`);
    } catch (e: any) {
        bgLog(`[错误] 内置测试集装载失败: ${e.message}。请确保 /public/data/ 下有该文件。`);
    }
};

onMounted(() => {
    initWorkers();
    loadTestset();
});

onUnmounted(() => {
    if (vectorWorker) vectorWorker.terminate();
});

const startBenchmark = async () => {
    if (!isWorkersReady.value) {
        alert('引擎还在加载中，请稍候！');
        return;
    }

    let parsedTestset: Array<{query: string, expected_otid: string}> = [];
    try {
        parsedTestset = JSON.parse(testsetRaw.value);
    } catch (e) {
        alert('JSON 格式错误，请检查输入！');
        return;
    }

    if (!Array.isArray(parsedTestset) || parsedTestset.length === 0) {
        alert('测试集不能空！');
        return;
    }

    isRunning.value = true;
    hasFinished.value = false;
    reportLines.value = [];
    badCases.value = [];
    metrics.value = { coarseHit1: 0, coarseHit5: 0, fineHit1: 0, fineHit5: 0 };

    bgLog('============= 开始 E2E 流水线评测 =============');
    
    for (let i = 0; i < parsedTestset.length; i++) {
        const item = parsedTestset[i];
        
        try {
            // 第一跳：WASM 粗排
            const coarseResults = await runWorkerAsync(vectorWorker!, 'SEARCH', item.query.trim());
            const top15 = coarseResults.slice(0, 15);
            
            // 粗排核算
            let coarseRank = -1;
            for(let k=0; k<top15.length; k++) {
                if((top15[k].otid || top15[k].pkid) === item.expected_otid) {
                    coarseRank = k + 1; break;
                }
            }
            if (coarseRank === 1) metrics.value.coarseHit1++;
            if (coarseRank >= 1 && coarseRank <= 5) metrics.value.coarseHit5++;

            if (coarseRank === -1) {
                bgLog(`[丢失] Q${i+1}: "${item.query}" -> 粗排未能挤进 Top 15，打捞失败。`);
                badCases.value.push({
                    query: item.query,
                    expected_otid: item.expected_otid,
                    coarseRank: '未命中',
                    fineRank: '未命中',
                    top3: top15.slice(0, 3).map((r: any) => ({
                        otid: r.otid || r.pkid,
                        title: r.title || '无标题',
                        score: r.score || 0
                    }))
                });
                
                processedCount.value++;
                continue; 
            }

            // 第二跳：获取真实内容（模拟真实流请求）
            const topIds = top15.map((r: any) => r.otid || r.pkid);
            const response = await fetch(`/api/get_answers`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ otids: topIds })
            });
            const textResult = await response.json();
            const docsToRerank = textResult.data || [];

            // 第三跳：同一 Worker 内精排重推
            const fineResults = await runWorkerAsync(vectorWorker!, 'RERANK', {
                query: item.query.trim(),
                documents: docsToRerank
            });

            // 精排核算
            let fineRank = -1;
            for(let k=0; k<fineResults.length; k++) {
                if((fineResults[k].otid || fineResults[k].pkid) === item.expected_otid) {
                    fineRank = k + 1; break;
                }
            }
            if (fineRank === 1) metrics.value.fineHit1++;
            if (fineRank >= 1 && fineRank <= 5) metrics.value.fineHit5++;

            if (fineRank !== 1) {
                badCases.value.push({
                    query: item.query,
                    expected_otid: item.expected_otid,
                    coarseRank: coarseRank,
                    fineRank: fineRank === -1 ? '未命中' : fineRank,
                    top3: fineResults.slice(0, 3).map((r: any) => ({
                        otid: r.otid || r.pkid,
                        title: r.title || '无标题',
                        score: r.rerankScore || r.score || 0
                    }))
                });
            }

            bgLog(`[成功] Q${i+1}: "${item.query}" -> 粗排名次: ${coarseRank} | WebGPU拯救名次: ${fineRank}`);

        } catch (e: any) {
            bgLog(`[错误] Q${i+1}: "${item.query}" 测试异常: ${e.message}`);
        }
        
        processedCount.value++;
    }

    bgLog('============= E2E 流水线评测完成 =============');
    isRunning.value = false;
    hasFinished.value = true;
    
    // 自动触发出具报告
    downloadReport();
};

const pRate = (val: number) => totalQueries.value === 0 ? '0.0%' : ((val / totalQueries.value) * 100).toFixed(1) + '%';
</script>

<template>
  <div class="bg-indigo-900/20 backdrop-blur-xl border border-indigo-500/30 rounded-2xl shadow-2xl p-6 flex flex-col relative overflow-hidden h-full">
    <div class="absolute -top-10 -right-10 w-40 h-40 bg-purple-500/10 blur-[60px] pointer-events-none rounded-full" />
    
    <header class="z-10 border-b border-indigo-500/20 pb-4 w-full flex items-center justify-between">
      <div>
        <h2 class="text-xl font-bold tracking-tight text-indigo-100 flex items-center gap-2">
            <Target class="w-5 h-5 text-indigo-400" />
            前端实时基准评测舱 (RAG E2E Benchmarker)
        </h2>
        <p class="text-indigo-300/60 text-[11px] mt-1 font-mono uppercase tracking-widest">
            Evaluation metrics running on internal WASM & WebGPU engines.
        </p>
      </div>
      <button 
        @click="startBenchmark" 
        :disabled="isRunning || !isWorkersReady"
        class="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white px-5 py-2.5 rounded-lg font-bold text-sm shadow-lg flex items-center gap-2 transition-colors">
        <Loader2 v-if="isRunning" class="w-4 h-4 animate-spin" />
        <Play v-else class="w-4 h-4" />
        {{ isRunning ? `正在压测 ${processedCount}/${totalQueries}` : (isWorkersReady ? '开始端侧测评' : '引擎启动中...') }}
      </button>
    </header>

    <div class="flex flex-col md:flex-row gap-6 mt-6 z-10">
        <!-- Input Panel -->
        <div class="flex-1 flex flex-col gap-2">
            <label class="text-xs font-bold text-indigo-300 uppercase tracking-widest flex justify-between">
                <span>注入测试集 (JSON Array)</span>
            </label>
            <textarea 
                v-model="testsetRaw" 
                spellcheck="false"
                :disabled="isRunning"
                class="w-full h-48 bg-[#0a0f1c] border border-indigo-500/20 rounded-xl p-4 text-xs font-mono text-indigo-200 focus:border-indigo-500/50 outline-none custom-scrollbar resize-none"
            ></textarea>
            
            <div class="mt-4 flex flex-col gap-2 bg-black/30 p-4 rounded-xl border border-indigo-500/10">
                <h3 class="text-[10px] text-indigo-400 font-bold uppercase tracking-widest mb-1">端侧测试指标仪表盘 (E2E Accuracy Metrics)</h3>
                
                <div class="grid grid-cols-2 gap-3 mb-2">
                   <div class="bg-indigo-500/10 rounded-lg p-3 text-center border border-indigo-500/20">
                      <div class="text-[9px] text-indigo-300 font-bold uppercase mb-1">粗排 Hits@1</div>
                      <div class="text-xl font-bold font-mono text-indigo-100">{{ pRate(metrics.coarseHit1) }}</div>
                   </div>
                   <div class="bg-indigo-500/10 rounded-lg p-3 text-center border border-indigo-500/20">
                      <div class="text-[9px] text-indigo-300 font-bold uppercase mb-1">粗排 Hits@5</div>
                      <div class="text-xl font-bold font-mono text-indigo-100">{{ pRate(metrics.coarseHit5) }}</div>
                   </div>
                </div>
                
                <div class="grid grid-cols-2 gap-3">
                   <div class="bg-purple-500/20 rounded-lg p-3 text-center border border-purple-500/30">
                      <div class="text-[9px] text-purple-300 font-bold uppercase mb-1 flex items-center justify-center gap-1"><Cpu class="w-2 h-2"/> 终排 Hits@1</div>
                      <div class="text-xl font-bold font-mono text-purple-100">{{ pRate(metrics.fineHit1) }}</div>
                   </div>
                   <div class="bg-purple-500/20 rounded-lg p-3 text-center border border-purple-500/30">
                      <div class="text-[9px] text-purple-300 font-bold uppercase mb-1 flex items-center justify-center gap-1"><Cpu class="w-2 h-2"/> 终排 Hits@5</div>
                      <div class="text-xl font-bold font-mono text-purple-100">{{ pRate(metrics.fineHit5) }}</div>
                   </div>
                </div>
            </div>
        </div>

        <div class="flex-1 flex flex-col gap-2 mt-4 md:mt-0">
            <!-- 指标面板全宽化，代替原本的日志区域 -->
            <div class="flex justify-between items-center bg-black/30 p-4 rounded-xl border border-indigo-500/10">
                <h3 class="text-[12px] text-indigo-400 font-bold uppercase tracking-widest leading-none">端侧双跳测试指标核心 (E2E Accuracy Metrics)</h3>
                <button v-if="hasFinished" @click="downloadReport" class="px-3 py-1 bg-indigo-500/20 hover:bg-indigo-500/40 border border-indigo-500/50 rounded text-[10px] text-indigo-200 uppercase flex items-center gap-1 transition">导出离线测评报告.txt</button>
            </div>
            
            <div class="grid grid-cols-2 gap-3 mb-2 flex-hidden-grow">
               <div class="bg-indigo-500/10 rounded-lg p-3 text-center border border-indigo-500/20 flex flex-col justify-center h-full">
                  <div class="text-[11px] text-indigo-300 font-bold uppercase mb-1">第一跳 WASM 粗排 Hits@1</div>
                  <div class="text-3xl font-bold font-mono text-indigo-100">{{ pRate(metrics.coarseHit1) }}</div>
               </div>
               <div class="bg-indigo-500/10 rounded-lg p-3 text-center border border-indigo-500/20 flex flex-col justify-center h-full">
                  <div class="text-[11px] text-indigo-300 font-bold uppercase mb-1">第一跳 WASM 粗排 Hits@5</div>
                  <div class="text-3xl font-bold font-mono text-indigo-100">{{ pRate(metrics.coarseHit5) }}</div>
               </div>
            </div>
            
            <div class="grid grid-cols-2 gap-3 flex-1 flex-hidden-grow">
               <div class="bg-purple-500/20 rounded-lg p-3 text-center border border-purple-500/30 flex flex-col justify-center h-full">
                  <div class="text-[11px] text-purple-300 font-bold uppercase mb-1 flex items-center justify-center gap-1"><Cpu class="w-3 h-3"/> 第二跳 WebGPU 精排 Hits@1</div>
                  <div class="text-3xl font-bold font-mono text-purple-100">{{ pRate(metrics.fineHit1) }}</div>
               </div>
               <div class="bg-purple-500/20 rounded-lg p-3 text-center border border-purple-500/30 flex flex-col justify-center h-full">
                  <div class="text-[11px] text-purple-300 font-bold uppercase mb-1 flex items-center justify-center gap-1"><Cpu class="w-3 h-3"/> 第二跳 WebGPU 精排 Hits@5</div>
                  <div class="text-3xl font-bold font-mono text-purple-100">{{ pRate(metrics.fineHit5) }}</div>
               </div>
            </div>
        </div>
    </div>
  </div>
</template>

<style scoped>
.flex-hidden-grow { min-height: 100px; }
.custom-scrollbar::-webkit-scrollbar { width: 4px; }
.custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 10px; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.6); }
</style>
