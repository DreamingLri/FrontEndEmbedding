import { onMounted, onUnmounted, ref } from 'vue';
import localforage from 'localforage';
import VectorWorker from '../worker/embedding.worker.ts?worker';
import type {
    PipelineDecision,
    PipelineDocumentRecord,
    SearchPipelineResult,
} from '../worker/search_pipeline.ts';
import type { SearchRejection } from '../worker/vector_engine.ts';
import { buildSearchTraceData, type SearchTraceData } from '../utils/searchUi';

type SearchResultDoc = PipelineDocumentRecord;
type WorkerDecision = PipelineDecision;
type WorkerSearchResult = SearchPipelineResult;

type WorkerTaskType = 'INIT' | 'SEARCH';

type PendingTask = {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    type: WorkerTaskType;
};

type SearchRuntimeOptions = {
    onTraceUpdated?: (traceData: SearchTraceData) => void;
};

type WorkerMessage = {
    taskId?: string;
    status:
        | 'loading'
        | 'progress'
        | 'info'
        | 'ready'
        | 'error'
        | 'search_complete';
    message?: string;
    result?: WorkerSearchResult;
    error?: string;
    stats?: {
        elapsedMs?: string;
        itemsScanned?: number;
        partitionUsed?: boolean;
        partitionCandidateCount?: number;
    };
};

type IndexPayload = {
    matrixBuffer: ArrayBuffer;
    metadataJson: unknown;
};

const SEARCH_INDEX_CONFIG = {
    mainDbVersion: 'main_v2_plus',
    indexCacheVersion: '20260417-main_v2_plus',
};

const SEARCH_INDEX_RESOURCE = {
    vectorUrl: `/data/frontend_vectors_dmeta_small_${SEARCH_INDEX_CONFIG.mainDbVersion}.bin`,
    metadataUrl: `/data/frontend_metadata_dmeta_small_${SEARCH_INDEX_CONFIG.mainDbVersion}.json`,
    cacheKeys: {
        matrix: `rag_vector_matrix_dmeta_${SEARCH_INDEX_CONFIG.indexCacheVersion}`,
        metadata: `rag_metadata_dmeta_${SEARCH_INDEX_CONFIG.indexCacheVersion}`,
    },
};

function getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
}

export function useSearchRuntime(options: SearchRuntimeOptions = {}) {
    const searchQuery = ref('');
    const results = ref<SearchResultDoc[]>([]);
    const weakResults = ref<SearchResultDoc[]>([]);
    const rejectionInfo = ref<SearchRejection | null>(null);
    const decisionInfo = ref<WorkerDecision | null>(null);
    const isProcessing = ref(false);
    const statusMsg = ref('正在唤起 Web Worker...');
    const errorMsg = ref<string | null>(null);
    const diagnosticLogs = ref<string[]>([]);
    const isWorkerReady = ref(false);

    const pendingTasks = new Map<string, PendingTask>();
    let worker: Worker | null = null;

    const logDiagnostic = (message: string) => {
        const time = new Date().toLocaleTimeString();
        diagnosticLogs.value = [`[${time}] ${message}`, ...diagnosticLogs.value].slice(
            0,
            5,
        );
    };

    const resolvePendingTask = (taskId: string | undefined, value: unknown) => {
        if (!taskId) {
            return;
        }

        const task = pendingTasks.get(taskId);
        if (!task) {
            return;
        }

        task.resolve(value);
        pendingTasks.delete(taskId);
    };

    const rejectPendingTask = (taskId: string | undefined, error: Error) => {
        if (!taskId) {
            return;
        }

        const task = pendingTasks.get(taskId);
        if (!task) {
            return;
        }

        task.reject(error);
        pendingTasks.delete(taskId);
    };

    const rejectAllPendingTasks = (message: string) => {
        pendingTasks.forEach((task) => task.reject(new Error(message)));
        pendingTasks.clear();
    };

    const handleWorkerMessage = (event: MessageEvent<WorkerMessage>) => {
        const { status, message, result, error, stats, taskId } = event.data;
        const pendingTask = taskId ? pendingTasks.get(taskId) : undefined;

        if (status === 'loading' || status === 'progress') {
            if (message) {
                statusMsg.value = message;
                logDiagnostic(`[引擎通知] ${message}`);
            }
            return;
        }

        if (status === 'info') {
            if (message) {
                logDiagnostic(`[系统提示] ${message}`);
            }
            return;
        }

        if (status === 'ready') {
            isWorkerReady.value = true;
            statusMsg.value = 'AI 引擎就绪';
            if (message) {
                logDiagnostic(message);
            }
            resolvePendingTask(taskId, true);
            return;
        }

        if (status === 'error') {
            isProcessing.value = false;
            if (pendingTask?.type === 'INIT') {
                isWorkerReady.value = false;
            }

            const failureLabel =
                pendingTask?.type === 'SEARCH' ? '检索失败' : '引擎启动失败';
            const errorMessage = `${failureLabel}: ${error ?? '未知错误'}`;
            errorMsg.value = errorMessage;
            logDiagnostic(
                `${pendingTask?.type === 'SEARCH' ? '检索错误' : '致命错误'}: ${error ?? '未知错误'}`,
            );
            rejectPendingTask(taskId, new Error(errorMessage));
            return;
        }

        if (status === 'search_complete') {
            if (stats) {
                logDiagnostic(
                    `统一链路完成主搜索阶段，扫描 ${stats.itemsScanned ?? '-'} 条，耗时 ${stats.elapsedMs ?? '-'}ms`,
                );
                if (stats.partitionUsed) {
                    logDiagnostic(
                        `[Partition] candidate=${stats.partitionCandidateCount ?? '-'}`,
                    );
                }
            }
            resolvePendingTask(taskId, result);
        }
    };

    const createWorkerInstance = () => {
        const nextWorker = new VectorWorker();
        nextWorker.onmessage = handleWorkerMessage;
        return nextWorker;
    };

    const dispatchToWorker = <T>(
        type: WorkerTaskType,
        payload: unknown,
        transfer: Transferable[] = [],
    ): Promise<T> =>
        new Promise((resolve, reject) => {
            if (!worker) {
                reject(new Error('Worker 尚未创建'));
                return;
            }

            const taskId = crypto.randomUUID();
            pendingTasks.set(taskId, {
                resolve: resolve as (value: unknown) => void,
                reject,
                type,
            });
            worker.postMessage({ type, payload, taskId }, transfer);
        });

    const loadIndexPayload = async (): Promise<IndexPayload> => {
        const { matrix: matrixCacheKey, metadata: metadataCacheKey } =
            SEARCH_INDEX_RESOURCE.cacheKeys;

        let matrixBuffer = await localforage.getItem<ArrayBuffer>(matrixCacheKey);
        let metadataJson = await localforage.getItem(metadataCacheKey);

        if (matrixBuffer && metadataJson) {
            logDiagnostic('已从 IndexedDB 命中本地缓存');
            return { matrixBuffer, metadataJson };
        }

        logDiagnostic('本地缓存未命中，开始同步数据...');
        const [matrixRes, metaRes] = await Promise.all([
            fetch(SEARCH_INDEX_RESOURCE.vectorUrl),
            fetch(SEARCH_INDEX_RESOURCE.metadataUrl),
        ]);

        if (!matrixRes.ok || !metaRes.ok) {
            throw new Error('网络请求核心数据失败');
        }

        matrixBuffer = await matrixRes.arrayBuffer();
        metadataJson = await metaRes.json();

        logDiagnostic('同步完成，正在写入 IndexedDB...');
        await localforage.setItem(matrixCacheKey, matrixBuffer);
        await localforage.setItem(metadataCacheKey, metadataJson);

        return { matrixBuffer, metadataJson };
    };

    const initWorker = async () => {
        errorMsg.value = null;
        isWorkerReady.value = false;
        statusMsg.value = '加载核心数据中...';
        logDiagnostic('开始检查核心数据缓存...');

        try {
            const { matrixBuffer, metadataJson } = await loadIndexPayload();
            const transferBuffer = matrixBuffer.slice(0);

            await dispatchToWorker<boolean>(
                'INIT',
                {
                    metadata: metadataJson,
                    vectorMatrix: new Int8Array(transferBuffer),
                },
                [transferBuffer],
            );
        } catch (error) {
            const message = getErrorMessage(error);
            errorMsg.value = `加载失败: ${message}`;
            logDiagnostic(`初始化失败: ${message}`);
        }
    };

    const resetSearchState = () => {
        errorMsg.value = null;
        results.value = [];
        weakResults.value = [];
        rejectionInfo.value = null;
        decisionInfo.value = null;
    };

    const applySearchResult = (searchResult: WorkerSearchResult) => {
        const finalRender = searchResult.results || [];
        const localWeakResults = searchResult.weakResults || [];
        const finalDecision = searchResult.finalDecision || null;
        const rejection = searchResult.rejection || null;
        const trace = searchResult.trace;

        results.value = finalRender;
        weakResults.value = localWeakResults;
        rejectionInfo.value = rejection;
        decisionInfo.value = finalDecision;
        options.onTraceUpdated?.(buildSearchTraceData(searchResult));

        if (!finalDecision) {
            statusMsg.value = '未能解析当前查询结果';
            logDiagnostic('未收到统一链路的最终决策');
            return;
        }

        if (finalDecision.behavior === 'reject') {
            if (rejection?.reason === 'low_topic_coverage') {
                statusMsg.value =
                    '当前知识库暂无该主题的直接内容，已给出弱相关入口。';
                logDiagnostic('主题覆盖不足，已拒绝直接回答并保留弱相关入口');
                return;
            }

            statusMsg.value = '当前查询未能形成稳定的可信结果，系统已拒答。';
            logDiagnostic('系统判定当前查询应拒答');
            return;
        }

        statusMsg.value = `找到 ${finalRender.length} 篇相关结果（总耗时 ${Number(
            trace?.totalMs ?? 0,
        ).toFixed(1)}ms）`;
        logDiagnostic('统一链路已完成结果展示');
    };

    const handleSearch = async () => {
        const query = searchQuery.value.trim();
        if (!query || isProcessing.value || !isWorkerReady.value) {
            return;
        }

        searchQuery.value = query;
        isProcessing.value = true;
        statusMsg.value = '正在分词与向量化...';
        resetSearchState();

        try {
            logDiagnostic('开始提交检索请求...');
            const searchResult = await dispatchToWorker<WorkerSearchResult>(
                'SEARCH',
                query,
            );
            applySearchResult(searchResult);
        } catch (error) {
            console.error(error);
            if (!errorMsg.value) {
                const message = getErrorMessage(error);
                errorMsg.value = `检索发生错误: ${message}`;
                logDiagnostic(`检索异常: ${message}`);
            }
        } finally {
            isProcessing.value = false;
        }
    };

    onMounted(() => {
        worker = createWorkerInstance();
        void initWorker();
    });

    onUnmounted(() => {
        worker?.terminate();
        worker = null;
        rejectAllPendingTasks('Vector worker 已终止');
    });

    return {
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
        handleSearch,
    };
}
