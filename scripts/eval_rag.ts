import * as fs from 'fs';
import * as path from 'path';
import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers';
import { performance } from 'perf_hooks';

// We import the same vector_engine algorithms the frontend uses
import { buildBM25Stats, searchAndRank, getQuerySparse, type BM25Stats, type Metadata } from '../src/worker/vector_engine.ts';

// --- 配置环境 ---
const MODEL_NAME = 'DMetaSoul/Dmeta-embedding-zh-small';
let DIMENSIONS = 768;

// 让 transformers.js 能够加载本地模型 (此时我们在 FrontEnd)
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = path.resolve(process.cwd(), '../Backend/models'); 

// --- 工具函数与核心算法预制 ---
let vocabMap = new Map<string, number>();
let globalBM25Stats: BM25Stats | null = null;
const REJECTION_THRESHOLD = 0.4;

function fmmTokenize(text: string): string[] {
    const tokens: string[] = [];
    let i = 0;
    while (i < text.length) {
        let matched = false;
        const maxLen = Math.min(10, text.length - i);
        for (let len = maxLen; len > 0; len--) {
            const word = text.substring(i, i + len);
            if (vocabMap.has(word)) {
                tokens.push(word);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            i++;
        }
    }
    return tokens;
}

function splitIntoSemanticChunks(text: string, maxLen = 150): string[] {
    const sentences =
        text.match(/[^\u3002\uff01\uff1f\n]+[\u3002\uff01\uff1f\n]*/g) || [text];
    const chunks: string[] = [];
    let currentChunk = "";

    for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxLen && currentChunk.length > 0) {
            chunks.push(currentChunk);
            currentChunk = "";
        }
        currentChunk += sentence;
    }
    if (currentChunk) chunks.push(currentChunk);
    
    return chunks.slice(0, 4); // 防爆锁
}

// --- 主程序类 ---
class EvaluationRunner {
    private extractor: FeatureExtractionPipeline | null = null;
    private metadataList: Metadata[] = [];
    private vectorMatrix: Int8Array | null = null;
    
    public async loadData() {
        const metadataPath = path.resolve(process.cwd(), 'public/data/frontend_metadata_dmeta_small.json');
        const vectorBinPath = path.resolve(process.cwd(), 'public/data/frontend_vectors_dmeta_small.bin');

        console.log("Loading Metadata...");
        const jsonRaw = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
        this.metadataList = Array.isArray(jsonRaw.data) ? jsonRaw.data : jsonRaw;
        
        const vocabList = jsonRaw.vocab || [];
        vocabMap.clear();
        vocabList.forEach((word: string, index: number) => vocabMap.set(word, index));

        console.log(`Building FMM BM25 Stats on ${this.metadataList.length} documents...`);
        globalBM25Stats = buildBM25Stats(this.metadataList);

        console.log("Loading Vector Bindata...");
        const vectorBuffer = fs.readFileSync(vectorBinPath);
        this.vectorMatrix = new Int8Array(vectorBuffer.buffer, vectorBuffer.byteOffset, vectorBuffer.byteLength);
        
        if (this.metadataList.length > 0 && this.vectorMatrix.length > 0) {
            DIMENSIONS = Math.round(this.vectorMatrix.length / this.metadataList.length);
        }
        
        console.log(`Loading Extractor Model (Node CPU 模拟)...`);
        this.extractor = await pipeline('feature-extraction', MODEL_NAME, {
            dtype: 'q8',
            device: 'cpu'
        });
    }

    public async search(query: string, topK: number = 80) {
        if (!this.extractor || !this.vectorMatrix || !globalBM25Stats) throw new Error("Not Initialized");
        
        const output = await this.extractor(query, { pooling: 'mean', normalize: true, truncation: true, max_length: 512 } as any);
        const queryVector = output.data as Float32Array;
        
        const queryWords = fmmTokenize(query);
        const querySparse = getQuerySparse(queryWords, vocabMap);

        const queryYears = query.match(/20\d{2}/g);
        const queryYearWordIds: number[] = [];
        if (queryYears) {
            queryYears.forEach(y => {
                const id = vocabMap.get(y);
                if (id !== undefined) queryYearWordIds.push(id);
            });
        }

        const matches = searchAndRank({
            queryVector,
            querySparse,
            queryYearWordIds,
            metadata: this.metadataList,
            vectorMatrix: this.vectorMatrix,
            dimensions: DIMENSIONS,
            currentTimestamp: 0,
            bm25Stats: globalBM25Stats
        });

        return matches.slice(0, topK);
    }

    public async fetchOriginalDocuments(otids: string[]) {
        try {
            const response = await fetch(`http://127.0.0.1:8000/api/get_answers`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ otids })
            });
            if (!response.ok) throw new Error(`Backend Request Failed ${response.status}`);
            const result: any = await response.json();
            return result.data || [];
        } catch (e: any) {
            console.error("无法获取原文(请确认后端在 8000 端口运行):", e.message);
            return [];
        }
    }

    public async rerank(query: string, documents: any[]) {
        if (!this.extractor) throw new Error("Not Initialized");

        const output = await this.extractor(query, { pooling: 'mean', normalize: true, truncation: true, max_length: 512 } as any);
        const queryVector = output.data as Float32Array;

        const allChunks: { docIdx: number, text: string }[] = [];
        for (let i = 0; i < documents.length; i++) {
            const docChunks = splitIntoSemanticChunks(documents[i].ot_text || "", 150);
            for (const chunk of docChunks) {
                allChunks.push({ docIdx: i, text: chunk });
            }
        }
        
        const totalChunks = allChunks.length;
        const documentScores = new Float32Array(documents.length).fill(-Infinity);
        const documentBestSentence: string[] = new Array(documents.length).fill("");
        
        // Batch Processing
        const BATCH_SIZE = 8; 
        for (let i = 0; i < totalChunks; i += BATCH_SIZE) {
            const batchChunks = allChunks.slice(i, i + BATCH_SIZE);
            const batchTexts = batchChunks.map(c => (c.text || "").substring(0, 400));
            const batchOutputs = await this.extractor(batchTexts, { pooling: 'mean', normalize: true, truncation: true, max_length: 512 } as any);
            
            const validElements = batchChunks.length * DIMENSIONS;
            const pureData = (batchOutputs.data as Float32Array).subarray(0, validElements);
            
            for (let k = 0; k < batchChunks.length; k++) {
                const chunkVec = pureData.subarray(k * DIMENSIONS, (k + 1) * DIMENSIONS);
                let score = 0;
                for (let d = 0; d < DIMENSIONS; d++) score += queryVector[d] * chunkVec[d];
                
                const docIdx = batchChunks[k].docIdx;
                if (score > documentScores[docIdx]) {
                    documentScores[docIdx] = score;
                    documentBestSentence[docIdx] = batchChunks[k].text;
                }
            }
        }

        const results = [];
        for (let j = 0; j < documents.length; j++) {
            results.push({ 
                ...documents[j], 
                rerankScore: documentScores[j],
                bestSentence: documentBestSentence[j]
            });
        }

        return results.sort((a, b) => b.rerankScore - a.rerankScore);
    }
}

// ==============
//  启动测试
// ==============
async function runEval() {
    const datasetPath = process.argv[2] || path.resolve(process.cwd(), '../Backend/test/test_dataset_v2/test_dataset_standard.json');
    if (!fs.existsSync(datasetPath)) {
        console.error("Dataset not found:", datasetPath);
        process.exit(1);
    }
    
    const dataset = JSON.parse(fs.readFileSync(datasetPath, 'utf-8'));
    console.log(`\n========== RAG E2E Rerank Evaluation Pipeline ==========`);
    console.log(`Dataset: ${datasetPath} (${dataset.length} cases)`);

    const runner = new EvaluationRunner();
    await runner.loadData();
    
    let coarseHits1 = 0;
    let coarseHits5 = 0;
    let fineHits1 = 0;
    let fineHits5 = 0;
    let rejectCount = 0;
    let badRejectCount = 0;
    let goodRejectCount = 0;
    let acceptedCount = 0;
    let acceptedHit1 = 0;
    let acceptedHit5 = 0;
    
    interface BadCase {
        id: number;
        query: string;
        expected: string;
        cRank: number;
        fRank: number;
        rejected?: boolean;
        confidence?: number;
        error?: string;
        found?: string;
    }
    const badCases: BadCase[] = [];
    
    let totalTime = 0;
    
    console.log(`\n\nStarting pipeline evaluation...\n`);
    
    for (let i = 0; i < dataset.length; i++) {
        const item = dataset[i];
        const t0 = performance.now();
        
        try {
            // 第一跳：粗排
            const coarseMatches = await runner.search(item.query, 80);
            const coarseTopIds = coarseMatches.map((r: any) => r.otid || r.parent_otid || r.pkid || r.id);
            const expected = item.expected_otid;
            
            let cRank = coarseTopIds.indexOf(expected) + 1;
            if (cRank === 0) cRank = Infinity;
            
            if (cRank === 1) coarseHits1++;
            if (cRank <= 5) coarseHits5++;
            
            if (cRank === Infinity) {
                badCases.push({
                    id: i + 1,
                    query: item.query,
                    expected,
                    cRank: Infinity,
                    fRank: Infinity,
                    found: coarseTopIds.slice(0, 3).join(',')
                });
                process.stdout.write('❌');
                continue;
            }
            
            // 获取并去重 
            const uniqueOtids = Array.from(new Set(coarseTopIds)).slice(0, 15);
            
            // 第二跳：拉取原文精排
            const originals = await runner.fetchOriginalDocuments(uniqueOtids);
            const fineMatches = await runner.rerank(item.query, originals);
            const fineRankIds = fineMatches.map(r => r.otid || r.pkid || r.id);
            const topConfidence = fineMatches[0]?.confidenceScore ?? fineMatches[0]?.rerankScore ?? -999;
            const rejected = fineMatches.length > 0 && topConfidence < REJECTION_THRESHOLD;
            
            let fRank = fineRankIds.indexOf(expected) + 1;
            if (fRank === 0) fRank = Infinity;
            
            if (fRank === 1) fineHits1++;
            if (fRank <= 5) fineHits5++;
            if (rejected) {
                rejectCount++;
                if (fRank === 1) badRejectCount++;
                else goodRejectCount++;
            } else {
                acceptedCount++;
                if (fRank === 1) acceptedHit1++;
                if (fRank <= 5) acceptedHit5++;
            }
            
            if (fRank !== 1) {
                badCases.push({
                    id: i + 1,
                    query: item.query,
                    expected,
                    cRank,
                    fRank,
                    rejected,
                    confidence: topConfidence
                });
                process.stdout.write('⚠️');
            } else {
                process.stdout.write('✅');
            }
            
        } catch (e: any) {
            badCases.push({
                id: i + 1,
                query: item.query,
                expected: item.expected_otid,
                cRank: -1,
                fRank: -1,
                error: e.message
            });
            process.stdout.write('💥');
        }
        totalTime += (performance.now() - t0);
    }
    
    const reportSummary = `
============= 终结报告 =============
数据集: ${datasetPath}
测评时间: ${new Date().toLocaleString()}
总耗时: ${(totalTime / 1000).toFixed(2)}s 跑完 ${dataset.length} 条测试
粗排 Hit@1 : ${((coarseHits1 / dataset.length) * 100).toFixed(1)}%
粗排 Hit@5 : ${((coarseHits5 / dataset.length) * 100).toFixed(1)}%
精排 Hit@1 : ${((fineHits1 / dataset.length) * 100).toFixed(1)}% (WASM FMM + Semantic Chunking)
精排 Hit@5 : ${((fineHits5 / dataset.length) * 100).toFixed(1)}%
`;

    console.log(reportSummary);

    const rejectSummary = `
============= Reject Metrics =============
Threshold: ${REJECTION_THRESHOLD}
Reject Rate: ${((rejectCount / dataset.length) * 100).toFixed(1)}%
Good Reject Rate: ${((goodRejectCount / dataset.length) * 100).toFixed(1)}%
Bad Reject Rate: ${((badRejectCount / dataset.length) * 100).toFixed(1)}%
Coverage: ${((acceptedCount / dataset.length) * 100).toFixed(1)}%
Accepted Hit@1: ${acceptedCount > 0 ? ((acceptedHit1 / acceptedCount) * 100).toFixed(1) : '0.0'}%
Accepted Hit@5: ${acceptedCount > 0 ? ((acceptedHit5 / acceptedCount) * 100).toFixed(1) : '0.0'}%
`;

    console.log(rejectSummary);

    let reportDetail = reportSummary + `\n\n--- 重点观察的 Bad Case (按严重程度排序) ---\n`;

    if (badCases.length > 0) {
        // 排序逻辑：未命中 (Infinity) 最优先，其次按精排名次倒序
        const sortedBadCases = [...badCases].sort((a, b) => {
            const rA = a.fRank === Infinity ? 999999 : (a.fRank === -1 ? 1000000 : a.fRank);
            const rB = b.fRank === Infinity ? 999999 : (b.fRank === -1 ? 1000000 : b.fRank);
            return rB - rA;
        });

        sortedBadCases.forEach(bc => {
            let line = `[题 ${bc.id}] "${bc.query}"\n`;
            if (bc.error) {
                line += `   💥 报错: ${bc.error}\n`;
            } else if (bc.fRank === Infinity) {
                if (bc.cRank === Infinity) {
                    line += `   🔴 粗排阶段即落选 (Top 80 查无此人)。期望: ${bc.expected}。前三探测: ${bc.found}\n`;
                } else {
                    line += `   🟠 粗排名次: ${bc.cRank} -> 精排后诡异丢失 (可能切片分数过低或未返回原文)\n`;
                }
            } else {
                line += `   ⚠️  名次浮动: 粗排 ${bc.cRank}位 -> 精排重排后 ${bc.fRank}位\n`;
            }
            reportDetail += line + `-------------------------------------------\n`;
        });
        console.log(`已在报告中记录 ${badCases.length} 个 Bad Case。`);
    } else {
        const win = `\n🎉 完美的命中率！FMM 和 自然语义切片 大放异彩！`;
        console.log(win);
        reportDetail += win;
    }

    // 输出到文件
    const resultsDir = path.resolve(process.cwd(), 'scripts/results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

    const fileName = `eval_report_${path.basename(datasetPath, '.json')}_${new Date().getTime()}.txt`;
    const outputPath = path.resolve(resultsDir, fileName);
    fs.writeFileSync(outputPath, reportDetail, 'utf-8');
    
    console.log(`\n✅ 完整测评报告已保存至: ${outputPath}`);
}

runEval().catch(console.error);
