import * as fs from 'fs';
import * as path from 'path';
import {
    env,
    pipeline,
    type FeatureExtractionPipeline,
} from '@huggingface/transformers';

import {
    buildBM25Stats,
    type BM25Stats,
    type Metadata,
} from '../src/worker/vector_engine.ts';
import {
    buildTopicPartitionIndex,
    type TopicPartitionIndex,
} from '../src/worker/topic_partition.ts';
import {
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    FRONTEND_METADATA_FILE,
    FRONTEND_MODEL_NAME,
    FRONTEND_VECTOR_FILE,
} from './eval_shared.ts';

export type FrontendEvalEngine = {
    extractor: FeatureExtractionPipeline;
    vocabMap: Map<string, number>;
    metadataList: Metadata[];
    vectorMatrix: Int8Array;
    dimensions: number;
    bm25Stats: BM25Stats;
    topicPartitionIndex: TopicPartitionIndex;
};

let envConfigured = false;

function configureEvalEnv() {
    if (envConfigured) return;

    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = path.resolve(process.cwd(), '../Backend/models');
    envConfigured = true;
}

export async function loadFrontendEvalEngine(): Promise<FrontendEvalEngine> {
    configureEvalEnv();

    const metadataPath = path.resolve(process.cwd(), FRONTEND_METADATA_FILE);
    const vectorPath = path.resolve(process.cwd(), FRONTEND_VECTOR_FILE);
    const metadataPayload = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));

    const metadataList: Metadata[] = Array.isArray(metadataPayload.data)
        ? metadataPayload.data
        : metadataPayload;
    const vocabList: string[] = metadataPayload.vocab || [];
    const vocabMap = new Map<string, number>();
    vocabList.forEach((word, index) => vocabMap.set(word, index));

    const vectorBuffer = fs.readFileSync(vectorPath);
    const vectorMatrix = new Int8Array(
        vectorBuffer.buffer,
        vectorBuffer.byteOffset,
        vectorBuffer.byteLength,
    );

    const dimensions =
        metadataList.length > 0 && vectorMatrix.length > 0
            ? Math.round(vectorMatrix.length / metadataList.length)
            : 768;

    return {
        extractor: await pipeline('feature-extraction', FRONTEND_MODEL_NAME, {
            dtype: 'q8',
            device: 'cpu',
        }),
        vocabMap,
        metadataList,
        vectorMatrix,
        dimensions,
        bm25Stats: buildBM25Stats(metadataList),
        topicPartitionIndex: buildTopicPartitionIndex(metadataList),
    };
}

export async function embedQueries(
    extractor: FeatureExtractionPipeline,
    queries: string[],
    dimensions: number,
    options?: {
        batchSize?: number;
        onProgress?: (done: number, total: number) => void;
    },
): Promise<Float32Array[]> {
    const batchSize = options?.batchSize ?? DEFAULT_QUERY_EMBED_BATCH_SIZE;
    const vectors: Float32Array[] = [];

    for (let start = 0; start < queries.length; start += batchSize) {
        const batch = queries.slice(start, start + batchSize);
        const output = await extractor(batch, {
            pooling: 'mean',
            normalize: true,
            truncation: true,
            max_length: 512,
        } as any);

        const data = output.data as Float32Array;
        for (let index = 0; index < batch.length; index++) {
            const begin = index * dimensions;
            const end = begin + dimensions;
            vectors.push(new Float32Array(data.slice(begin, end)));
        }

        options?.onProgress?.(Math.min(start + batch.length, queries.length), queries.length);
    }

    return vectors;
}
