import * as fs from 'fs';
import * as path from 'path';

export type EvalDatasetCase = {
    query: string;
    expected_otid: string;
    query_type?: string;
    dataset: string;
};

type RawEvalDatasetCase = Omit<EvalDatasetCase, 'dataset'>;

export const FRONTEND_MODEL_NAME = 'DMetaSoul/Dmeta-embedding-zh-small';
export const FRONTEND_METADATA_FILE = 'public/data/frontend_metadata_dmeta_small.json';
export const FRONTEND_VECTOR_FILE = 'public/data/frontend_vectors_dmeta_small.bin';
export const DEFAULT_QUERY_EMBED_BATCH_SIZE = 16;

export function loadDataset(datasetPath: string): EvalDatasetCase[] {
    const absolutePath = path.resolve(process.cwd(), datasetPath);
    const raw = JSON.parse(
        fs.readFileSync(absolutePath, 'utf-8'),
    ) as RawEvalDatasetCase[];
    const dataset = path.basename(datasetPath, '.json');

    return raw.map((item) => ({
        ...item,
        dataset,
    }));
}
