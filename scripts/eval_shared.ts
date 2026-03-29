import * as fs from 'fs';
import * as path from 'path';

export type EvalDatasetCase = {
    query: string;
    expected_otid: string;
    expected_kpid?: string;
    support_kpids?: string[];
    query_type?: string;
    query_scope?: string;
    preferred_granularity?: string;
    support_pattern?: string;
    granularity_sensitive?: boolean;
    theme_family?: string;
    source_dataset?: string;
    source_seed_id?: string;
    challenge_tags?: string[];
    notes?: string;
    dataset: string;
};

type RawEvalDatasetCase = Omit<EvalDatasetCase, 'dataset'>;

export type EvalDatasetSource = {
    path: string;
    datasetLabel?: string;
    queryTypes?: string[];
};

export type EvalDatasetConfig = {
    datasetVersion: string;
    datasetMode: 'split' | 'single_file';
    datasetKey: string;
    tuneSources: EvalDatasetSource[];
    holdoutSources: EvalDatasetSource[];
    allSources: EvalDatasetSource[];
};

export const FRONTEND_MODEL_NAME = 'DMetaSoul/Dmeta-embedding-zh-small';
export const FRONTEND_METADATA_FILE = 'public/data/frontend_metadata_dmeta_small.json';
export const FRONTEND_VECTOR_FILE = 'public/data/frontend_vectors_dmeta_small.bin';
export const DEFAULT_QUERY_EMBED_BATCH_SIZE = 16;

export function loadDataset(
    datasetPath: string,
    options?: {
        datasetLabel?: string;
        queryTypes?: string[];
    },
): EvalDatasetCase[] {
    const absolutePath = path.resolve(process.cwd(), datasetPath);
    const raw = JSON.parse(
        fs.readFileSync(absolutePath, 'utf-8'),
    ) as RawEvalDatasetCase[];
    const dataset = options?.datasetLabel || path.basename(datasetPath, '.json');
    const queryTypes = options?.queryTypes;
    const filtered = queryTypes?.length
        ? raw.filter((item) => item.query_type && queryTypes.includes(item.query_type))
        : raw;

    return filtered.map((item) => ({
        ...item,
        dataset,
    }));
}

export function loadDatasetSources(
    sources: readonly EvalDatasetSource[],
    options?: {
        limitPerSource?: number;
    },
): EvalDatasetCase[] {
    return sources.flatMap((source) => {
        const cases = loadDataset(source.path, {
            datasetLabel: source.datasetLabel,
            queryTypes: source.queryTypes,
        });

        if (
            Number.isFinite(options?.limitPerSource) &&
            (options?.limitPerSource || 0) > 0
        ) {
            return cases.slice(0, options?.limitPerSource);
        }

        return cases;
    });
}

function buildSplitSources(datasetDir: string): EvalDatasetConfig {
    const tuneSources: EvalDatasetSource[] = [
        { path: `${datasetDir}/test_dataset_standard.json` },
        { path: `${datasetDir}/test_dataset_short_keyword.json` },
    ];
    const holdoutSources: EvalDatasetSource[] = [
        { path: `${datasetDir}/test_dataset_situational.json` },
    ];

    return {
        datasetVersion: path.basename(datasetDir).replace(/^test_dataset_/, ''),
        datasetMode: 'split',
        datasetKey: path.basename(datasetDir),
        tuneSources,
        holdoutSources,
        allSources: [...tuneSources, ...holdoutSources],
    };
}

function buildSingleFileSources(
    datasetVersion: string,
    datasetFile: string,
): EvalDatasetConfig {
    const datasetKey = path.basename(datasetFile, '.json');
    const tuneSources: EvalDatasetSource[] = [
        {
            path: datasetFile,
            datasetLabel: `${datasetKey}_standard`,
            queryTypes: ['standard'],
        },
        {
            path: datasetFile,
            datasetLabel: `${datasetKey}_short_keyword`,
            queryTypes: ['short_keyword'],
        },
    ];
    const holdoutSources: EvalDatasetSource[] = [
        {
            path: datasetFile,
            datasetLabel: `${datasetKey}_situational`,
            queryTypes: ['situational'],
        },
    ];

    return {
        datasetVersion,
        datasetMode: 'single_file',
        datasetKey,
        tuneSources,
        holdoutSources,
        allSources: [...tuneSources, ...holdoutSources],
    };
}

export function resolveEvalDatasetConfig(options?: {
    datasetVersion?: string;
    datasetFile?: string;
    singleFileAsAll?: boolean;
}): EvalDatasetConfig {
    const datasetVersion = options?.datasetVersion || 'v2';
    const explicitDatasetFile = options?.datasetFile;
    const singleFileAsAll = options?.singleFileAsAll || false;

    if (explicitDatasetFile) {
        if (singleFileAsAll) {
            const datasetKey = path.basename(explicitDatasetFile, '.json');
            const tuneSources: EvalDatasetSource[] = [
                {
                    path: explicitDatasetFile,
                    datasetLabel: datasetKey,
                },
            ];
            return {
                datasetVersion,
                datasetMode: 'single_file',
                datasetKey,
                tuneSources,
                holdoutSources: [],
                allSources: [...tuneSources],
            };
        }
        return buildSingleFileSources(datasetVersion, explicitDatasetFile);
    }

    const datasetDir = `../Backend/test/test_dataset_${datasetVersion}`;
    const splitPaths = [
        `${datasetDir}/test_dataset_standard.json`,
        `${datasetDir}/test_dataset_short_keyword.json`,
        `${datasetDir}/test_dataset_situational.json`,
    ];
    const hasSplitDatasets = splitPaths.every((item) =>
        fs.existsSync(path.resolve(process.cwd(), item)),
    );

    if (hasSplitDatasets) {
        return buildSplitSources(datasetDir);
    }

    const singleFilePath = `${datasetDir}/test_dataset_${datasetVersion}.json`;
    if (fs.existsSync(path.resolve(process.cwd(), singleFilePath))) {
        return buildSingleFileSources(datasetVersion, singleFilePath);
    }

    throw new Error(
        `Unable to resolve evaluation dataset for version "${datasetVersion}". Checked split files under ${datasetDir} and single file ${singleFilePath}.`,
    );
}
