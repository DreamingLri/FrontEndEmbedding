import * as fs from "fs";
import * as path from "path";

export const DEFAULT_MAIN_DB_VERSION = "main_v2_plus";
export const MAIN_DB_VERSION =
    process.env.SUASK_MAIN_DB_VERSION || DEFAULT_MAIN_DB_VERSION;

function resolveExistingPath(
    candidates: string[],
    fallbackToFirst = true,
): string {
    for (const candidate of candidates) {
        const absolutePath = path.resolve(process.cwd(), candidate);
        if (fs.existsSync(absolutePath)) {
            return candidate;
        }
    }
    return fallbackToFirst && candidates.length > 0 ? candidates[0] : "";
}

export function resolveFrontendMetadataFile(): string {
    const candidates = [
        `public/data/frontend_metadata_dmeta_small_${MAIN_DB_VERSION}.json`,
    ];
    if (MAIN_DB_VERSION === DEFAULT_MAIN_DB_VERSION) {
        candidates.push("public/data/frontend_metadata_dmeta_small.json");
    }
    return resolveExistingPath(candidates);
}

export function resolveFrontendVectorFile(): string {
    const candidates = [
        `public/data/frontend_vectors_dmeta_small_${MAIN_DB_VERSION}.bin`,
    ];
    if (MAIN_DB_VERSION === DEFAULT_MAIN_DB_VERSION) {
        candidates.push("public/data/frontend_vectors_dmeta_small.bin");
    }
    return resolveExistingPath(candidates);
}

export function resolveBackendArticlesFile(): string {
    const candidates = [
        `../Backend/data/embeddings_v2/backend_articles_${MAIN_DB_VERSION}.json`,
    ];
    if (MAIN_DB_VERSION === DEFAULT_MAIN_DB_VERSION) {
        candidates.push("../Backend/data/embeddings_v2/backend_articles.json");
    }
    return resolveExistingPath(candidates);
}

export function resolveBackendKnowledgePointsFile(): string {
    const candidates = [
        `../Backend/data/embeddings_v2/backend_knowledge_points_${MAIN_DB_VERSION}.json`,
    ];
    if (MAIN_DB_VERSION === DEFAULT_MAIN_DB_VERSION) {
        candidates.push("../Backend/data/embeddings_v2/backend_knowledge_points.json");
    }
    return resolveExistingPath(candidates);
}
