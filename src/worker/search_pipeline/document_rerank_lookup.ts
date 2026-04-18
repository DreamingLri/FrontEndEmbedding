import type {
    PipelineCoarseMatch,
    PipelineDocumentRecord,
} from "./types.ts";

export type SearchOutputMatchRecord = {
    otid: string;
    score: number;
    best_kpid?: string;
};

export function resolveDynamicFetchLimit(
    baseLimit: number,
    delta: number,
    maxLimit: number,
): number {
    return Math.max(baseLimit, Math.min(baseLimit + delta, maxLimit));
}

export function buildPipelineDocumentLookup(
    documents: PipelineDocumentRecord[],
): Map<string, PipelineDocumentRecord> {
    return new Map(
        documents.map((document) => [document.otid || document.id || "", document]),
    );
}

export function mergeCoarseMatchesWithDocumentLookup(
    documentLookup: Map<string, PipelineDocumentRecord>,
    coarseMatches: PipelineCoarseMatch[],
): PipelineDocumentRecord[] {
    return coarseMatches
        .map((match) => {
            const document = documentLookup.get(match.otid);
            if (!document) {
                return null;
            }

            return {
                ...document,
                score: match.score ?? document.score,
                coarseScore: match.score ?? document.coarseScore ?? document.score,
                displayScore: match.score ?? document.displayScore ?? document.score,
                best_kpid: match.best_kpid ?? document.best_kpid,
            };
        })
        .filter(Boolean) as PipelineDocumentRecord[];
}

export function selectLimitedCoarseMatches(
    matches: SearchOutputMatchRecord[],
    limit: number,
): PipelineCoarseMatch[] {
    return matches.slice(0, limit).map((match) => ({
        otid: match.otid,
        score: match.score,
        best_kpid: match.best_kpid,
    }));
}

export function collectUniqueFetchOtids(
    ...coarseMatchGroups: PipelineCoarseMatch[][]
): string[] {
    const otids: string[] = [];
    const seen = new Set<string>();

    coarseMatchGroups.forEach((group) => {
        group.forEach((match) => {
            if (!match.otid || seen.has(match.otid)) {
                return;
            }
            seen.add(match.otid);
            otids.push(match.otid);
        });
    });

    return otids;
}
