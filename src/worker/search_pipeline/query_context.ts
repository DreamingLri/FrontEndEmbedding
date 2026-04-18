import { fmmTokenize } from "../fmm_tokenize.ts";
import { buildQueryPlan } from "../query_planner.ts";
import {
    getQuerySparse,
    parseQueryIntent,
    QUERY_SCOPE_SPECIFICITY_TERMS,
} from "../vector_engine.ts";
import {
    getCandidateIndicesForQuery,
    type TopicPartitionIndex,
} from "../topic_partition.ts";
import {
    type PipelinePreset,
    type PipelineTermMaps,
    type SearchPipelineQueryContext,
} from "./types.ts";
import { CANONICAL_PIPELINE_PRESET } from "./presets.ts";

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
}

const QUERY_EXPANSION_RULES: Array<{
    pattern: RegExp;
    terms: string[];
    intentTerms?: string[];
}> = [
    {
        pattern: /现场确认/,
        terms: ["网上确认"],
        intentTerms: ["网上确认"],
    },
];

function buildExpandedQueryWords(
    query: string,
    vocabMap: Map<string, number>,
): string[] {
    const baseWords = fmmTokenize(query, vocabMap);
    const expandedWords = QUERY_EXPANSION_RULES.flatMap((rule) => {
        if (!rule.pattern.test(query)) {
            return [];
        }
        return rule.terms.flatMap((term) => fmmTokenize(term, vocabMap));
    });

    return dedupe([...baseWords, ...expandedWords]);
}

function buildExpandedIntentQuery(query: string): string {
    const expandedTerms = dedupe(
        QUERY_EXPANSION_RULES.flatMap((rule) =>
            rule.pattern.test(query) ? rule.intentTerms || [] : [],
        ),
    );
    if (expandedTerms.length === 0) {
        return query;
    }
    return `${query} ${expandedTerms.join(" ")}`;
}

export function buildPipelineTermMaps(
    vocabMap: Map<string, number>,
): PipelineTermMaps {
    const scopeSpecificityWordIdToTerm = new Map<number, string>();
    QUERY_SCOPE_SPECIFICITY_TERMS.forEach((term) => {
        const wordId = vocabMap.get(term);
        if (wordId !== undefined) {
            scopeSpecificityWordIdToTerm.set(wordId, term);
        }
    });

    return {
        scopeSpecificityWordIdToTerm,
    };
}

export function buildSearchPipelineQueryContext(
    query: string,
    vocabMap: Map<string, number>,
    topicPartitionIndex: TopicPartitionIndex,
    preset: PipelinePreset = CANONICAL_PIPELINE_PRESET,
): SearchPipelineQueryContext {
    // 主链入口先把自然语言问题压成统一查询上下文：
    // 这里一次性完成意图解析、候选裁剪、稀疏词特征和 planner 信号构建，
    // 后续检索与展示阶段都只消费这个结构，不再重复解析原始 query。
    const expandedIntentQuery = preset.retrieval.useQueryExpansion
        ? buildExpandedIntentQuery(query)
        : query;
    const parsedQueryIntent = parseQueryIntent(expandedIntentQuery);
    const queryIntent =
        expandedIntentQuery === query
            ? parsedQueryIntent
            : {
                  ...parsedQueryIntent,
                  rawQuery: query,
              };
    const candidateIndices = preset.retrieval.useTopicPartition
        ? getCandidateIndicesForQuery(queryIntent, topicPartitionIndex)
        : undefined;
    const queryWords = preset.retrieval.useQueryExpansion
        ? buildExpandedQueryWords(query, vocabMap)
        : Array.from(new Set(fmmTokenize(query, vocabMap)));
    const querySparse = getQuerySparse(queryWords, vocabMap);
    const queryYearWordIds = queryIntent.years
        .map(String)
        .map((year) => vocabMap.get(year))
        .filter((item): item is number => item !== undefined);
    // queryPlan 负责抽取“覆盖型/结果型/时间型”等高层策略信号，
    // 既影响检索阶段的 boost，也影响文档抓取与展示重排。
    const queryPlan = buildQueryPlan(query, queryIntent);

    return {
        query,
        queryIntent,
        queryPlan,
        queryWords,
        querySparse,
        queryYearWordIds,
        candidateIndices,
    };
}

