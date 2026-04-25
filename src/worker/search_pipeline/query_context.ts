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

function buildPlannerExpansionWords(
    queryPlan: ReturnType<typeof buildQueryPlan>,
    vocabMap: Map<string, number>,
): string[] {
    const plannerTerms: string[] = [];

    if (
        queryPlan.asksCoverageLike ||
        queryPlan.intentType === "procedure" ||
        queryPlan.intentType === "system_timeline"
    ) {
        plannerTerms.push("流程", "步骤");
    }
    if (queryPlan.asksRequirementLike || queryPlan.asksCoverageLike) {
        plannerTerms.push("条件", "材料", "要求");
    }
    if (
        queryPlan.asksTimeLike ||
        queryPlan.intentType === "system_timeline" ||
        queryPlan.asksCoverageLike
    ) {
        plannerTerms.push("时间", "截止");
    }
    if (
        queryPlan.asksProcedureLike ||
        queryPlan.intentType === "procedure"
    ) {
        plannerTerms.push("报名", "提交", "确认");
    }

    return dedupe(
        plannerTerms.flatMap((term) => fmmTokenize(term, vocabMap)),
    );
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
    // queryPlan 负责抽取“覆盖型/结果型/时间型”等高层策略信号，
    // 既影响检索阶段的 boost，也影响文档抓取与展示重排。
    const queryPlan = buildQueryPlan(query, queryIntent);
    const baseQueryWords = preset.retrieval.useQueryExpansion
        ? buildExpandedQueryWords(query, vocabMap)
        : Array.from(new Set(fmmTokenize(query, vocabMap)));
    const plannerExpansionWords =
        queryPlan.asksCoverageLike || queryPlan.intentType === "procedure"
            ? buildPlannerExpansionWords(queryPlan, vocabMap)
            : [];
    const queryWords = dedupe([...baseQueryWords, ...plannerExpansionWords]);
    const querySparse = getQuerySparse(queryWords, vocabMap);
    const queryYearWordIds = queryIntent.years
        .map(String)
        .map((year) => vocabMap.get(year))
        .filter((item): item is number => item !== undefined);
    const candidateIndices = preset.retrieval.useTopicPartition
        ? getCandidateIndicesForQuery(queryIntent, topicPartitionIndex)
        : undefined;

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

