import * as fs from "fs";

export type AnswerQualityCase = {
    id: string;
    query: string;
    query_style_mode?: string;
    query_type?: string;
    query_scope?: string;
    preferred_granularity?: string;
    support_pattern?: string;
    theme_family?: string;
    expected_otid: string;
    expected_kpid?: string;
    expected_support_kpids?: string[];
    source_query?: string;
    source_item_ref?: string;
    source_file?: string;
    source_dataset?: string;
    selected_variant?: string;
    title?: string;
    publish_time?: string;
    notes?: string;
};

export function loadAnswerQualityDataset(filePath: string): {
    cases: AnswerQualityCase[];
    datasetNote: string;
} {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf-8")) as unknown;
    if (!Array.isArray(payload)) {
        throw new Error(`AnswerQuality dataset must be a JSON array: ${filePath}`);
    }

    const cases = payload.map((item, index) => {
        if (!item || typeof item !== "object") {
            throw new Error(`Invalid item at index ${index} in ${filePath}`);
        }
        const row = item as Record<string, unknown>;
        const id = String(row.id || "").trim();
        const query = String(row.query || "").trim();
        const expectedOtid = String(row.expected_otid || "").trim();
        if (!id || !query || !expectedOtid) {
            throw new Error(
                `Invalid answer quality item at index ${index}: missing id/query/expected_otid`,
            );
        }

        return {
            id,
            query,
            query_style_mode: String(row.query_style_mode || "").trim() || undefined,
            query_type: String(row.query_type || "").trim() || undefined,
            query_scope: String(row.query_scope || "").trim() || undefined,
            preferred_granularity:
                String(row.preferred_granularity || "").trim() || undefined,
            support_pattern: String(row.support_pattern || "").trim() || undefined,
            theme_family: String(row.theme_family || "").trim() || undefined,
            expected_otid: expectedOtid,
            expected_kpid: String(row.expected_kpid || "").trim() || undefined,
            expected_support_kpids: Array.isArray(row.expected_support_kpids)
                ? row.expected_support_kpids
                      .filter((value): value is string => typeof value === "string")
                      .map((value) => value.trim())
                      .filter(Boolean)
                : [],
            source_query: String(row.source_query || "").trim() || undefined,
            source_item_ref: String(row.source_item_ref || "").trim() || undefined,
            source_file: String(row.source_file || "").trim() || undefined,
            source_dataset: String(row.source_dataset || "").trim() || undefined,
            selected_variant: String(row.selected_variant || "").trim() || undefined,
            title: String(row.title || "").trim() || undefined,
            publish_time: String(row.publish_time || "").trim() || undefined,
            notes: String(row.notes || "").trim() || undefined,
        } satisfies AnswerQualityCase;
    });

    return {
        cases,
        datasetNote:
            "AnswerQuality-blind 仅评估应答侧：关注回答正确率与潜在误导率，不混入 reject 主线。",
    };
}
