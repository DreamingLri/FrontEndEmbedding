import * as fs from "fs";
import * as path from "path";

export type AnswerRejectBehavior = "answer" | "reject";
export type AnswerSubtype =
    | "legacy_direct_answer"
    | "legacy_clarify"
    | "legacy_route_to_entry"
    | null;
export type RejectSubtype = "kb_absent" | "hard_reject" | null;
export type PairRole = "positive" | "negative" | null;

export type AnswerRejectCase = {
    id: string;
    query: string;
    query_type?: string;
    expected_behavior: AnswerRejectBehavior;
    expected_otid?: string | null;
    expected_support_kpids?: string[];
    answer_subtype?: AnswerSubtype;
    reject_subtype?: RejectSubtype;
    pair_id?: string | null;
    pair_role?: PairRole;
    theme_family?: string;
    challenge_tags?: string[];
    source_family?: string;
    source_file?: string;
    source_item_ref?: string;
    notes?: string;
};

type LegacyRouteExpectedAction = "clarify" | "route_to_entry" | "reject";

type LegacyRouteCase = {
    id: string;
    query: string;
    query_type?: string;
    expected_action: LegacyRouteExpectedAction;
    source_otid?: string;
    theme_family?: string;
    challenge_tags?: string[];
    source_family?: string;
    notes?: string;
};

type LegacyBehaviorCase = {
    id: string;
    query: string;
    query_type?: string;
    expected_behavior: "reject" | "direct_answer";
    expected_otid?: string;
    expected_support_kpids?: string[];
    pair_id?: string;
    pair_role?: "positive" | "negative";
    theme_family?: string;
    challenge_tags?: string[];
    source_family?: string;
    source_file?: string;
    source_item_ref?: string;
    notes?: string;
};

type AnswerRejectManifestSourceKind =
    | "route_answer"
    | "route_hard_reject"
    | "kb_absent_mainline"
    | "pair_control_mainline";

type AnswerRejectManifestSource = {
    file: string;
    source_kind: AnswerRejectManifestSourceKind;
};

type AnswerRejectManifest = {
    dataset_type: "answer_reject_manifest_v1";
    dataset_name?: string;
    notes?: string;
    sources: AnswerRejectManifestSource[];
};

function toForwardSlashes(input: string): string {
    return input.replace(/\\/g, "/");
}

function toRepoRelativePath(filePath: string): string {
    return toForwardSlashes(path.relative(process.cwd(), filePath));
}

function readJsonFile<T>(filePath: string): T {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function normalizeStringArray(value: unknown): string[] {
    return Array.isArray(value)
        ? value.filter((item): item is string => typeof item === "string")
        : [];
}

function toRouteAnswerSubtype(
    action: LegacyRouteExpectedAction,
): Exclude<AnswerSubtype, "legacy_direct_answer" | null> {
    return action === "clarify"
        ? "legacy_clarify"
        : "legacy_route_to_entry";
}

function toRouteAnswerCases(
    items: LegacyRouteCase[],
    sourceFile: string,
): AnswerRejectCase[] {
    return items
        .filter((item) => item.expected_action !== "reject")
        .map((item) => ({
            id: item.id,
            query: item.query,
            query_type: item.query_type || "standard",
            expected_behavior: "answer",
            expected_otid: item.source_otid || null,
            expected_support_kpids: [],
            answer_subtype: toRouteAnswerSubtype(item.expected_action),
            reject_subtype: null,
            pair_id: null,
            pair_role: null,
            theme_family: item.theme_family || "",
            challenge_tags: normalizeStringArray(item.challenge_tags),
            source_family: item.source_family || "route_or_clarify_answer",
            source_file: toRepoRelativePath(sourceFile),
            source_item_ref: item.id,
            notes: item.notes || "",
        }));
}

function toRouteHardRejectCases(
    items: LegacyRouteCase[],
    sourceFile: string,
): AnswerRejectCase[] {
    return items
        .filter((item) => item.expected_action === "reject")
        .map((item) => ({
            id: item.id,
            query: item.query,
            query_type: item.query_type || "standard",
            expected_behavior: "reject",
            expected_otid: null,
            expected_support_kpids: [],
            answer_subtype: null,
            reject_subtype: "hard_reject",
            pair_id: null,
            pair_role: null,
            theme_family: item.theme_family || "",
            challenge_tags: normalizeStringArray(item.challenge_tags),
            source_family: item.source_family || "route_or_clarify_hard_reject",
            source_file: toRepoRelativePath(sourceFile),
            source_item_ref: item.id,
            notes: item.notes || "",
        }));
}

function toBehaviorCase(
    item: LegacyBehaviorCase,
    fallbackSourceFile: string,
    pairMode: boolean,
): AnswerRejectCase {
    const isReject = item.expected_behavior === "reject";
    const pairRole =
        item.pair_role === "positive" || item.pair_role === "negative"
            ? item.pair_role
            : null;

    return {
        id: item.id,
        query: item.query,
        query_type: item.query_type || "standard",
        expected_behavior: isReject ? "reject" : "answer",
        expected_otid: isReject ? null : item.expected_otid || null,
        expected_support_kpids: normalizeStringArray(item.expected_support_kpids),
        answer_subtype: isReject ? null : "legacy_direct_answer",
        reject_subtype: isReject ? "kb_absent" : null,
        pair_id: pairMode ? item.pair_id || null : null,
        pair_role: pairMode ? pairRole : null,
        theme_family: item.theme_family || "",
        challenge_tags: normalizeStringArray(item.challenge_tags),
        source_family: item.source_family || "answer_reject_mainline",
        source_file: item.source_file || toRepoRelativePath(fallbackSourceFile),
        source_item_ref: item.source_item_ref || item.id,
        notes: item.notes || "",
    };
}

function toKbAbsentMainlineCases(
    items: LegacyBehaviorCase[],
    sourceFile: string,
): AnswerRejectCase[] {
    return items.map((item) => toBehaviorCase(item, sourceFile, false));
}

function toPairControlMainlineCases(
    items: LegacyBehaviorCase[],
    sourceFile: string,
): AnswerRejectCase[] {
    return items.map((item) => toBehaviorCase(item, sourceFile, true));
}

function isAnswerRejectManifest(
    value: unknown,
): value is AnswerRejectManifest {
    return Boolean(
        value &&
            !Array.isArray(value) &&
            typeof value === "object" &&
            (value as AnswerRejectManifest).dataset_type ===
                "answer_reject_manifest_v1" &&
            Array.isArray((value as AnswerRejectManifest).sources),
    );
}

function loadManifestSourceCases(
    manifestFile: string,
    source: AnswerRejectManifestSource,
): AnswerRejectCase[] {
    const resolvedSourceFile = path.resolve(path.dirname(manifestFile), source.file);

    if (source.source_kind === "route_answer") {
        return toRouteAnswerCases(
            readJsonFile<LegacyRouteCase[]>(resolvedSourceFile),
            resolvedSourceFile,
        );
    }
    if (source.source_kind === "route_hard_reject") {
        return toRouteHardRejectCases(
            readJsonFile<LegacyRouteCase[]>(resolvedSourceFile),
            resolvedSourceFile,
        );
    }
    if (source.source_kind === "kb_absent_mainline") {
        return toKbAbsentMainlineCases(
            readJsonFile<LegacyBehaviorCase[]>(resolvedSourceFile),
            resolvedSourceFile,
        );
    }
    return toPairControlMainlineCases(
        readJsonFile<LegacyBehaviorCase[]>(resolvedSourceFile),
        resolvedSourceFile,
    );
}

export function loadAnswerRejectDataset(datasetFile: string): {
    cases: AnswerRejectCase[];
    datasetNote: string | null;
} {
    const payload = readJsonFile<AnswerRejectCase[] | AnswerRejectManifest>(datasetFile);

    if (Array.isArray(payload)) {
        return {
            cases: payload,
            datasetNote: null,
        };
    }

    if (!isAnswerRejectManifest(payload)) {
        throw new Error(
            `Unsupported answer_reject dataset format: ${datasetFile}`,
        );
    }

    return {
        cases: payload.sources.flatMap((source) =>
            loadManifestSourceCases(datasetFile, source),
        ),
        datasetNote: payload.notes || null,
    };
}
