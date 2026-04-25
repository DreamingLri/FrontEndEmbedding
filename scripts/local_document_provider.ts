import * as fs from "fs";
import * as path from "path";

import type {
    PipelineDocumentLoader,
    PipelineDocumentRecord,
} from "../src/worker/search_pipeline.ts";
import {
    resolveBackendArticlesFile,
    resolveBackendKnowledgePointsFile,
} from "./kb_version_paths.ts";

type ApiDocumentLoaderOptions = {
    baseUrl?: string;
    path?: string;
};

const ARTICLE_FILE = resolveBackendArticlesFile();
const KNOWLEDGE_POINT_FILE = resolveBackendKnowledgePointsFile();

let articleMapCache: Map<string, PipelineDocumentRecord> | null = null;

function ensureArticleMap(): Map<string, PipelineDocumentRecord> {
    if (articleMapCache) {
        return articleMapCache;
    }

    const articlePath = path.resolve(process.cwd(), ARTICLE_FILE);
    const knowledgePointPath = path.resolve(process.cwd(), KNOWLEDGE_POINT_FILE);
    const rawArticles = JSON.parse(
        fs.readFileSync(articlePath, "utf-8"),
    ) as Array<{
        otid?: string;
        ot_text?: string;
        ot_title?: string;
        publish_time?: string;
        link?: string;
    }>;
    const rawKnowledgePoints = JSON.parse(
        fs.readFileSync(knowledgePointPath, "utf-8"),
    ) as Array<{
        pkid?: string;
        otid?: string;
        kp_text?: string;
        kp_role_tags?: string[];
    }>;

    const articleMap = new Map<string, PipelineDocumentRecord>();
    rawArticles.forEach((article) => {
        if (!article.otid) {
            return;
        }

        articleMap.set(article.otid, {
            otid: article.otid,
            id: article.otid,
            ot_text: article.ot_text,
            ot_title: article.ot_title,
            publish_time: article.publish_time,
            link: article.link,
            kps: [],
        });
    });

    rawKnowledgePoints.forEach((item) => {
        if (!item.otid) {
            return;
        }

        const article = articleMap.get(item.otid);
        if (!article) {
            return;
        }

        if (!Array.isArray(article.kps)) {
            article.kps = [];
        }

        article.kps.push({
            kpid: item.pkid,
            kp_text: item.kp_text,
            kp_role_tags: item.kp_role_tags,
        });
    });

    articleMapCache = articleMap;
    return articleMap;
}

export function createLocalDocumentLoader(): PipelineDocumentLoader {
    const articleMap = ensureArticleMap();

    return async ({ otids }) =>
        otids
            .map((otid) => {
                const article = articleMap.get(otid);
                if (!article) {
                    return null;
                }

                return {
                    ...article,
                    kps: Array.isArray(article.kps) ? [...article.kps] : [],
                };
            })
            .filter(Boolean) as PipelineDocumentRecord[];
}

function normalizeApiBaseUrl(baseUrl?: string): string {
    const resolvedBaseUrl = (baseUrl || "http://127.0.0.1:8000").trim();
    return resolvedBaseUrl.endsWith("/")
        ? resolvedBaseUrl.slice(0, -1)
        : resolvedBaseUrl;
}

function normalizeApiPath(apiPath?: string): string {
    const resolvedPath = (apiPath || "/api/get_answers").trim();
    if (!resolvedPath) {
        return "/api/get_answers";
    }

    return resolvedPath.startsWith("/") ? resolvedPath : `/${resolvedPath}`;
}

export function createApiDocumentLoader(
    options: ApiDocumentLoaderOptions = {},
): PipelineDocumentLoader {
    const baseUrl = normalizeApiBaseUrl(options.baseUrl);
    const apiPath = normalizeApiPath(options.path);
    const endpoint = `${baseUrl}${apiPath}`;

    return async ({ query, otids }) => {
        if (otids.length === 0) {
            return [];
        }

        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                query,
                otids,
            }),
        });

        if (!response.ok) {
            throw new Error(
                `HTTP document loader failed: ${response.status} ${response.statusText}`,
            );
        }

        const payload = await response.json();
        return Array.isArray(payload?.data)
            ? (payload.data as PipelineDocumentRecord[])
            : [];
    };
}
