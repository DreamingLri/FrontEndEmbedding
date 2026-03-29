import * as fs from "fs";
import * as path from "path";

import type {
    PipelineDocumentLoader,
    PipelineDocumentRecord,
} from "../src/worker/search_pipeline.ts";

const ARTICLE_FILE = "../Backend/data/embeddings_v2/backend_articles.json";
const KNOWLEDGE_POINT_FILE =
    "../Backend/data/embeddings_v2/backend_knowledge_points.json";

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
