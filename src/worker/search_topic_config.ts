import searchDomainConfig from "./search_domain_config.json";

export type IntentVectorTableItem = {
    intent_id: string;
    topic_id: string;
    intent_name: string;
    aliases: readonly string[];
    negative_intents: readonly string[];
    related_intents: readonly string[];
};

export type TopicConfigItem = {
    topic_id: string;
    aliases: readonly string[];
    prefer_latest: boolean;
};

type SearchDomainConfig = {
    degree_levels: readonly string[];
    event_type_table: readonly string[];
    latest_query_hints: readonly string[];
    historical_query_hints: readonly string[];
    intent_vector_table: readonly IntentVectorTableItem[];
    topic_configs: readonly TopicConfigItem[];
};

const DOMAIN_CONFIG = searchDomainConfig as SearchDomainConfig;

export const DEGREE_LEVEL_TABLE = DOMAIN_CONFIG.degree_levels;

export const EVENT_TYPE_TABLE = DOMAIN_CONFIG.event_type_table;

export const LATEST_QUERY_HINTS = DOMAIN_CONFIG.latest_query_hints;

export const HISTORICAL_QUERY_HINTS = DOMAIN_CONFIG.historical_query_hints;

export const INTENT_VECTOR_TABLE = DOMAIN_CONFIG.intent_vector_table;

export const TOPIC_CONFIGS = DOMAIN_CONFIG.topic_configs;
