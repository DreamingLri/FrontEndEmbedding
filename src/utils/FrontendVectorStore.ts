/**
 * Frontend Vector Search Core (Binary & IndexedDB Edition)
 * Loads the ultra-compact raw binary matrix and JSON metadata, and caches them using IndexedDB.
 * Compatible with TypeScript, Vue, React, or Vanilla JS.
 */

// We will use standard IndexedDB. To make it Promise-based without external dependencies, we write a tiny wrapper.
const DB_NAME = 'SuAsk_VectorDB';
const STORE_NAME = 'cache_store';

class IDBHelper {
    static async openDB(): Promise<IDBDatabase> {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, 1);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            request.onupgradeneeded = (e: any) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    db.createObjectStore(STORE_NAME);
                }
            };
        });
    }

    static async get(key: string): Promise<any> {
        const db = await this.openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readonly');
            const store = tx.objectStore(STORE_NAME);
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    static async set(key: string, value: any): Promise<void> {
        const db = await this.openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readwrite');
            const store = tx.objectStore(STORE_NAME);
            const request = store.put(value, key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
}

export interface MatchedResult {
    id: string;
    parent_otid?: string;
    type: 'Q' | 'KP' | 'OT';
    score: number;
}

export class FrontendVectorStore {
    private vectorIndex: any[] = [];
    private vectorDim: number = 512;
    public isReady: boolean = false;

    /**
     * Initialize the local database. Tries to load from IndexedDB first.
     * If not found, downloads from network, parses, and caches into IndexedDB.
     */
    async initDatabase(modelId: 'gte_small' | 'dmeta_small' = 'gte_small') {
        const metaUrl = `/embeddings/frontend_metadata_${modelId}.json`;
        const binUrl = `/embeddings/frontend_vectors_${modelId}.bin`;
        
        const cacheKeyMeta = `meta_v2_${modelId}`;
        const cacheKeyBin = `bin_v2_${modelId}`;

        console.group(`[VectorStore] Initializing Model: ${modelId}`);
        const startTime = performance.now();

        try {
            console.log(`[IndexedDB] Checking for cached assets: ${cacheKeyMeta}, ${cacheKeyBin}`);
            let rawPayload = await IDBHelper.get(cacheKeyMeta);
            let binBuffer = await IDBHelper.get(cacheKeyBin);

            if (!rawPayload || !binBuffer) {
                console.warn(`[Cache] Miss! Progressing to network fetch...`);
                console.log(`[Network] Fetching: ${metaUrl} and ${binUrl}`);
                
                // Fetch in parallel for speed
                const fetchStart = performance.now();
                const [metaResponse, binResponse] = await Promise.all([
                    fetch(metaUrl),
                    fetch(binUrl)
                ]);

                if (!metaResponse.ok) {
                    throw new Error(`[HTTP Error] Meta JSON fetch failed: ${metaResponse.status} ${metaResponse.statusText} at ${metaUrl}`);
                }
                if (!binResponse.ok) {
                    throw new Error(`[HTTP Error] Binary vectors fetch failed: ${binResponse.status} ${binResponse.statusText} at ${binUrl}`);
                }

                console.log(`[Network] Fetch successful in ${(performance.now() - fetchStart).toFixed(1)}ms`);

                rawPayload = await metaResponse.json();
                binBuffer = await binResponse.arrayBuffer();

                // Validation
                if (!Array.isArray(rawPayload)) {
                    throw new Error(`[Data Integrity] Invalid Meta JSON format: Expected flat array, got ${typeof rawPayload}`);
                }
                if (binBuffer.byteLength === 0) {
                    throw new Error(`[Data Integrity] Downloaded binary buffer is empty.`);
                }

                // Fire and forget caching
                console.log(`[IndexedDB] Archiving assets to local storage...`);
                Promise.all([
                    IDBHelper.set(cacheKeyMeta, rawPayload),
                    IDBHelper.set(cacheKeyBin, binBuffer)
                ]).then(() => {
                    console.log(`[IndexedDB] Cache updated successfully.`);
                }).catch(e => {
                    console.error("[IndexedDB] Storage error:", e);
                });
            } else {
                console.log(`[Cache] Hit! Loaded directly from IndexedDB in ${(performance.now() - startTime).toFixed(1)}ms`);
            }

            this._hydrateMemory(rawPayload, binBuffer);
            console.log(`[VectorStore] Initialization complete. Total time: ${(performance.now() - startTime).toFixed(1)}ms`);

        } catch (error) {
            console.error("[VectorStore] CRITICAL ERROR during initialization:");
            console.error(error);
            this.isReady = false;
            throw error;
        } finally {
            console.groupEnd();
        }
    }

    /**
     * Map the binary flat array and metadata together into workable objects
     */
    private _hydrateMemory(rawPayload: any[], binBuffer: ArrayBuffer) {
        // We create an Int8 view over the entire buffer, extremely fast.
        const giantInt8View = new Int8Array(binBuffer);

        // Calculate dimension based on buffer size and metadata length
        if (rawPayload.length > 0) {
            this.vectorDim = Math.floor(giantInt8View.length / rawPayload.length);
        } else {
            this.vectorDim = 512;
        }

        // Unpack the compressed array format
        this.vectorIndex = rawPayload.map((meta: any, idx: number) => {
            const scale = meta.scale || 1.0;

            // Decode this document's chunk of the binary buffer back to Float32 once
            const offset = idx * this.vectorDim;
            const myInt8Chunk = giantInt8View.subarray(offset, offset + this.vectorDim);

            const float32Arr = new Float32Array(this.vectorDim);
            for (let i = 0; i < this.vectorDim; i++) {
                float32Arr[i] = myInt8Chunk[i] * scale;
            }

            return {
                id: meta.id,
                type: meta.type,
                parent_otid: meta.parent_otid,
                floatVector: float32Arr
            };
        });

        this.isReady = true;
        console.log(`✅ Ready! ${this.vectorIndex.length} items embedded. Computed Dim: ${this.vectorDim}.`);
    }

    /**
     * 3. Calculate Cosine Similarity
     */
    private _cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        // JS engine will JIT compile this loop into hardware registers, blazing fast
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * 4. Perform Dense Search and return Top K Result objects
     */
    async searchIds(queryDenseVector: Float32Array | number[], topK: number = 5): Promise<MatchedResult[]> {
        if (!this.isReady) throw new Error("Index not ready yet!");

        // Ensure Float32Array
        let qVec = queryDenseVector;
        if (!(qVec instanceof Float32Array)) {
            qVec = new Float32Array(queryDenseVector);
        }

        const results = this.vectorIndex.map(item => {
            const denseScore = this._cosineSimilarity(qVec as Float32Array, item.floatVector);
            return { 
                id: item.id, 
                type: item.type,
                parent_otid: item.parent_otid,
                score: denseScore 
            };
        });

        // Sort descending
        results.sort((a, b) => b.score - a.score);

        return results.slice(0, topK);
    }
}
