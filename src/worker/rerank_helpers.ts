export function splitIntoSemanticChunks(
    text: string,
    maxLen = 150,
    maxChunks?: number,
): string[] {
    const sentences =
        text.match(/[^\u3002\uff01\uff1f\n]+[\u3002\uff01\uff1f\n]*/g) || [text];
    const chunks: string[] = [];
    let currentChunk = "";

    for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxLen && currentChunk.length > 0) {
            chunks.push(currentChunk);
            currentChunk = "";
        }
        currentChunk += sentence;
    }
    if (currentChunk) chunks.push(currentChunk);

    return typeof maxChunks === "number" ? chunks.slice(0, maxChunks) : chunks;
}

export function normalizeSnippetScore(rawScore: number): number {
    const normalized = (rawScore + 1) / 2;
    return Math.min(1, Math.max(0, normalized));
}

export function normalizeMinMax(values: number[]): number[] {
    if (values.length === 0) return [];

    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    if (Math.abs(maxValue - minValue) < 1e-9) {
        return values.map(() => 1);
    }

    return values.map((value) => (value - minValue) / (maxValue - minValue));
}
