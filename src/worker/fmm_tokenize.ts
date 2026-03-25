export function fmmTokenize(
    text: string,
    vocabMap: ReadonlyMap<string, number>,
): string[] {
    const tokens: string[] = [];
    let index = 0;

    while (index < text.length) {
        let matched = false;
        const maxLen = Math.min(10, text.length - index);

        for (let len = maxLen; len > 0; len--) {
            const word = text.substring(index, index + len);
            if (vocabMap.has(word)) {
                tokens.push(word);
                index += len;
                matched = true;
                break;
            }
        }

        if (!matched) {
            index += 1;
        }
    }

    return tokens;
}
