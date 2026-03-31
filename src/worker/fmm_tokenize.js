"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.fmmTokenize = fmmTokenize;
function fmmTokenize(text, vocabMap) {
    var tokens = [];
    var index = 0;
    while (index < text.length) {
        var matched = false;
        var maxLen = Math.min(10, text.length - index);
        for (var len = maxLen; len > 0; len--) {
            var word = text.substring(index, index + len);
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
