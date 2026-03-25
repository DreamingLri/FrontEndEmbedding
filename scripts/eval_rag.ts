console.error(
    [
        'npm run eval 已停用。',
        '这个历史脚本与当前 searchAndRank 返回结构不兼容，继续使用会产出错误的 0 分报告。',
        '请改用以下脚本：',
        '- npm run eval:compare',
        '- npm run eval:rerank',
        '- npm run eval:ablate',
    ].join('\n'),
);

process.exit(1);
