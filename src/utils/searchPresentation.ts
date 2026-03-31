type SearchPreviewable = {
  bestSentence?: string
  bestPoint?: string
  best_kpid?: string
  kps?: Array<{ kpid?: string; kp_text?: string }>
  score?: number
  coarseScore?: number
  displayScore?: number
  rerankScore?: number
  confidenceScore?: number
  snippetScore?: number
}

export function getPreviewText(res: SearchPreviewable): string {
  if (res.bestSentence) return res.bestSentence
  if (res.bestPoint) return res.bestPoint

  if (res.best_kpid && Array.isArray(res.kps)) {
    const hitKp = res.kps.find((kp) => kp.kpid === res.best_kpid)
    if (hitKp?.kp_text) {
      return hitKp.kp_text
    }
  }

  return ''
}

export function isOriginalSnippet(
  res: Pick<SearchPreviewable, 'snippetScore' | 'bestSentence'>,
  threshold = 0.4,
): boolean {
  return (res.snippetScore ?? -999) > threshold && !!res.bestSentence
}

export function getDisplayScore(res: SearchPreviewable): number {
  return res.displayScore ?? res.coarseScore ?? res.confidenceScore ?? res.rerankScore ?? res.score ?? 0
}

export function formatRetrievalScore(score: number): string {
  return `Score ${Number(score || 0).toFixed(2)}`
}

export function formatPercent(score: number): string {
  return `${((score || 0) * 100).toFixed(1)}%`
}
