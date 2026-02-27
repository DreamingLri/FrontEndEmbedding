import { useState, useEffect } from 'react'
import { pipeline, env } from '@huggingface/transformers'

// Disable local models
env.allowLocalModels = false;

function App() {
  const [inputText, setInputText] = useState('This is a benchmark test document to evaluate WebGPU, WebGL, and WASM performance.')
  const [backend, setBackend] = useState('wasm')
  const [isComputing, setIsComputing] = useState(false)
  const [progressMsg, setProgressMsg] = useState('')

  const [metrics, setMetrics] = useState<{ loadTime: number, inferTime: number } | null>(null)
  const [embeddings, setEmbeddings] = useState<Float32Array | null>(null)
  const [error, setError] = useState<string | null>(null)

  // WebGPU support check
  const [gpuSupported, setGpuSupported] = useState<boolean | null>(null)
  useEffect(() => {
    setGpuSupported('gpu' in navigator)
  }, [])

  const handleCompute = async () => {
    if (!inputText.trim()) return;

    setIsComputing(true);
    setError(null);
    setMetrics(null);
    setEmbeddings(null);
    setProgressMsg('Loading model (Warmup + Pipeline setup)...');

    try {
      const loadStart = performance.now();

      const extractor = await pipeline('feature-extraction', 'MongoDB/mdbr-leaf-ir', {
        device: backend as any,
      });

      const loadEnd = performance.now();

      // 2. Inference latency
      setProgressMsg('Running inference...');
      const inferStart = performance.now();
      const output = await extractor(inputText, { pooling: 'cls', normalize: true });
      const inferEnd = performance.now();

      setMetrics({
        loadTime: loadEnd - loadStart,
        inferTime: inferEnd - inferStart
      });

      // output.data should be Float32Array
      if (output && output.data) {
        setEmbeddings(new Float32Array(output.data as any));
      } else {
        setEmbeddings(new Float32Array(output.tolist()[0])); // fallback if shape differs
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred during embedding generation.');
    } finally {
      setIsComputing(false);
      setProgressMsg('');
    }
  }

  return (
    <div className="min-h-screen w-full flex items-center justify-center p-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans">
      <div className="max-w-5xl w-full bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-2xl p-6 sm:p-10 flex flex-col gap-8 relative overflow-hidden">
        {/* Glow effect */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1/2 bg-blue-500/10 blur-[120px] pointer-events-none rounded-full" />

        <header className="z-10 border-b border-slate-700/50 pb-6 w-full text-center">
          <h1 className="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 inline-block mb-3">
            EmbeddingBench
          </h1>
          <p className="text-slate-400 text-base max-w-2xl mx-auto">
            Client-side ONNX Runtime Web benchmarks for <code className="bg-slate-900/60 px-2 py-1 rounded-md text-blue-300 font-mono text-sm shadow-inner">mdbr-leaf-ir</code>. Complete privacy, zero server calls.
          </p>
        </header>

        {error && (
          <div className="z-10 bg-red-500/10 border border-red-500/30 text-red-400 px-6 py-4 rounded-xl flex items-center gap-4">
            <svg className="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-sm font-medium">{error}</span>
          </div>
        )}

        {backend === 'webgpu' && gpuSupported === false && (
          <div className="z-10 bg-yellow-500/10 border border-yellow-500/30 text-yellow-500 px-6 py-4 rounded-xl flex items-center gap-4 mb-[-1rem]">
            <svg className="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-sm font-medium">Warning: WebGPU is not supported by your browser. Running this may fallback to CPU or crash.</span>
          </div>
        )}

        <div className="flex flex-col gap-4 z-10 w-full">
          <label htmlFor="inputText" className="text-sm font-semibold text-slate-300 uppercase tracking-widest pl-1">
            Input Query
          </label>
          <textarea
            id="inputText"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isComputing}
            className="w-full bg-slate-900/60 border border-slate-700/50 rounded-xl p-5 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-y min-h-[140px] shadow-inner disabled:opacity-50 disabled:cursor-not-allowed"
            placeholder="Type a sentence to encode (e.g., 'What is the capital of France?')"
          />
        </div>

        <div className="flex flex-col sm:flex-row gap-6 items-end z-10 w-full">
          <div className="flex flex-col gap-4 flex-1 w-full">
            <label htmlFor="backend" className="text-sm font-semibold text-slate-300 uppercase tracking-widest pl-1">
              Execution Backend (EP)
            </label>
            <div className="relative">
              <select
                id="backend"
                value={backend}
                onChange={(e) => setBackend(e.target.value)}
                disabled={isComputing}
                className="w-full bg-slate-900/60 border border-slate-700/50 rounded-xl p-4 pr-10 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none appearance-none cursor-pointer shadow-inner font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <option value="wasm">WASM (SIMD/Threads) - CPU</option>
                <option value="webgl">WebGL - GPU</option>
                <option value="webgpu">WebGPU - GPU (Modern)</option>
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none text-slate-400">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                </svg>
              </div>
            </div>
          </div>
          <button
            onClick={handleCompute}
            disabled={isComputing || !inputText.trim()}
            className="w-full sm:w-auto px-10 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-400 disabled:shadow-none text-white font-bold rounded-xl transition-all shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:-translate-y-0.5 active:translate-y-0 disabled:transform-none flex items-center justify-center gap-3 min-w-[220px]"
          >
            {isComputing ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : 'Generate Embeddings'}
          </button>
        </div>

        {isComputing && progressMsg && (
          <div className="z-10 text-center text-blue-400 font-medium animate-pulse mt-2">{progressMsg}</div>
        )}

        {(metrics || embeddings) && !isComputing && (
          <div className="z-10 mt-6 grid grid-cols-1 md:grid-cols-2 gap-6 w-full pt-6 border-t border-slate-700/50">
            {/* Metrics Panel */}
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-bold text-slate-200">Performance Metrics</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-900/60 border border-slate-700/50 rounded-xl p-5 shadow-inner flex flex-col items-center justify-center">
                  <span className="text-slate-400 text-sm font-medium mb-1">Cold Load Time</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-extrabold text-blue-400">{metrics?.loadTime.toFixed(1)}</span>
                    <span className="text-slate-500 font-medium">ms</span>
                  </div>
                </div>
                <div className="bg-slate-900/60 border border-slate-700/50 rounded-xl p-5 shadow-inner flex flex-col items-center justify-center">
                  <span className="text-slate-400 text-sm font-medium mb-1">Inference Latency</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-extrabold text-purple-400">{metrics?.inferTime.toFixed(1)}</span>
                    <span className="text-slate-500 font-medium">ms</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Vector Panel */}
            <div className="flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold text-slate-200">Output Vector</h2>
                <span className="text-xs font-mono bg-slate-800 text-slate-400 px-2 py-1 rounded">Dim: {embeddings?.length || 0}</span>
              </div>
              <div className="bg-slate-900/60 border border-slate-700/50 rounded-xl p-4 shadow-inner">
                <div className="font-mono text-xs text-slate-400 break-words h-32 overflow-y-auto custom-scrollbar p-1">
                  [{embeddings ? Array.from(embeddings.slice(0, 50)).map(v => v.toFixed(4)).join(', ') + (embeddings.length > 50 ? ', ...' : '') : ''}]
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5); 
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(71, 85, 105, 0.8); 
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(100, 116, 139, 1); 
        }
      `}</style>
    </div>
  )
}

export default App
