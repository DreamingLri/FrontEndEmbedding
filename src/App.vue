<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { pipeline, env } from '@huggingface/transformers'

// Disable local models
env.allowLocalModels = false

const model = ref('MongoDB/mdbr-leaf-ir')
const inputText = ref('This is a benchmark test document to evaluate WebGPU, WebGL, and WASM performance.')
const backend = ref('wasm')
const isComputing = ref(false)
const progressMsg = ref('')

const metrics = ref<{ loadTime: number; inferTime: number } | null>(null)
const embeddings = ref<Float32Array | null>(null)
const error = ref<string | null>(null)

// WebGPU support check
const gpuSupported = ref<boolean | null>(null)
onMounted(() => {
  gpuSupported.value = 'gpu' in navigator
})

const embeddingPreview = () => {
  if (!embeddings.value) return '[]'
  const arr = Array.from(embeddings.value.slice(0, 50)).map(v => v.toFixed(4)).join(', ')
  return `[${arr}${embeddings.value.length > 50 ? ', ...' : ''}]`
}

const handleCompute = async () => {
  if (!inputText.value.trim()) return

  isComputing.value = true
  error.value = null
  metrics.value = null
  embeddings.value = null
  progressMsg.value = 'Loading model (Warmup + Pipeline setup)...'

  try {
    const loadStart = performance.now()

    const extractor = await pipeline('feature-extraction', model.value, {
      device: backend.value === 'webgpu' ? 'webgpu' : 'wasm',
    })

    const loadEnd = performance.now()

    progressMsg.value = 'Running inference...'
    const inferStart = performance.now()
    const output = await extractor(inputText.value, { pooling: 'cls', normalize: true })
    const inferEnd = performance.now()

    metrics.value = {
      loadTime: loadEnd - loadStart,
      inferTime: inferEnd - inferStart,
    }

    if (output && output.data) {
      embeddings.value = new Float32Array(output.data as any)
    } else {
      embeddings.value = new Float32Array(output.tolist()[0])
    }
  } catch (err: any) {
    console.error(err)
    error.value = err.message || 'An error occurred during embedding generation.'
  } finally {
    isComputing.value = false
    progressMsg.value = ''
  }
}
</script>

<template>
  <div
    class="min-h-screen w-full flex items-center justify-center p-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans"
  >
    <div
      class="max-w-5xl w-full bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-2xl p-6 sm:p-10 flex flex-col gap-8 relative overflow-hidden"
    >
      <div
        class="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1/2 bg-blue-500/10 blur-[120px] pointer-events-none rounded-full"
      />

      <!-- Header -->
      <header class="z-10 border-b border-slate-700/50 pb-6 w-full text-center">
        <h1
          class="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 inline-block mb-3"
        >
          EmbeddingBench
        </h1>
        <p class="text-slate-400 text-base max-w-2xl mx-auto">
          Client-side ONNX Runtime Web benchmarks for
          <code class="bg-slate-900/60 px-2 py-1 rounded-md text-blue-300 font-mono text-sm shadow-inner">{{
            model
          }}</code
          >.
        </p>
      </header>

      <!-- Error alert -->
      <div
        v-if="error"
        class="z-10 bg-red-500/10 border border-red-500/30 text-red-400 px-6 py-4 rounded-xl flex items-center gap-4"
      >
        <svg class="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <span class="text-sm font-medium">{{ error }}</span>
      </div>

      <!-- WebGPU warning -->
      <div
        v-if="backend === 'webgpu' && gpuSupported === false"
        class="z-10 bg-yellow-500/10 border border-yellow-500/30 text-yellow-500 px-6 py-4 rounded-xl flex items-center gap-4 mb-[-1rem]"
      >
        <svg class="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <span class="text-sm font-medium"
          >Warning: WebGPU is not supported by your browser. Running this may fallback to CPU or crash.</span
        >
      </div>

      <!-- Input Query -->
      <div class="flex flex-col gap-4 z-10 w-full">
        <label for="inputText" class="text-sm font-semibold text-slate-300 uppercase tracking-widest pl-1">
          Input Query
        </label>
        <textarea
          id="inputText"
          v-model="inputText"
          :disabled="isComputing"
          class="w-full bg-slate-900/60 border border-slate-700/50 rounded-xl p-5 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-y min-h-[140px] shadow-inner disabled:opacity-50 disabled:cursor-not-allowed"
          placeholder="Type a sentence to encode (e.g., 'What is the capital of France?')"
        />
      </div>

      <!-- Model Selector -->
      <div class="flex flex-col sm:flex-row gap-6 items-end z-10 w-full mb-4">
        <div class="flex flex-col gap-4 flex-1 w-full">
          <label for="model" class="text-sm font-semibold text-slate-300 uppercase tracking-widest pl-1">
            Select Model
          </label>
          <div class="relative">
            <select
              id="model"
              v-model="model"
              :disabled="isComputing"
              class="w-full bg-slate-900/60 border border-slate-700/50 rounded-xl p-4 pr-10 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none appearance-none cursor-pointer shadow-inner font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <option value="MongoDB/mdbr-leaf-ir">MongoDB/mdbr-leaf-ir (IR v10 - Modern)</option>
              <option value="Xenova/all-MiniLM-L6-v2">Xenova/all-MiniLM-L6-v2 (IR v8 - Classic/Compatible)</option>
            </select>
            <div class="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none text-slate-400">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      <!-- Backend Selector + Button -->
      <div class="flex flex-col sm:flex-row gap-6 items-end z-10 w-full">
        <div class="flex flex-col gap-4 flex-1 w-full">
          <label for="backend" class="text-sm font-semibold text-slate-300 uppercase tracking-widest pl-1">
            Execution Backend (EP)
          </label>
          <div class="relative">
            <select
              id="backend"
              v-model="backend"
              :disabled="isComputing"
              class="w-full bg-slate-900/60 border border-slate-700/50 rounded-xl p-4 pr-10 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none appearance-none cursor-pointer shadow-inner font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <option value="wasm">WASM (SIMD/Threads) - CPU</option>
              <option value="webgpu">WebGPU - GPU (Modern)</option>
            </select>
            <div class="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none text-slate-400">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>
        <button
          :disabled="isComputing || !inputText.trim()"
          class="w-full sm:w-auto px-10 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-400 disabled:shadow-none text-white font-bold rounded-xl transition-all shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:-translate-y-0.5 active:translate-y-0 disabled:transform-none flex items-center justify-center gap-3 min-w-[220px]"
          @click="handleCompute"
        >
          <template v-if="isComputing">
            <svg class="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
              <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Processing...
          </template>
          <template v-else> Generate Embeddings </template>
        </button>
      </div>

      <!-- Progress message -->
      <div v-if="isComputing && progressMsg" class="z-10 text-center text-blue-400 font-medium animate-pulse mt-2">
        {{ progressMsg }}
      </div>

      <!-- Results -->
      <div
        v-if="(metrics || embeddings) && !isComputing"
        class="z-10 mt-6 grid grid-cols-1 md:grid-cols-2 gap-6 w-full pt-6 border-t border-slate-700/50"
      >
        <!-- Performance Metrics -->
        <div class="flex flex-col gap-4">
          <h2 class="text-xl font-bold text-slate-200">Performance Metrics</h2>
          <div class="grid grid-cols-2 gap-4">
            <div
              class="bg-slate-900/60 border border-slate-700/50 rounded-xl p-5 shadow-inner flex flex-col items-center justify-center"
            >
              <span class="text-slate-400 text-sm font-medium mb-1">Cold Load Time</span>
              <div class="flex items-baseline gap-1">
                <span class="text-3xl font-extrabold text-blue-400">{{ metrics?.loadTime.toFixed(1) }}</span>
                <span class="text-slate-500 font-medium">ms</span>
              </div>
            </div>
            <div
              class="bg-slate-900/60 border border-slate-700/50 rounded-xl p-5 shadow-inner flex flex-col items-center justify-center"
            >
              <span class="text-slate-400 text-sm font-medium mb-1">Inference Latency</span>
              <div class="flex items-baseline gap-1">
                <span class="text-3xl font-extrabold text-purple-400">{{ metrics?.inferTime.toFixed(1) }}</span>
                <span class="text-slate-500 font-medium">ms</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Output Vector -->
        <div class="flex flex-col gap-4">
          <div class="flex justify-between items-center">
            <h2 class="text-xl font-bold text-slate-200">Output Vector</h2>
            <span class="text-xs font-mono bg-slate-800 text-slate-400 px-2 py-1 rounded"
              >Dim: {{ embeddings?.length || 0 }}</span
            >
          </div>
          <div class="bg-slate-900/60 border border-slate-700/50 rounded-xl p-4 shadow-inner">
            <div class="font-mono text-xs text-slate-400 break-words h-32 overflow-y-auto custom-scrollbar p-1">
              {{ embeddingPreview() }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
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
</style>
