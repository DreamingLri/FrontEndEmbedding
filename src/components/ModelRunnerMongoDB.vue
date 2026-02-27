<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { AutoModel, AutoTokenizer, env } from '@huggingface/transformers'

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';

const props = defineProps<{
  inputText: string;
  defaultBackend?: 'wasm' | 'webgpu';
}>()

const modelPath = 'MongoDB/mdbr-leaf-ir'
const modelTitle = 'IR v10 - Modern (Local)'
const backend = ref(props.defaultBackend || 'webgpu')
const dtype = ref<'q8' | 'fp16' | 'fp32' | 'q4' | 'q4f16'>('q4')
const isComputing = ref(false)
const progressMsg = ref('')

const metrics = ref<{ loadTime: number; inferTime: number } | null>(null)
const embeddings = ref<Float32Array | null>(null)
const error = ref<string | null>(null)

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
  if (!props.inputText.trim()) return

  isComputing.value = true
  error.value = null
  metrics.value = null
  embeddings.value = null
  progressMsg.value = 'Loading model...'

  try {
    const loadStart = performance.now()

    const tokenizer = await AutoTokenizer.from_pretrained(modelPath)
    const model = await AutoModel.from_pretrained(modelPath, {
      device: backend.value === 'webgpu' ? 'webgpu' : 'wasm',
      dtype: dtype.value,
    })

    const loadEnd = performance.now()

    progressMsg.value = 'Running inference...'
    const inferStart = performance.now()
    
        const inputs = await tokenizer(
        ["Represent this sentence for searching relevant passages: " + props.inputText],
        { padding: true }
        )
    
    const { sentence_embedding } = await model(inputs)
    const inferEnd = performance.now()

    metrics.value = {
      loadTime: loadEnd - loadStart,
      inferTime: inferEnd - inferStart,
    }

    if (sentence_embedding && sentence_embedding.data) {
      embeddings.value = new Float32Array(sentence_embedding.data as any)
    } else {
      embeddings.value = new Float32Array(sentence_embedding.tolist()[0])
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
  <div class="bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-2xl p-6 flex flex-col gap-6 relative overflow-hidden h-full">
    <div class="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1/2 bg-blue-500/5 blur-[100px] pointer-events-none rounded-full" />
    
    <header class="z-10 border-b border-slate-700/50 pb-4 w-full">
      <h2 class="text-xl font-bold tracking-tight text-slate-100 mb-1">
        {{ modelPath }}
      </h2>
      <p class="text-slate-400 text-sm">
        {{ modelTitle }}
      </p>
    </header>

    <div v-if="error" class="z-10 bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-xl flex items-center gap-3">
      <svg class="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
      <span class="text-xs font-medium">{{ error }}</span>
    </div>

    <div v-if="backend === 'webgpu' && gpuSupported === false" class="z-10 bg-yellow-500/10 border border-yellow-500/30 text-yellow-500 px-4 py-3 rounded-xl flex items-center gap-3 mb-[-1rem]">
      <svg class="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
         <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
      <span class="text-xs font-medium">Warning: WebGPU is not supported by your browser.</span>
    </div>

    <div class="flex flex-col sm:flex-row gap-4 z-10 w-full mt-auto">
      <div class="flex flex-col gap-2 flex-1 relative">
        <label class="text-xs font-semibold text-slate-300 uppercase tracking-widest pl-1">Backend</label>
        <select v-model="backend" :disabled="isComputing" class="w-full bg-slate-900/60 border border-slate-700/50 rounded-lg p-3 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none appearance-none cursor-pointer">
          <option value="wasm">WASM (CPU)</option>
          <option value="webgpu">WebGPU (GPU)</option>
        </select>
        <div class="absolute inset-y-0 right-0 flex items-center px-3 pt-6 pointer-events-none text-slate-400">
           <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
        </div>
      </div>
      
      <div class="flex flex-col gap-2 flex-1 relative">
        <label class="text-xs font-semibold text-slate-300 uppercase tracking-widest pl-1">Precision</label>
        <select v-model="dtype" :disabled="isComputing" class="w-full bg-slate-900/60 border border-slate-700/50 rounded-lg p-3 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none appearance-none cursor-pointer">
          <option value="q4">INT4 (q4)</option>
          <option value="q8">INT8 (q8)</option>
          <option value="q4f16">INT4F16 (q4f16)</option>
          <option value="fp16">FP16 (fp16)</option>
          <option value="fp32">FP32 (fp32)</option>
        </select>
        <div class="absolute inset-y-0 right-0 flex items-center px-3 pt-6 pointer-events-none text-slate-400">
           <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
        </div>
      </div>
    </div>

    <button
      :disabled="isComputing || !inputText.trim()"
      class="z-10 w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-400 text-white text-sm font-bold rounded-xl transition-all shadow-lg flex items-center justify-center gap-2 mt-2"
      @click="handleCompute"
    >
      <template v-if="isComputing">
        <svg class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        {{ progressMsg }}
      </template>
      <template v-else>Generate Embeddings</template>
    </button>

    <div v-if="(metrics || embeddings) && !isComputing" class="z-10 mt-4 flex flex-col gap-4 w-full pt-4 border-t border-slate-700/50 fill-available">
      <div class="grid grid-cols-2 gap-3">
        <div class="bg-slate-900/60 border border-slate-700/50 rounded-xl p-3 shadow-inner flex flex-col items-center justify-center">
          <span class="text-slate-400 text-xs font-medium mb-1">Cold Load Time</span>
          <div class="flex items-baseline gap-1">
            <span class="text-xl font-extrabold text-blue-400">{{ metrics?.loadTime.toFixed(1) }}</span>
            <span class="text-slate-500 text-xs font-medium">ms</span>
          </div>
        </div>
        <div class="bg-slate-900/60 border border-slate-700/50 rounded-xl p-3 shadow-inner flex flex-col items-center justify-center">
          <span class="text-slate-400 text-xs font-medium mb-1">Inference Latency</span>
          <div class="flex items-baseline gap-1">
            <span class="text-xl font-extrabold text-purple-400">{{ metrics?.inferTime.toFixed(1) }}</span>
            <span class="text-slate-500 text-xs font-medium">ms</span>
          </div>
        </div>
      </div>

      <div class="flex flex-col gap-2">
        <div class="flex justify-between items-center">
          <h3 class="text-sm font-bold text-slate-200">Output Vector</h3>
          <span class="text-[10px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded">Dim: {{ embeddings?.length || 0 }}</span>
        </div>
        <div class="bg-slate-900/60 border border-slate-700/50 rounded-xl p-3 shadow-inner">
          <div class="font-mono text-[10px] text-slate-400 break-words h-24 overflow-y-auto custom-scrollbar p-1">
            {{ embeddingPreview() }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
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
.fill-available {
  flex: 1;
}
</style>
