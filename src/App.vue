<script setup lang="ts">
import { ref } from 'vue'
import ModelRunnerMongoDB from './components/ModelRunnerMongoDB.vue'

const inputText = ref('This is a benchmark test document to evaluate WebGPU, WebGL, and WASM performance.')
</script>

<template>
  <div
    class="min-h-screen w-full flex flex-col items-center p-4 sm:p-8 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans"
  >
    <!-- Header -->
    <header class="z-10 w-full text-center pb-8 pt-4">
      <h1
        class="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 inline-block mb-3"
      >
        EmbeddingBench
      </h1>
      <p class="text-slate-400 text-base max-w-2xl mx-auto">
        Client-side ONNX Runtime Web benchmarks.
      </p>
    </header>

    <!-- Global Input Query -->
    <div class="flex flex-col gap-4 z-10 w-full max-w-6xl mb-8">
      <label for="inputText" class="text-sm font-semibold text-slate-300 uppercase tracking-widest pl-1">
        Global Input Query
      </label>
      <textarea
        id="inputText"
        v-model="inputText"
        class="w-full bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-xl p-5 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-y min-h-[140px] shadow-lg"
        placeholder="Type a sentence to encode (e.g., 'What is the capital of France?')"
      />
    </div>

    <!-- Dual Model Components -->
    <div class="flex flex-col md:flex-row gap-6 w-full max-w-6xl z-10 items-stretch pb-12">
      <ModelRunnerMongoDB
        class="flex-1"
        default-backend="webgpu"
        :input-text="inputText"
      />
    </div>
  </div>
</template>
