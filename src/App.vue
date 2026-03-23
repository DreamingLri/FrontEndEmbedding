<script setup lang="ts">
import { ref } from 'vue'
import SearchRAG from './components/SearchRAG.vue'
import RetrievalTrace from './components/RetrievalTrace.vue'

const traceData = ref<any>(null)

const handleTraceUpdate = (data: any) => {
  traceData.value = data
}
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
        SuAsk Search Engine
      </h1>
      <p class="text-slate-400 text-base max-w-2xl mx-auto">
        高性能本地语义检索与校园政策搜索底座
      </p>
    </header>

    <!-- Search Section (The New Addition) -->
    <div class="flex flex-col md:flex-row gap-6 w-full max-w-6xl z-10 items-stretch mb-12 h-[75vh] min-h-[600px] max-h-[850px]">
      <div class="flex-[2] overflow-hidden">
        <SearchRAG 
          @trace-updated="handleTraceUpdate"
        />
      </div>
      <div class="flex-1 overflow-hidden">
        <RetrievalTrace :trace-data="traceData" />
      </div>
    </div>
  </div>
</template>

<style>
/* Smooth transitions */
* {
  transition: background-color 0.2s ease, border-color 0.2s ease;
}
</style>
