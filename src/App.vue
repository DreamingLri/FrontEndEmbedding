<script setup lang="ts">
import { ref, watch } from 'vue'
import SearchRAG from './components/SearchRAG.vue'
import RetrievalTrace from './components/RetrievalTrace.vue'
import ReviewWorkbench from './components/review/ReviewWorkbench.vue'

const traceData = ref<any>(null)
const initialMode = window.location.hash === '#review' ? 'review' : 'search'
const mode = ref<'search' | 'review'>(initialMode)

const handleTraceUpdate = (data: any) => {
  traceData.value = data
}

watch(mode, value => {
  window.location.hash = value === 'review' ? 'review' : ''
})
</script>

<template>
  <div
    class="min-h-screen w-full flex flex-col items-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-900 px-4 py-4 text-slate-100 font-sans sm:px-6 sm:py-6"
  >
    <header class="z-10 w-full max-w-7xl text-center pb-5 pt-2">
      <h1
        class="mb-2 inline-block bg-gradient-to-r from-sky-300 via-blue-300 to-indigo-300 bg-clip-text text-[32px] font-extrabold tracking-tight text-transparent sm:text-[36px]"
      >
        SuAsk Search Engine
      </h1>
      <p class="mx-auto max-w-xl text-sm text-slate-400 sm:text-[15px]">
        <template v-if="mode === 'search'">
          优先返回官方原话，同时保留可展开的相关政策要点
        </template>
        <template v-else>
          用统一界面完成 granularity challenge 的人工复核与标签确认
        </template>
      </p>
      <div class="mt-4 flex items-center justify-center gap-2">
        <button
          class="rounded-full border px-4 py-2 text-sm font-semibold transition-all"
          :class="mode === 'search'
            ? 'border-cyan-400/70 bg-cyan-400/10 text-cyan-100'
            : 'border-slate-700 bg-slate-900/70 text-slate-300 hover:border-cyan-400/40'"
          type="button"
          @click="mode = 'search'"
        >
          Search Mode
        </button>
        <button
          class="rounded-full border px-4 py-2 text-sm font-semibold transition-all"
          :class="mode === 'review'
            ? 'border-emerald-400/70 bg-emerald-500/10 text-emerald-100'
            : 'border-slate-700 bg-slate-900/70 text-slate-300 hover:border-emerald-400/40'"
          type="button"
          @click="mode = 'review'"
        >
          Review Mode
        </button>
      </div>
    </header>

    <div
      v-if="mode === 'search'"
      class="z-10 flex h-[78vh] min-h-[620px] max-h-[860px] w-full max-w-7xl flex-col items-stretch gap-4 lg:flex-row lg:gap-5"
    >
      <div class="overflow-hidden lg:min-w-0 lg:flex-[1.9]">
        <SearchRAG 
          @trace-updated="handleTraceUpdate"
        />
      </div>
      <div class="overflow-hidden lg:min-w-[320px] lg:flex-[0.95]">
        <RetrievalTrace :trace-data="traceData" />
      </div>
    </div>

    <div
      v-else
      class="z-10 w-full max-w-[1680px]"
    >
      <ReviewWorkbench />
    </div>
  </div>
</template>

<style>
/* Smooth transitions */
* {
  transition: background-color 0.2s ease, border-color 0.2s ease;
}
</style>
