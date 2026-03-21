<template>
  <div class="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
    <!-- Header -->
    <header class="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center py-4">
          <div class="flex items-center">
            <DocumentTextIcon class="h-8 w-8 text-indigo-600 mr-3" />
            <h1 class="text-2xl font-bold text-gray-900">
              TrustformeRS Document Analyzer
            </h1>
          </div>
          <div class="flex items-center space-x-4">
            <span class="text-sm text-gray-500">Vue 3 + Composition API</span>
            <div class="flex items-center">
              <div 
                :class="[
                  'h-2 w-2 rounded-full mr-2',
                  modelStatus.loaded ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'
                ]"
              ></div>
              <span class="text-sm text-gray-600">
                {{ modelStatus.loaded ? 'Models Ready' : 'Loading Models...' }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Document Input Section -->
      <div class="bg-white rounded-xl shadow-lg p-6 mb-8">
        <div class="mb-6">
          <h2 class="text-xl font-semibold text-gray-800 mb-2">
            Document Input
          </h2>
          <p class="text-gray-600">
            Upload a document or paste text to analyze sentiment, extract keywords, and summarize content.
          </p>
        </div>

        <!-- File Upload Area -->
        <div 
          class="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-6 transition-colors hover:border-indigo-400"
          @drop="handleFileDrop"
          @dragover.prevent
          @dragenter.prevent
        >
          <div class="text-center">
            <CloudArrowUpIcon class="mx-auto h-12 w-12 text-gray-400" />
            <div class="mt-2">
              <label for="file-upload" class="cursor-pointer">
                <span class="text-indigo-600 hover:text-indigo-500 font-medium">
                  Upload a file
                </span>
                <input 
                  id="file-upload" 
                  type="file" 
                  class="sr-only"
                  accept=".txt,.pdf,.doc,.docx,.md"
                  @change="handleFileSelect"
                />
              </label>
              <span class="text-gray-500"> or drag and drop</span>
            </div>
            <p class="text-xs text-gray-500">
              TXT, PDF, DOC, DOCX, MD up to 10MB
            </p>
          </div>
        </div>

        <!-- Text Input Area -->
        <div class="mb-4">
          <label for="document-text" class="block text-sm font-medium text-gray-700 mb-2">
            Or paste your text here:
          </label>
          <textarea
            id="document-text"
            v-model="documentText"
            placeholder="Paste or type your document content here... The analyzer will process sentiment, extract keywords, generate summaries, and identify key topics."
            class="w-full h-40 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none text-gray-700"
            :disabled="isAnalyzing"
          ></textarea>
        </div>

        <!-- Analysis Controls -->
        <div class="flex flex-wrap gap-4 items-center justify-between">
          <div class="flex items-center space-x-4">
            <button
              @click="analyzeDocument"
              :disabled="!canAnalyze"
              class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <template v-if="isAnalyzing">
                <ArrowPathIcon class="animate-spin -ml-1 mr-2 h-4 w-4" />
                Analyzing...
              </template>
              <template v-else>
                <MagnifyingGlassIcon class="-ml-1 mr-2 h-4 w-4" />
                Analyze Document
              </template>
            </button>
            
            <button
              @click="clearAll"
              class="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <XMarkIcon class="-ml-1 mr-2 h-4 w-4" />
              Clear
            </button>
          </div>

          <div class="text-sm text-gray-500">
            {{ documentText.length }} / 10,000 characters
          </div>
        </div>
      </div>

      <!-- Error Display -->
      <div v-if="error" class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <div class="flex">
          <ExclamationTriangleIcon class="h-5 w-5 text-red-400" />
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">
              Analysis Error
            </h3>
            <div class="mt-2 text-sm text-red-700">
              {{ error }}
            </div>
          </div>
        </div>
      </div>

      <!-- Analysis Results -->
      <div v-if="analysisResults" class="space-y-6">
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <!-- Sentiment Card -->
          <div class="bg-white rounded-lg shadow-md p-6">
            <div class="flex items-center">
              <div 
                :class="[
                  'flex-shrink-0 p-3 rounded-lg',
                  getSentimentColors(analysisResults.sentiment.label).background
                ]"
              >
                <component 
                  :is="getSentimentIcon(analysisResults.sentiment.label)"
                  :class="['h-6 w-6', getSentimentColors(analysisResults.sentiment.label).text]"
                />
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Sentiment</p>
                <p class="text-2xl font-semibold text-gray-900 capitalize">
                  {{ analysisResults.sentiment.label }}
                </p>
                <p class="text-xs text-gray-500">
                  {{ Math.round(analysisResults.sentiment.confidence) }}% confidence
                </p>
              </div>
            </div>
          </div>

          <!-- Reading Time Card -->
          <div class="bg-white rounded-lg shadow-md p-6">
            <div class="flex items-center">
              <div class="flex-shrink-0 p-3 bg-blue-100 rounded-lg">
                <ClockIcon class="h-6 w-6 text-blue-600" />
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Reading Time</p>
                <p class="text-2xl font-semibold text-gray-900">
                  {{ analysisResults.readingTime }} min
                </p>
                <p class="text-xs text-gray-500">
                  {{ analysisResults.wordCount }} words
                </p>
              </div>
            </div>
          </div>

          <!-- Complexity Card -->
          <div class="bg-white rounded-lg shadow-md p-6">
            <div class="flex items-center">
              <div class="flex-shrink-0 p-3 bg-purple-100 rounded-lg">
                <AcademicCapIcon class="h-6 w-6 text-purple-600" />
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Complexity</p>
                <p class="text-2xl font-semibold text-gray-900 capitalize">
                  {{ analysisResults.complexity }}
                </p>
                <p class="text-xs text-gray-500">
                  Grade {{ analysisResults.readabilityScore }}
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- Detailed Analysis -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Keywords & Topics -->
          <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-medium text-gray-900 mb-4 flex items-center">
              <TagIcon class="h-5 w-5 text-gray-500 mr-2" />
              Keywords & Topics
            </h3>
            
            <div class="space-y-4">
              <div>
                <h4 class="text-sm font-medium text-gray-700 mb-2">Key Terms</h4>
                <div class="flex flex-wrap gap-2">
                  <span 
                    v-for="keyword in analysisResults.keywords" 
                    :key="keyword.word"
                    class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800"
                  >
                    {{ keyword.word }}
                    <span class="ml-1 text-indigo-600">{{ keyword.frequency }}</span>
                  </span>
                </div>
              </div>

              <div>
                <h4 class="text-sm font-medium text-gray-700 mb-2">Topics</h4>
                <div class="space-y-2">
                  <div 
                    v-for="topic in analysisResults.topics" 
                    :key="topic.name"
                    class="flex items-center justify-between"
                  >
                    <span class="text-sm text-gray-600">{{ topic.name }}</span>
                    <div class="flex items-center">
                      <div class="w-20 bg-gray-200 rounded-full h-2 mr-2">
                        <div 
                          class="bg-indigo-600 h-2 rounded-full"
                          :style="{ width: `${topic.confidence * 100}%` }"
                        ></div>
                      </div>
                      <span class="text-xs text-gray-500">
                        {{ Math.round(topic.confidence * 100) }}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Summary -->
          <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-medium text-gray-900 mb-4 flex items-center">
              <DocumentTextIcon class="h-5 w-5 text-gray-500 mr-2" />
              AI Summary
            </h3>
            
            <div class="prose prose-sm text-gray-600">
              <p>{{ analysisResults.summary }}</p>
            </div>

            <div class="mt-4 pt-4 border-t border-gray-200">
              <h4 class="text-sm font-medium text-gray-700 mb-2">Key Points</h4>
              <ul class="space-y-1">
                <li 
                  v-for="point in analysisResults.keyPoints" 
                  :key="point"
                  class="text-sm text-gray-600 flex items-start"
                >
                  <ChevronRightIcon class="h-4 w-4 text-gray-400 mr-1 mt-0.5 flex-shrink-0" />
                  {{ point }}
                </li>
              </ul>
            </div>
          </div>
        </div>

        <!-- Detailed Analytics -->
        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <ChartBarIcon class="h-5 w-5 text-gray-500 mr-2" />
            Detailed Analytics
          </h3>

          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="text-center">
              <p class="text-2xl font-semibold text-gray-900">{{ analysisResults.sentences }}</p>
              <p class="text-sm text-gray-600">Sentences</p>
            </div>
            <div class="text-center">
              <p class="text-2xl font-semibold text-gray-900">{{ analysisResults.paragraphs }}</p>
              <p class="text-sm text-gray-600">Paragraphs</p>
            </div>
            <div class="text-center">
              <p class="text-2xl font-semibold text-gray-900">{{ analysisResults.avgWordsPerSentence }}</p>
              <p class="text-sm text-gray-600">Avg Words/Sentence</p>
            </div>
            <div class="text-center">
              <p class="text-2xl font-semibold text-gray-900">{{ analysisResults.uniqueWords }}</p>
              <p class="text-sm text-gray-600">Unique Words</p>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-50 border-t border-gray-200 mt-16">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div class="text-center text-gray-600 text-sm">
          <p>Powered by TrustformeRS • Built with Vue 3 Composition API • Real-time ML Document Analysis</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useFileDialog, useDropZone } from '@vueuse/core'
import {
  DocumentTextIcon,
  CloudArrowUpIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  AcademicCapIcon,
  TagIcon,
  ChartBarIcon,
  ChevronRightIcon,
  FaceSmileIcon,
  FaceFrownIcon,
  MinusCircleIcon
} from '@heroicons/vue/24/outline'

// Reactive state
const documentText = ref('')
const isAnalyzing = ref(false)
const analysisResults = ref(null)
const error = ref(null)
const modelStatus = ref({ loaded: false })

// Computed properties
const canAnalyze = computed(() => {
  return documentText.value.trim().length > 0 && !isAnalyzing.value && modelStatus.value.loaded
})

// Initialize TrustformeRS
onMounted(async () => {
  try {
    const { TrustformersVue } = await import('@trustformers/js/frameworks')
    
    await TrustformersVue.initialize({
      debug: import.meta.env.DEV
    })
    
    modelStatus.value.loaded = true
    console.log('TrustformeRS Vue integration initialized')
  } catch (err) {
    console.error('Failed to initialize TrustformeRS:', err)
    error.value = 'Failed to initialize analysis models. Please refresh the page.'
  }
})

// File handling
const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    readFile(file)
  }
}

const handleFileDrop = (event) => {
  event.preventDefault()
  const file = event.dataTransfer.files[0]
  if (file) {
    readFile(file)
  }
}

const readFile = (file) => {
  if (file.size > 10 * 1024 * 1024) {
    error.value = 'File size must be less than 10MB'
    return
  }

  const reader = new FileReader()
  reader.onload = (e) => {
    documentText.value = e.target.result
    error.value = null
  }
  reader.onerror = () => {
    error.value = 'Failed to read file. Please try again.'
  }
  reader.readAsText(file)
}

// Analysis functions
const analyzeDocument = async () => {
  if (!canAnalyze.value) return

  isAnalyzing.value = true
  error.value = null

  try {
    const { TrustformersVue } = await import('@trustformers/js/frameworks')
    
    // Perform comprehensive document analysis
    const results = await TrustformersVue.analyzeDocument(documentText.value, {
      includeSentiment: true,
      includeKeywords: true,
      includeSummary: true,
      includeTopics: true,
      includeReadability: true
    })

    analysisResults.value = {
      sentiment: results.sentiment,
      summary: results.summary,
      keywords: results.keywords.slice(0, 10), // Top 10 keywords
      topics: results.topics.slice(0, 5), // Top 5 topics
      keyPoints: results.keyPoints,
      wordCount: results.wordCount,
      sentences: results.sentenceCount,
      paragraphs: results.paragraphCount,
      readingTime: Math.ceil(results.wordCount / 200), // Assuming 200 WPM
      complexity: getComplexityLabel(results.readabilityScore),
      readabilityScore: results.readabilityScore,
      avgWordsPerSentence: Math.round(results.wordCount / results.sentenceCount),
      uniqueWords: results.uniqueWordCount
    }

  } catch (err) {
    console.error('Analysis failed:', err)
    error.value = 'Analysis failed. Please check your text and try again.'
  } finally {
    isAnalyzing.value = false
  }
}

const clearAll = () => {
  documentText.value = ''
  analysisResults.value = null
  error.value = null
}

// Helper functions
const getSentimentIcon = (sentiment) => {
  switch (sentiment?.toLowerCase()) {
    case 'positive':
      return FaceSmileIcon
    case 'negative':
      return FaceFrownIcon
    default:
      return MinusCircleIcon
  }
}

const getSentimentColors = (sentiment) => {
  switch (sentiment?.toLowerCase()) {
    case 'positive':
      return { background: 'bg-green-100', text: 'text-green-600' }
    case 'negative':
      return { background: 'bg-red-100', text: 'text-red-600' }
    default:
      return { background: 'bg-gray-100', text: 'text-gray-600' }
  }
}

const getComplexityLabel = (score) => {
  if (score <= 6) return 'elementary'
  if (score <= 9) return 'middle school'
  if (score <= 13) return 'high school'
  if (score <= 16) return 'college'
  return 'graduate'
}
</script>

<style scoped>
/* Custom styles for Vue transitions and animations */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

.slide-up-enter-active, .slide-up-leave-active {
  transition: all 0.3s ease;
}

.slide-up-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.slide-up-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}

/* Custom scrollbar for analysis results */
.prose {
  max-height: 200px;
  overflow-y: auto;
}

.prose::-webkit-scrollbar {
  width: 4px;
}

.prose::-webkit-scrollbar-track {
  background: #f1f5f9;
}

.prose::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 2px;
}

/* Smooth focus transitions */
textarea:focus,
input:focus,
button:focus {
  transition: all 0.2s ease-in-out;
}

/* Enhanced hover effects */
.hover-lift:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease-in-out;
}
</style>