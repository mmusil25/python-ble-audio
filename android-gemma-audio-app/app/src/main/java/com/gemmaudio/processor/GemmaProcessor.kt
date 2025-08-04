package com.gemmaudio.processor

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import android.util.Log

class GemmaProcessor(private val context: Context) {
    private var llmInference: LlmInference? = null
    private val gson = Gson()
    
    companion object {
        private const val TAG = "GemmaProcessor"
        private const val MODEL_PATH = "gemma-2b-it-gpu-int4.bin"
        private const val MAX_TOKENS = 1024
        private const val TEMPERATURE = 0.7f
        private const val TOP_K = 40
    }
    
    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing MediaPipe LLM Inference for Gemma...")
            
            // Check if model exists in app's files directory
            val modelFile = File(context.filesDir, MODEL_PATH)
            
            if (!modelFile.exists()) {
                Log.d(TAG, "Model not found locally. Please download the model first.")
                throw RuntimeException("Gemma model not found. Please download it using the download function.")
            }
            
            // Initialize MediaPipe LLM Inference
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(MAX_TOKENS)
                .setTemperature(TEMPERATURE)
                .setTopK(TOP_K)
                .setRandomSeed(101)
                .build()
            
            llmInference = LlmInference.createFromOptions(context, options)
            Log.d(TAG, "Gemma model loaded successfully via MediaPipe")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Gemma model", e)
            throw RuntimeException("Failed to initialize Gemma model: ${e.message}", e)
        }
    }
    
    suspend fun downloadModel(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Downloading Gemma model...")
            
            // In a production app, you would download from your server
            // For now, we'll create instructions for the user
            val instructions = """
                To use Gemma on your device:
                
                1. Download the Gemma 2B model from:
                   https://www.kaggle.com/models/google/gemma/tfLite/gemma-2b-it-gpu-int4
                
                2. Copy the model file to your device
                
                3. Use the app's file picker to select and import the model
                
                The model will be saved for future use.
            """.trimIndent()
            
            Log.i(TAG, instructions)
            false // Model not downloaded automatically
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in download process", e)
            false
        }
    }
    
    suspend fun importModelFromUri(uri: android.net.Uri): Boolean = withContext(Dispatchers.IO) {
        try {
            val modelFile = File(context.filesDir, MODEL_PATH)
            
            context.contentResolver.openInputStream(uri)?.use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            
            Log.d(TAG, "Model imported successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to import model", e)
            false
        }
    }
    
    suspend fun processTranscription(transcription: String): String = withContext(Dispatchers.Default) {
        try {
            if (llmInference == null) {
                Log.w(TAG, "LLM not initialized, returning mock response")
                return createMockResponse(transcription)
            }
            
            // Create the prompt for Gemma
            val prompt = createPrompt(transcription)
            
            Log.d(TAG, "Sending prompt to Gemma: ${prompt.take(100)}...")
            
            // Generate response using MediaPipe LLM Inference
            val response = withContext(Dispatchers.IO) {
                llmInference?.generateResponse(prompt) ?: ""
            }
            
            Log.d(TAG, "Received response from Gemma: ${response.take(100)}...")
            
            // Parse the response to extract JSON
            return parseGemmaResponse(response, transcription)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing transcription", e)
            return createErrorResponse(transcription, e.message)
        }
    }
    
    private fun createPrompt(transcription: String): String {
        return """You are a helpful assistant that extracts structured information from transcribed text.

Extract the intent and entities from the following transcription and return ONLY a valid JSON object.

Intent categories: question, request, statement, greeting, farewell, gratitude, command

Extract ALL important keywords including:
- Nouns (people, places, things, concepts)
- Verbs (actions, activities)
- Times, dates, numbers
- Descriptive words
- Any word that carries meaning

Transcription: "$transcription"

Return ONLY this JSON format with no additional text:
{
  "transcript": "$transcription",
  "timestamp_ms": ${System.currentTimeMillis()},
  "intent": "<intent>",
  "entities": ["<entity1>", "<entity2>", ...]
}"""
    }
    
    private fun parseGemmaResponse(response: String, originalTranscript: String): String {
        return try {
            // Try to extract JSON from the response
            val jsonStart = response.indexOf("{")
            val jsonEnd = response.lastIndexOf("}")
            
            if (jsonStart >= 0 && jsonEnd > jsonStart) {
                val jsonStr = response.substring(jsonStart, jsonEnd + 1)
                // Validate it's proper JSON
                gson.fromJson(jsonStr, JsonObject::class.java)
                jsonStr
            } else {
                // If no valid JSON found, create one from the response
                createFormattedResponse(originalTranscript, response)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse Gemma response as JSON", e)
            createMockResponse(originalTranscript)
        }
    }
    
    private fun createFormattedResponse(transcript: String, gemmaOutput: String): String {
        // Try to extract intent and entities from non-JSON response
        val intent = detectIntentFromText(gemmaOutput) ?: detectIntent(transcript)
        val entities = extractEntitiesFromText(gemmaOutput).ifEmpty { 
            extractEntities(transcript) 
        }
        
        val result = JsonObject().apply {
            addProperty("transcript", transcript)
            addProperty("timestamp_ms", System.currentTimeMillis())
            addProperty("intent", intent)
            add("entities", gson.toJsonTree(entities))
            addProperty("raw_llm_output", gemmaOutput.take(200))
        }
        
        return gson.toJson(result)
    }
    
    private fun detectIntentFromText(text: String): String? {
        val lowercaseText = text.lowercase()
        return when {
            lowercaseText.contains("question") -> "question"
            lowercaseText.contains("request") -> "request"
            lowercaseText.contains("command") -> "command"
            lowercaseText.contains("greeting") -> "greeting"
            lowercaseText.contains("farewell") -> "farewell"
            lowercaseText.contains("gratitude") -> "gratitude"
            lowercaseText.contains("statement") -> "statement"
            else -> null
        }
    }
    
    private fun extractEntitiesFromText(text: String): List<String> {
        // Simple extraction from Gemma's output
        val entities = mutableListOf<String>()
        
        // Look for common patterns like "entities: [...]" or "keywords: ..."
        val entityPattern = Regex("""(?:entities|keywords|extracted):\s*\[([^\]]+)\]""", RegexOption.IGNORE_CASE)
        val match = entityPattern.find(text)
        if (match != null) {
            val entitiesStr = match.groupValues[1]
            entities.addAll(
                entitiesStr.split(",")
                    .map { it.trim().replace("\"", "").replace("'", "") }
                    .filter { it.isNotEmpty() }
            )
        }
        
        return entities
    }
    
    private fun createMockResponse(transcription: String): String {
        val intent = detectIntent(transcription)
        val entities = extractEntities(transcription)
        
        val result = JsonObject().apply {
            addProperty("transcript", transcription)
            addProperty("timestamp_ms", System.currentTimeMillis())
            addProperty("intent", intent)
            add("entities", gson.toJsonTree(entities))
            addProperty("model", "mock (Gemma not loaded)")
        }
        
        return gson.toJson(result)
    }
    
    private fun createErrorResponse(transcription: String, error: String?): String {
        val result = JsonObject().apply {
            addProperty("transcript", transcription)
            addProperty("timestamp_ms", System.currentTimeMillis())
            addProperty("intent", "unknown")
            add("entities", gson.toJsonTree(emptyList<String>()))
            addProperty("error", error ?: "Unknown error")
        }
        
        return gson.toJson(result)
    }
    
    private fun detectIntent(text: String): String {
        return when {
            text.contains("?") -> "question"
            text.startsWith("please", ignoreCase = true) || 
            text.startsWith("could you", ignoreCase = true) -> "request"
            text.startsWith("hi", ignoreCase = true) || 
            text.startsWith("hello", ignoreCase = true) -> "greeting"
            text.contains("thank", ignoreCase = true) -> "gratitude"
            text.startsWith("bye", ignoreCase = true) || 
            text.contains("goodbye", ignoreCase = true) -> "farewell"
            text.contains("!") -> "command"
            else -> "statement"
        }
    }
    
    private fun extractEntities(text: String): List<String> {
        val words = text.split(" ")
            .map { it.trim().lowercase() }
            .filter { it.isNotEmpty() }
        
        val entities = mutableListOf<String>()
        
        // Extract numbers
        entities.addAll(words.filter { it.any { char -> char.isDigit() } })
        
        // Extract capitalized words (potential names/places)
        entities.addAll(
            text.split(" ")
                .filter { it.firstOrNull()?.isUpperCase() == true && it.length > 1 }
        )
        
        // Extract time-related words
        val timeWords = listOf("today", "tomorrow", "yesterday", "am", "pm", "morning", "evening", "night")
        entities.addAll(words.filter { it in timeWords })
        
        // Extract action words (verbs)
        val commonVerbs = listOf("call", "send", "email", "write", "read", "play", "stop", "start", "open", "close")
        entities.addAll(words.filter { it in commonVerbs })
        
        return entities.distinct()
    }
    
    fun isModelLoaded(): Boolean {
        return llmInference != null
    }
    
    fun close() {
        try {
            llmInference?.close()
            llmInference = null
            Log.d(TAG, "Gemma processor closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing Gemma processor", e)
        }
    }
}