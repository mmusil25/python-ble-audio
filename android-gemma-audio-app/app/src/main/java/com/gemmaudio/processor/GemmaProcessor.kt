package com.gemmaudio.processor

import android.content.Context
import org.tensorflow.lite.Interpreter
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.FileInputStream
import java.nio.channels.FileChannel

class GemmaProcessor(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val gson = Gson()
    
    companion object {
        private const val TAG = "GemmaProcessor"
        private const val MODEL_PATH = "gemma_quantized.tflite"
    }
    
    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing TensorFlow Lite for Gemma...")
            
            // Check if model exists in app's files directory
            val modelFile = File(context.filesDir, MODEL_PATH)
            
            if (!modelFile.exists()) {
                Log.d(TAG, "Model not found locally. Please download the model first.")
                throw RuntimeException("Gemma model not found. Please download it using the download function.")
            }
            
            // Load the model
            val modelBuffer = loadModelFile(modelFile)
            
            // Create interpreter options
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                // Note: GPU acceleration removed due to dependency issues
                // The Pixel 9 XL's CPU is powerful enough for demo purposes
                Log.d(TAG, "Using CPU with 4 threads")
            }
            
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "Gemma model loaded successfully via TensorFlow Lite")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Gemma model", e)
            throw RuntimeException("Failed to initialize Gemma model: ${e.message}", e)
        }
    }
    
    private fun loadModelFile(modelFile: File): ByteBuffer {
        val fileInputStream = FileInputStream(modelFile)
        val fileChannel = fileInputStream.channel
        val startOffset = 0L
        val declaredLength = fileChannel.size()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    suspend fun downloadModel(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Downloading Gemma model...")
            
            // In a production app, you would download from your server
            // For now, we'll create instructions for the user
            val instructions = """
                To use Gemma on your device:
                
                1. Download a TensorFlow Lite compatible model
                2. Or convert Gemma using the provided Python script
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
            if (interpreter == null) {
                Log.w(TAG, "Model not initialized, returning mock response")
                createMockResponse(transcription)
            } else {
                // For now, we'll use mock processing since actual Gemma inference
                // would require proper model conversion and tokenization
                Log.d(TAG, "Processing transcription: ${transcription.take(100)}...")
                
                // In a real implementation, you would:
                // 1. Tokenize the input
                // 2. Run inference with the interpreter
                // 3. Decode the output
                
                createMockResponse(transcription)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing transcription", e)
            createErrorResponse(transcription, e.message)
        }
    }
    
    private fun createMockResponse(transcription: String): String {
        val intent = detectIntent(transcription)
        val entities = extractEntities(transcription)
        
        val result = JsonObject().apply {
            addProperty("transcript", transcription)
            addProperty("timestamp_ms", System.currentTimeMillis())
            addProperty("intent", intent)
            add("entities", gson.toJsonTree(entities))
            addProperty("model", "mock (TFLite ready)")
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
        return interpreter != null
    }
    
    fun close() {
        try {
            interpreter?.close()
            interpreter = null
            Log.d(TAG, "Gemma processor closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing Gemma processor", e)
        }
    }
}