package com.gemmaudio.processor

import android.app.Application
import android.content.ContentResolver
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import android.content.Intent
import android.os.Bundle
import java.util.*

data class ProcessingState(
    val isProcessing: Boolean = false,
    val isRecording: Boolean = false,
    val currentTranscription: String = "",
    val extractedJson: String = "",
    val status: String = "Ready",
    val error: String? = null,
    val processingTime: Long = 0,
    val isModelLoaded: Boolean = false,
    val modelDownloadInstructions: String? = null
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow(ProcessingState())
    val state: StateFlow<ProcessingState> = _state.asStateFlow()
    
    private val gemmaProcessor = GemmaProcessor(application)
    private var mediaRecorder: MediaRecorder? = null
    private var speechRecognizer: SpeechRecognizer? = null
    private var currentRecordingFile: File? = null
    
    init {
        initializeSpeechRecognizer()
        viewModelScope.launch {
            _state.value = _state.value.copy(status = "Checking for Gemma model...")
            try {
                gemmaProcessor.initialize()
                _state.value = _state.value.copy(
                    status = "Model ready",
                    isModelLoaded = true
                )
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    status = "Gemma model not found",
                    error = null,
                    isModelLoaded = false,
                    modelDownloadInstructions = """
                        To use Gemma on your device:
                        
                        1. Download a TensorFlow Lite compatible model
                           or convert Gemma using the provided Python script
                        
                        2. Tap "Import Model" below to select the .tflite file
                        
                        The app will run in demo mode until a model is loaded.
                    """.trimIndent()
                )
            }
        }
    }
    
    private fun initializeSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(getApplication())) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(getApplication())
            speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onError(error: Int) {
                    _state.value = _state.value.copy(
                        error = "Speech recognition error: $error"
                    )
                }
                
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val transcription = matches[0]
                        processTranscription(transcription)
                    }
                }
                
                override fun onPartialResults(partialResults: Bundle?) {
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        _state.value = _state.value.copy(currentTranscription = matches[0])
                    }
                }
                
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
    }
    
    fun processAudioFile(uri: Uri) {
        viewModelScope.launch {
            _state.value = _state.value.copy(
                isProcessing = true,
                status = "Processing audio file...",
                error = null
            )
            
            try {
                val startTime = System.currentTimeMillis()
                
                // Copy file to app's cache directory
                val contentResolver = getApplication<Application>().contentResolver
                val tempFile = File(getApplication<Application>().cacheDir, "temp_audio.wav")
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        input.copyTo(output)
                    }
                }
                
                // Transcribe using speech recognizer or a local model
                val transcription = transcribeAudioFile(tempFile)
                
                // Process with Gemma
                val extractedData = gemmaProcessor.processTranscription(transcription)
                
                val processingTime = System.currentTimeMillis() - startTime
                
                _state.value = _state.value.copy(
                    isProcessing = false,
                    currentTranscription = transcription,
                    extractedJson = extractedData,
                    status = "Processing complete",
                    processingTime = processingTime
                )
                
                // Clean up temp file
                tempFile.delete()
                
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    isProcessing = false,
                    status = "Error processing file",
                    error = e.message
                )
            }
        }
    }
    
    private suspend fun transcribeAudioFile(file: File): String {
        // For a real implementation, you would use a local speech-to-text model
        // or convert the audio to a format that can be processed by the speech recognizer
        // For now, we'll return a placeholder
        return "This is a placeholder transcription. In a real app, you would use a local STT model."
    }
    
    fun startRecording() {
        try {
            _state.value = _state.value.copy(
                isRecording = true,
                status = "Recording...",
                error = null,
                currentTranscription = "",
                extractedJson = ""
            )
            
            // Create recording file
            val recordingsDir = File(getApplication<Application>().filesDir, "recordings")
            recordingsDir.mkdirs()
            currentRecordingFile = File(recordingsDir, "recording_${System.currentTimeMillis()}.m4a")
            
            // Setup MediaRecorder
            mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                MediaRecorder(getApplication())
            } else {
                @Suppress("DEPRECATION")
                MediaRecorder()
            }.apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setOutputFile(currentRecordingFile?.absolutePath)
                prepare()
                start()
            }
            
            // Start speech recognition
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            }
            speechRecognizer?.startListening(intent)
            
        } catch (e: Exception) {
            _state.value = _state.value.copy(
                isRecording = false,
                status = "Error starting recording",
                error = e.message
            )
        }
    }
    
    fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            
            speechRecognizer?.stopListening()
            
            _state.value = _state.value.copy(
                isRecording = false,
                status = "Processing recording..."
            )
            
            // Process the final transcription if we have one
            if (_state.value.currentTranscription.isNotEmpty()) {
                processTranscription(_state.value.currentTranscription)
            }
            
        } catch (e: Exception) {
            _state.value = _state.value.copy(
                isRecording = false,
                status = "Error stopping recording",
                error = e.message
            )
        }
    }
    
    private fun processTranscription(transcription: String) {
        viewModelScope.launch {
            try {
                val extractedData = gemmaProcessor.processTranscription(transcription)
                _state.value = _state.value.copy(
                    currentTranscription = transcription,
                    extractedJson = extractedData,
                    status = "Ready"
                )
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    error = "Error processing transcription: ${e.message}"
                )
            }
        }
    }
    
    fun importModel(uri: Uri) {
        viewModelScope.launch {
            _state.value = _state.value.copy(
                status = "Importing model...",
                error = null
            )
            
            try {
                val success = gemmaProcessor.importModelFromUri(uri)
                if (success) {
                    // Try to initialize with the imported model
                    gemmaProcessor.initialize()
                    _state.value = _state.value.copy(
                        status = "Model imported and loaded successfully!",
                        isModelLoaded = true,
                        modelDownloadInstructions = null
                    )
                } else {
                    _state.value = _state.value.copy(
                        status = "Failed to import model",
                        error = "Could not import the model file"
                    )
                }
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    status = "Error importing model",
                    error = e.message
                )
            }
        }
    }
    
    fun proceedWithDemoMode() {
        _state.value = _state.value.copy(
            modelDownloadInstructions = null,
            status = "Running in demo mode",
            error = null
        )
    }
    
    override fun onCleared() {
        super.onCleared()
        mediaRecorder?.release()
        speechRecognizer?.destroy()
        gemmaProcessor.close()
    }
}