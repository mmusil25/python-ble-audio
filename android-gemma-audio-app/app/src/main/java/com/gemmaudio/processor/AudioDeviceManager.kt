package com.gemmaudio.processor

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import androidx.annotation.RequiresApi

data class AudioDevice(
    val id: Int,
    val name: String,
    val type: Int,
    val isSource: Boolean
)

class AudioDeviceManager(private val context: Context) {
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    
    fun getAvailableInputDevices(): List<AudioDevice> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS).map { device ->
                AudioDevice(
                    id = device.id,
                    name = getDeviceName(device),
                    type = device.type,
                    isSource = true
                )
            }
        } else {
            // For older devices, return default microphone
            listOf(AudioDevice(
                id = 0,
                name = "Default Microphone",
                type = AudioDeviceInfo.TYPE_BUILTIN_MIC,
                isSource = true
            ))
        }
    }
    
    @RequiresApi(Build.VERSION_CODES.M)
    private fun getDeviceName(device: AudioDeviceInfo): String {
        return when (device.type) {
            AudioDeviceInfo.TYPE_BUILTIN_MIC -> "Built-in Microphone"
            AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "Bluetooth Headset"
            AudioDeviceInfo.TYPE_WIRED_HEADSET -> "Wired Headset"
            AudioDeviceInfo.TYPE_USB_DEVICE -> "USB Device"
            AudioDeviceInfo.TYPE_USB_ACCESSORY -> "USB Accessory"
            AudioDeviceInfo.TYPE_TELEPHONY -> "Telephony"
            else -> device.productName.toString().ifEmpty { "Audio Device ${device.id}" }
        }
    }
    
    fun setPreferredInputDevice(deviceId: Int) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val devices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
            val preferredDevice = devices.find { it.id == deviceId }
            // Note: Setting preferred device requires additional implementation
            // with MediaRecorder.setPreferredDevice() method
        }
    }
}