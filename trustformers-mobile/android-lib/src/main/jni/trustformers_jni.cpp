#include <jni.h>
#include <android/log.h>
#include <string>
#include <cstring>

#define LOG_TAG "TrustformersJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// External Rust functions declarations
extern "C" {
    // These functions are implemented in Rust
    jlong Java_com_trustformers_TrustformersEngine_createEngine(JNIEnv* env, jclass clazz, jstring config_json);
    jboolean Java_com_trustformers_TrustformersEngine_loadModel(JNIEnv* env, jclass clazz, jlong engine_ptr, jstring model_path);
    jbyteArray Java_com_trustformers_TrustformersEngine_inference(JNIEnv* env, jclass clazz, jlong engine_ptr, jbyteArray input_data);
    jbyteArray Java_com_trustformers_TrustformersEngine_batchInference(JNIEnv* env, jclass clazz, jlong engine_ptr, jbyteArray batch_input_data, jint batch_size);
    jstring Java_com_trustformers_TrustformersEngine_getDeviceInfo(JNIEnv* env, jclass clazz);
    void Java_com_trustformers_TrustformersEngine_releaseEngine(JNIEnv* env, jclass clazz, jlong engine_ptr);
}

// JNI OnLoad callback
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    
    LOGI("TrustformersEngine JNI loaded");
    
    // Additional initialization if needed
    
    return JNI_VERSION_1_6;
}

// JNI OnUnload callback
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    LOGI("TrustformersEngine JNI unloaded");
    // Cleanup if needed
}