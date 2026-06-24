# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in android/sdk/tools/proguard/proguard-android.txt

# Keep TensorFlow Lite classes
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.** { *; }

# Keep MediaPipe classes — tasks-vision 0.10.9 native Graph.<clinit> walks the JVM
# stack reflectively to resolve its caller. If R8 strips/obfuscates the calling
# class, the native side throws `IllegalStateException: no caller found on the
# stack for: <obfuscated>` and the app crashes as soon as HandLandmarker is
# initialized. Keep classes, interfaces, enums and native methods, plus the
# reflection attributes MediaPipe reads at runtime.
-keep class com.google.mediapipe.** { *; }
-keep interface com.google.mediapipe.** { *; }
-keep enum com.google.mediapipe.** { *; }
-keepclassmembers class com.google.mediapipe.** { *; }
-keep,includedescriptorclasses class com.google.mediapipe.framework.** { *; }
-keep class com.google.mediapipe.proto.** { *; }
-keep class com.google.protobuf.** { *; }
-keepclassmembers class * {
    @com.google.mediapipe.framework.proto.* <fields>;
}
# MediaPipe uses Flogger for logging; the native Graph.<clinit> walks the JVM
# stack and fails if Flogger (or any other transitive com.google.* class) is
# missing or obfuscated.
-keep class com.google.common.flogger.** { *; }
-keep class com.google.common.** { *; }
-dontwarn com.google.common.flogger.**
-dontwarn com.google.common.**

# OkHttp — keep HttpLoggingInterceptor (R8 strips it as dead code because the
# app only references it via Lambda in NetworkModule). Without this, the
# BODY-level interceptor silently no-ops and we can't see HTTP traffic.
# (Solo afecta release; debug no usa R8.)
-keep class okhttp3.logging.** { *; }
-keep class okhttp3.** { *; }
-dontwarn okhttp3.logging.**
-keepclasseswithmembernames class * {
    native <methods>;
}
-keepattributes Signature,InnerClasses,EnclosingMethod,Exceptions,AnnotationDefault

# Keep Room entities
-keep class com.lsec.translator.data.local.entity.** { *; }

# Keep Hilt generated classes
-keepclasseswithmembers class * {
    @dagger.hilt.* <methods>;
}

# Keep data classes for serialization
-keepclassmembers class com.lsec.translator.** {
    <fields>;
}

# Kotlin Coroutines
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# Firebase
-keep class com.google.firebase.** { *; }
-keep class com.google.android.gms.** { *; }

# Keep SignLabel and related ML classes
-keep class com.lsec.translator.ml.** { *; }

# R8 missing classes — generated automatically
-dontwarn com.google.mediapipe.proto.CalculatorProfileProto$CalculatorProfile
-dontwarn com.google.mediapipe.proto.GraphTemplateProto$CalculatorGraphTemplate
-dontwarn javax.lang.model.SourceVersion
-dontwarn javax.lang.model.element.Element
-dontwarn javax.lang.model.element.ElementKind
-dontwarn javax.lang.model.element.Modifier
-dontwarn javax.lang.model.type.TypeMirror
-dontwarn javax.lang.model.type.TypeVisitor
-dontwarn javax.lang.model.util.SimpleTypeVisitor8
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options$GpuBackend
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options

# ONNX Runtime
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Moshi / Retrofit
-keepattributes *Annotation*
-keep class kotlin.reflect.jvm.internal.** { *; }
-keepclassmembers class * {
    @com.squareup.moshi.FromJson <methods>;
    @com.squareup.moshi.ToJson <methods>;
}
-keep class com.lsec.translator.data.remote.** { *; }

# DataStore (Preferences proto)
-keep class androidx.datastore.*.** { *; }
-keepclassmembers class * extends androidx.datastore.preferences.core.PreferenceKey { *; }
