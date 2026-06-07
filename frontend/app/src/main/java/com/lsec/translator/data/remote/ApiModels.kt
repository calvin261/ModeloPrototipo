package com.lsec.translator.data.remote

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

// === Data Upload ===

@JsonClass(generateAdapter = true)
data class LandmarkSampleDto(
    @Json(name = "label") val label: String,
    @Json(name = "landmarks") val landmarks: List<Float>
)

@JsonClass(generateAdapter = true)
data class StaticLandmarkUploadDto(
    @Json(name = "samples") val samples: List<LandmarkSampleDto>
)

@JsonClass(generateAdapter = true)
data class UploadResponseDto(
    @Json(name = "status") val status: String,
    @Json(name = "samples_added") val samplesAdded: Int,
    @Json(name = "total_samples") val totalSamples: Int
)

@JsonClass(generateAdapter = true)
data class SequenceUploadResponseDto(
    @Json(name = "status") val status: String,
    @Json(name = "label") val label: String,
    @Json(name = "frames") val frames: Int
)

@JsonClass(generateAdapter = true)
data class DataStatsResponseDto(
    @Json(name = "static_samples") val staticSamples: Int,
    @Json(name = "sequence_samples") val sequenceSamples: Int,
    @Json(name = "static_labels") val staticLabels: Map<String, Int>,
    @Json(name = "sequence_labels") val sequenceLabels: Map<String, Int>
)

// === Training ===

@JsonClass(generateAdapter = true)
data class TrainingParamsDto(
    @Json(name = "epochs") val epochs: Int = 50,
    @Json(name = "learning_rate") val learningRate: Float = 0.001f,
    @Json(name = "batch_size") val batchSize: Int = 64
)

@JsonClass(generateAdapter = true)
data class TrainingStartResponseDto(
    @Json(name = "task_id") val taskId: String,
    @Json(name = "status") val status: String
)

@JsonClass(generateAdapter = true)
data class TrainingStatusResponseDto(
    @Json(name = "task_id") val taskId: String,
    @Json(name = "status") val status: String,
    @Json(name = "epoch") val epoch: Int = 0,
    @Json(name = "total_epochs") val totalEpochs: Int = 0,
    @Json(name = "train_loss") val trainLoss: Float = 0f,
    @Json(name = "val_accuracy") val valAccuracy: Float = 0f,
    @Json(name = "best_accuracy") val bestAccuracy: Float = 0f,
    @Json(name = "error") val error: String? = null
)

@JsonClass(generateAdapter = true)
data class TrainingHistoryEntryDto(
    @Json(name = "task_id") val taskId: String,
    @Json(name = "model_type") val modelType: String,
    @Json(name = "status") val status: String,
    @Json(name = "best_accuracy") val bestAccuracy: Float,
    @Json(name = "epochs") val epochs: Int,
    @Json(name = "started_at") val startedAt: String,
    @Json(name = "completed_at") val completedAt: String? = null
)

// === Models ===

@JsonClass(generateAdapter = true)
data class ModelLatestResponseDto(
    @Json(name = "version") val version: String,
    @Json(name = "download_url") val downloadUrl: String,
    @Json(name = "size_kb") val sizeKb: Float,
    @Json(name = "accuracy") val accuracy: Float,
    @Json(name = "num_classes") val numClasses: Int
)

@JsonClass(generateAdapter = true)
data class ModelInfoDto(
    @Json(name = "name") val name: String,
    @Json(name = "version") val version: String,
    @Json(name = "accuracy") val accuracy: Float,
    @Json(name = "size_kb") val sizeKb: Float,
    @Json(name = "format") val format: String,
    @Json(name = "num_classes") val numClasses: Int,
    @Json(name = "created_at") val createdAt: String
)
