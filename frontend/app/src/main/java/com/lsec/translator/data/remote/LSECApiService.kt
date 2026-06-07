package com.lsec.translator.data.remote

import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*

/**
 * Retrofit interface para el backend FastAPI de LSEC.
 */
interface LSECApiService {

    // === Data Upload ===

    @POST("api/v1/data/upload/static")
    suspend fun uploadStaticLandmarks(
        @Body data: StaticLandmarkUploadDto
    ): Response<UploadResponseDto>

    @Multipart
    @POST("api/v1/data/upload/sequence")
    suspend fun uploadSequence(
        @Part file: MultipartBody.Part,
        @Part("label") label: String
    ): Response<SequenceUploadResponseDto>

    @GET("api/v1/data/stats")
    suspend fun getDataStats(): Response<DataStatsResponseDto>

    @DELETE("api/v1/data/static/{label}")
    suspend fun deleteStaticLabel(
        @Path("label") label: String
    ): Response<Unit>

    @DELETE("api/v1/data/sequence/{label}")
    suspend fun deleteSequenceLabel(
        @Path("label") label: String
    ): Response<Unit>

    // === Training ===

    @POST("api/v1/train/classifier")
    suspend fun startClassifierTraining(
        @Body params: TrainingParamsDto
    ): Response<TrainingStartResponseDto>

    @POST("api/v1/train/sequence")
    suspend fun startSequenceTraining(
        @Body params: TrainingParamsDto
    ): Response<TrainingStartResponseDto>

    @GET("api/v1/train/status/{taskId}")
    suspend fun getTrainingStatus(
        @Path("taskId") taskId: String
    ): Response<TrainingStatusResponseDto>

    @GET("api/v1/train/history")
    suspend fun getTrainingHistory(): Response<List<TrainingHistoryEntryDto>>

    // === Models ===

    @GET("api/v1/models/list")
    suspend fun listModels(): Response<List<ModelInfoDto>>

    @GET("api/v1/models/latest/{modelType}")
    suspend fun getLatestModel(
        @Path("modelType") modelType: String
    ): Response<ModelLatestResponseDto>

    @GET("api/v1/models/download/{modelType}/{version}")
    @Streaming
    suspend fun downloadModel(
        @Path("modelType") modelType: String,
        @Path("version") version: String
    ): Response<ResponseBody>

    @GET("api/v1/models/labels/{modelType}")
    suspend fun getLabels(
        @Path("modelType") modelType: String
    ): Response<Map<String, Int>>

    @POST("api/v1/models/export/{modelType}")
    suspend fun exportModel(
        @Path("modelType") modelType: String
    ): Response<Unit>
}
