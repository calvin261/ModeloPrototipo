package com.lsec.translator.di

import com.lsec.translator.BuildConfig
import com.lsec.translator.data.preferences.UserPreferences
import com.lsec.translator.data.remote.LSECApiService
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

/**
 * Modulo Hilt para proveer componentes de networking.
 */
@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideMoshi(): Moshi {
        return Moshi.Builder()
            .addLast(KotlinJsonAdapterFactory())
            .build()
    }

    @Provides
    @Singleton
    fun provideOkHttpClient(userPreferences: UserPreferences): OkHttpClient {
        val logging = HttpLoggingInterceptor().apply {
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.BODY
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        }

        // Interceptor que inyecta X-API-Key en cada petición.
        // Lee la clave guardada en cada request para que cambios surtan efecto sin reiniciar.
        val apiKeyInterceptor = Interceptor { chain ->
            val key = runBlocking { userPreferences.preferencesFlow.first().apiKey }
            val request = if (key.isNotBlank()) {
                chain.request().newBuilder()
                    .header("X-API-Key", key)
                    .build()
            } else {
                chain.request()
            }
            chain.proceed(request)
        }

        return OkHttpClient.Builder()
            .addInterceptor(apiKeyInterceptor)
            .addInterceptor(logging)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofit(
        okHttpClient: OkHttpClient,
        moshi: Moshi,
        userPreferences: UserPreferences
    ): Retrofit {
        // Lee la URL guardada por el usuario; si no tiene barra final, la agrega
        val savedUrl = runBlocking { userPreferences.preferencesFlow.first().serverUrl }
        val baseUrl = if (savedUrl.endsWith("/")) savedUrl else "$savedUrl/"

        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(MoshiConverterFactory.create(moshi))
            .build()
    }

    @Provides
    @Singleton
    fun provideLSECApiService(retrofit: Retrofit): LSECApiService {
        return retrofit.create(LSECApiService::class.java)
    }
}
