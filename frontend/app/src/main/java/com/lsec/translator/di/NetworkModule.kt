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
        // Prioridad: (1) clave guardada por el usuario en Settings, (2) clave
        // embebida en BuildConfig.BUNDLED_API_KEY (release builds), (3) ninguna.
        // Se registra DESPUES del logging para que la cabecera no aparezca en logs.
        val apiKeyInterceptor = Interceptor { chain ->
            val userKey = runBlocking { userPreferences.preferencesFlow.first().apiKey }
            val effectiveKey = userKey.ifBlank { BuildConfig.BUNDLED_API_KEY }
            val request = if (effectiveKey.isNotBlank()) {
                chain.request().newBuilder()
                    .header("X-API-Key", effectiveKey)
                    .build()
            } else {
                chain.request()
            }
            chain.proceed(request)
        }

        return OkHttpClient.Builder()
            .addInterceptor(logging)
            .addInterceptor(apiKeyInterceptor)
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
