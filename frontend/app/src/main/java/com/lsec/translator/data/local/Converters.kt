package com.lsec.translator.data.local

import androidx.room.TypeConverter

/**
 * Conversores de tipos para Room.
 */
class Converters {
    
    @TypeConverter
    fun fromStringList(value: List<String>?): String? {
        return value?.joinToString(",")
    }
    
    @TypeConverter
    fun toStringList(value: String?): List<String>? {
        return value?.split(",")?.map { it.trim() }
    }
}
