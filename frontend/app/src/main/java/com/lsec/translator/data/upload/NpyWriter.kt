package com.lsec.translator.data.upload

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Serializa una matriz de landmarks a formato .npy v1.0 (compatible con
 * `numpy.save` para arrays float32 contiguos en C-order).
 *
 * Formato:
 *   - Header magic: 0x93 b'NUMPY' 0x01 0x00
 *   - Header length (uint16 little-endian)
 *   - Header dict (ASCII): {'descr': '<f4', 'fortran_order': False, 'shape': (T, F), }
 *   - Padding con espacios hasta alcanzar header_len (múltiplo de 64)
 *   - Datos float32 little-endian: T * F * 4 bytes
 *
 * El backend (`/api/v1/data/upload/sequence`) hace `np.load(BytesIO(...))` y
 * valida `shape == (T, 63)`. Para nuestras landmarks: F = 63 (21 joints × 3 coords).
 */
object NpyWriter {

    fun writeFloat32Matrix(rows: List<FloatArray>, columns: Int = 63): ByteArray {
        require(columns == 63) { "Solo se soportan 63 columnas (21 landmarks × xyz)" }
        require(rows.isNotEmpty()) { "La secuencia no puede estar vacía" }
        require(rows.all { it.size == columns }) {
            "Todas las filas deben tener $columns floats; encontrado ${rows.first().size}"
        }

        val t = rows.size
        val shapeTuple = "($t, $columns)"
        // Numpy's _filter_header hace split('},')[0] + '}' para extraer el dict;
        // por lo tanto la coma trailing después del shape tuple es obligatoria.
        val dictStr = "{'descr': '<f4', 'fortran_order': False, 'shape': $shapeTuple, }"

        val headerPrefix = byteArrayOf(
            0x93.toByte(),
            'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(),
            'P'.code.toByte(), 'Y'.code.toByte(),
            0x01.toByte(), 0x00.toByte()
        )

        val dictBytes = dictStr.toByteArray(Charsets.US_ASCII)
        // Layout: dict + spaces_padding + '\n' — coincide con numpy.lib.format
        // para que ast.literal_eval parse el dict en una sola línea.
        val totalHeaderLen = ((dictBytes.size + 63) / 64) * 64
        val padding = totalHeaderLen - dictBytes.size - 1 // -1 para el '\n'

        check(totalHeaderLen <= 65535) {
            "Header demasiado largo ($totalHeaderLen bytes); reduzca columnas o filas"
        }

        val dataBytes = t * columns * 4
        val out = ByteArray(headerPrefix.size + 2 + totalHeaderLen + dataBytes)
        var i = 0
        headerPrefix.copyInto(out, i); i += headerPrefix.size
        out[i++] = (totalHeaderLen and 0xFF).toByte()
        out[i++] = ((totalHeaderLen shr 8) and 0xFF).toByte()
        dictBytes.copyInto(out, i); i += dictBytes.size
        repeat(padding) { out[i++] = ' '.code.toByte() }
        out[i++] = '\n'.code.toByte()

        val buf = ByteBuffer.wrap(out, i, dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        val fb: FloatBuffer = buf.asFloatBuffer()
        for (row in rows) {
            fb.put(row)
        }
        return out
    }
}