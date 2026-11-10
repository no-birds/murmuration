package org.nobirds.murmuration.gemini.internal

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.security.MessageDigest

internal inline fun <reified T : Any> T.logger(): Logger = LoggerFactory.getLogger(T::class.java)

internal fun ByteArray.sha256(): ByteArray =
    MessageDigest.getInstance("SHA-256").digest(this)

internal fun String.sha256(): String =
    toByteArray(Charsets.UTF_8).sha256().joinToString("") { "%02x".format(it) }

private fun getMimeTypeByExtension(extension: String): String = when (extension.lowercase()) {
    "txt", "md" -> "text/plain"
    "gif" -> "image/gif"
    "jpg", "jpeg" -> "image/jpeg"
    "png" -> "image/png"
    "webp" -> "image/webp"
    "svg" -> "image/svg+xml"
    "bmp" -> "image/bmp"
    "pdf" -> "application/pdf"
    "mp3" -> "audio/mpeg"
    else -> "text/plain"
}

internal fun getMimeTypeFromUrl(url: String): String {
    val path = try {
        java.net.URL(url).path
    } catch (e: Exception) {
        url
    }
    val extension = path.substringAfterLast('.', "")
    return getMimeTypeByExtension(extension)
}
