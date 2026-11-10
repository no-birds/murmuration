package org.nobirds.murmuration.internal

import java.security.MessageDigest

internal fun ByteArray.sha256(): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val hash = digest.digest(this)
    return hash.fold("") { str, it -> str + "%02x".format(it) }
}

internal fun String.sha256(): String = toByteArray().sha256()
