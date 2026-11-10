package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Storage for binary payloads (image generation output, audio, user-uploaded inline data).
 *
 * The framework hands raw bytes to the storage and gets back an opaque [MessagePart.File]
 * reference — a URI string with metadata. The URI is provider-specific (`https://…`,
 * `gs://…`, `file://…`, etc.); consumers should treat it as opaque and ask the same
 * storage (or another resolver) when they need to read the bytes back.
 *
 * Wired into prompt executors via [withBinaryDataStorage]: every
 * [PromptExecutionEvent.PartReceived] carrying a [MessagePart.InlineData] is intercepted,
 * the bytes are persisted via [save], and a new [PromptExecutionEvent.PartReceived] with
 * the resulting [MessagePart.File] is emitted in its place. Downstream consumers (the
 * aggregator, image agents) then naturally see file references and never have to know that
 * a raw blob ever existed. The request side (binary parts that the **caller** put into the
 * prompt) is handled the same way before the prompt is sent.
 */
fun interface BinaryDataStorage {
    suspend fun save(bytes: ByteArray, mimeType: String, name: String?): MessagePart.File
}

/**
 * Wrap [PromptExecutor] so that inline-binary content in the **response** gets persisted by
 * [storage] on the fly: every [PromptExecutionEvent.PartReceived] envelope carrying a
 * [MessagePart.InlineData] is mapped to a `PartReceived(MessagePart.File)` containing the
 * URI the storage assigned. The aggregator then materialises an
 * [ImagePromptExecutionResult.File] (instead of `.Bytes`) and the rest of the pipeline
 * works with the URI — downstream agents that need the image as input can reference it via
 * `fileUri(...)` instead of re-uploading bytes.
 *
 * Other events (`Statistics`/`Cache`/`Completed`/`Failed`/text/thought/tool-call
 * `PartReceived`) pass through unchanged. Request-side inline binaries aren't a concern
 * here — by the time a prompt reaches the executor it has already been converted via
 * [PromptContent.toMessage], which drops raw `DataPromptPart`s.
 */
fun PromptExecutor.withBinaryDataStorage(storage: BinaryDataStorage): PromptExecutor =
    PromptExecutor { prompt ->
        execute(prompt).mapInlineDataToFile(storage)
    }

private fun Flow<PromptExecutionEvent>.mapInlineDataToFile(
    storage: BinaryDataStorage,
): Flow<PromptExecutionEvent> = map { event ->
    if (event !is PromptExecutionEvent.PartReceived) return@map event
    val part = event.part
    if (part !is MessagePart.InlineData) return@map event
    val payload = part.data ?: return@map event
    val file = storage.save(payload, part.mimeType, part.name)
    PromptExecutionEvent.PartReceived(file)
}
