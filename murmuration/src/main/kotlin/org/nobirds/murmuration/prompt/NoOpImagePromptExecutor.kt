package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded

/**
 * Stub image-generating [PromptExecutor] for tests: returns a 1×1 transparent PNG as an
 * inline-data part plus minimal statistics and `Completed`. Doesn't make API calls.
 *
 * Fires on any prompt — filtering by `prompt.settings is ImagePromptSettings` is left to
 * DI which decides where to wire this stub.
 */
object NoOpImagePromptExecutor : PromptExecutor {

    private val EMPTY_PNG: ByteArray = byteArrayOf(
        0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4.toByte(), 0x89.toByte(),
        0x00, 0x00, 0x00, 0x0B, 0x49, 0x44, 0x41, 0x54,
        0x78, 0x9C.toByte(), 0x62, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE2.toByte(), 0x21,
        0xBC.toByte(), 0x33,
        0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE.toByte(), 0x42, 0x60, 0x82.toByte()
    )

    private val EMPTY_PNG_PART = MessagePart.InlineData(
        mimeType = "image/png",
        sizeBytes = EMPTY_PNG.size.toLong(),
        data = EMPTY_PNG,
    )

    override fun execute(prompt: Prompt): Flow<PromptExecutionEvent> = flowOf(
        PromptStarted(modelName = "noop", requestedModelName = "noop", requestMessages = emptyList()),
        StreamResponded(servedModelName = "noop"),
        PartReceived(EMPTY_PNG_PART),
        Completed(),
    )
}
