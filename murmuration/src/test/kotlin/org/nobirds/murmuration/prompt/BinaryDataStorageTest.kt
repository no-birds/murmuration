package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class BinaryDataStorageTest {

    private class RecordingStorage : BinaryDataStorage {
        val saved = mutableListOf<Triple<Int, String, String?>>()
        override suspend fun save(bytes: ByteArray, mimeType: String, name: String?): MessagePart.File {
            saved += Triple(bytes.size, mimeType, name)
            return MessagePart.File(uri = "stored://${saved.size}", mimeType = mimeType, name = name)
        }
    }

    @Test
    fun `inline data event with bytes is replaced by file part event`() = runBlocking {
        val bytes = byteArrayOf(1, 2, 3, 4)
        val baseExecutor = PromptExecutor {
            flowOf<PromptExecutionEvent>(
                PromptExecutionEvent.PromptStarted("test", null, emptyList()),
                PromptExecutionEvent.PartReceived(
                    MessagePart.InlineData(
                        mimeType = "image/png",
                        sizeBytes = bytes.size.toLong(),
                        name = "portrait.png",
                        data = bytes,
                    )
                ),
                PromptExecutionEvent.Completed(),
            )
        }
        val storage = RecordingStorage()
        val events = baseExecutor.withBinaryDataStorage(storage).execute(emptyPrompt()).toList()

        // The pipeline downstream should never see the inline blob — only the file reference.
        val partEvents = events.filterIsInstance<PromptExecutionEvent.PartReceived>()
        assertEquals(1, partEvents.size)
        val filePart = partEvents.single().part
        assertTrue(filePart is MessagePart.File)
        assertEquals("stored://1", (filePart as MessagePart.File).uri)
        assertEquals("image/png", filePart.mimeType)
        assertEquals("portrait.png", filePart.name)

        assertEquals(1, storage.saved.size)
        assertEquals(Triple(4, "image/png", "portrait.png"), storage.saved.single())
    }

    @Test
    fun `inline data event without bytes is left untouched`() = runBlocking {
        val baseExecutor = PromptExecutor {
            flowOf<PromptExecutionEvent>(
                PromptExecutionEvent.PromptStarted("test", null, emptyList()),
                PromptExecutionEvent.PartReceived(
                    MessagePart.InlineData(mimeType = "image/png", sizeBytes = 100, data = null)
                ),
                PromptExecutionEvent.Completed(),
            )
        }
        val storage = RecordingStorage()
        val events = baseExecutor.withBinaryDataStorage(storage).execute(emptyPrompt()).toList()

        val partEvents = events.filterIsInstance<PromptExecutionEvent.PartReceived>()
        val part = partEvents.single().part
        assertTrue(part is MessagePart.InlineData, "data=null InlineData must stay as InlineData")
        assertEquals(0, storage.saved.size, "storage must NOT be called for null-data InlineData")
    }

    @Test
    fun `aggregate produces File variant when storage transforms InlineData`() = runBlocking {
        val bytes = byteArrayOf(7, 8, 9)
        val baseExecutor = PromptExecutor {
            flowOf<PromptExecutionEvent>(
                PromptExecutionEvent.PromptStarted(
                    modelName = "image-model",
                    requestedModelName = "image-model",
                    requestMessages = emptyList(),
                ),
                PromptExecutionEvent.PartReceived(
                    MessagePart.InlineData(
                        mimeType = "image/jpeg",
                        sizeBytes = bytes.size.toLong(),
                        data = bytes,
                    )
                ),
                PromptExecutionEvent.UsageDelta(
                    input = InputTokenStatistics(total = 12, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 100, thinking = 5),
                ),
                PromptExecutionEvent.Completed(finishReason = "STOP"),
            )
        }
        val storage = RecordingStorage()
        val result = baseExecutor.withBinaryDataStorage(storage).execute(emptyPrompt()).aggregate()

        assertTrue(result is ImagePromptExecutionResult.File)
        result as ImagePromptExecutionResult.File
        assertEquals("stored://1", result.uri)
        assertEquals("image/jpeg", result.mimeType)

        // Aggregator built statistics: both input and output summed from UsageDelta deltas.
        val stats = result.statistics
        assertEquals(12L, stats.tokens.input.total)
        assertEquals(100L, stats.tokens.output.total)
        assertEquals(5L, stats.tokens.output.thinking)
        assertEquals(112L, stats.tokens.totalTokens)
        assertEquals("STOP", stats.finishReason)

        // Aggregator built the responseMessage with a File part (not InlineData).
        val responseParts = stats.responseMessage?.parts
        assertNotNull(responseParts)
        val filePart = responseParts.filterIsInstance<MessagePart.File>().single()
        assertEquals("stored://1", filePart.uri)
        assertTrue(responseParts.none { it is MessagePart.InlineData })
    }

    @Test
    fun `aggregate produces Bytes variant when no storage in the chain`() = runBlocking {
        val bytes = byteArrayOf(7, 8, 9)
        val baseExecutor = PromptExecutor {
            flowOf<PromptExecutionEvent>(
                PromptExecutionEvent.PromptStarted(
                    modelName = "image-model",
                    requestedModelName = "image-model",
                    requestMessages = emptyList(),
                ),
                PromptExecutionEvent.PartReceived(
                    MessagePart.InlineData(
                        mimeType = "image/jpeg",
                        sizeBytes = bytes.size.toLong(),
                        data = bytes,
                    )
                ),
                PromptExecutionEvent.Completed(),
            )
        }
        val result = baseExecutor.execute(emptyPrompt()).aggregate()

        assertTrue(result is ImagePromptExecutionResult.Bytes)
        result as ImagePromptExecutionResult.Bytes
        assertEquals(bytes.toList(), result.imageBytes.toList())
        assertEquals("image/jpeg", result.mimeType)
    }

    @Test
    fun `aggregator sums output usage deltas from multiple chunks`() = runBlocking {
        val baseExecutor = PromptExecutor {
            flowOf<PromptExecutionEvent>(
                PromptExecutionEvent.PromptStarted(
                    modelName = "text-model",
                    requestedModelName = "text-model",
                    requestMessages = emptyList(),
                ),
                PromptExecutionEvent.PartReceived(MessagePart.Text("hello ")),
                PromptExecutionEvent.UsageDelta(
                    input = InputTokenStatistics(total = 50, cached = 10, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 2, thinking = 0),
                ),
                PromptExecutionEvent.PartReceived(MessagePart.Text("world")),
                PromptExecutionEvent.UsageDelta(output = OutputTokenStatistics(total = 1, thinking = 0)),
                PromptExecutionEvent.Completed(finishReason = "STOP"),
            )
        }
        val result = baseExecutor.execute(emptyPrompt()).aggregate()
        assertTrue(result is TextPromptExecutionResult)
        result as TextPromptExecutionResult
        assertEquals(3L, result.statistics.tokens.output.total)
        assertEquals(50L, result.statistics.tokens.input.total)
        assertEquals("hello world", result.text)
    }

    private fun emptyPrompt(): Prompt = Prompt(
        llm = org.nobirds.murmuration.llm.Llm(
            provider = "test", family = "test", name = "test",
            version = "1", stable = true, model = "test",
        ),
        settings = null,
        content = emptyList(),
    )
}
