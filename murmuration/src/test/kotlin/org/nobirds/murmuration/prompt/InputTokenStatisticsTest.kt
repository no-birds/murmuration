package org.nobirds.murmuration.prompt

import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class InputTokenStatisticsTest {
    private val a = InputTokenStatistics(total = 100, cached = 10, cacheWriteShort = 20, cacheWriteLong = 30)
    private val b = InputTokenStatistics(total = 40, cached = 4, cacheWriteShort = 5, cacheWriteLong = 6)

    @Test
    fun `plus adds all four fields`() {
        assertEquals(InputTokenStatistics(140, 14, 25, 36), a + b)
    }

    @Test
    fun `minus subtracts all four fields`() {
        assertEquals(InputTokenStatistics(60, 6, 15, 24), a - b)
    }

    @Test
    fun `an old dump without write classes reads as zero`() {
        val old = Json.decodeFromString(InputTokenStatistics.serializer(), """{"total":7,"cached":3}""")
        assertEquals(InputTokenStatistics(total = 7, cached = 3, cacheWriteShort = 0, cacheWriteLong = 0), old)
        assertEquals(0L, old.cacheWriteShort)
        assertEquals(0L, old.cacheWriteLong)
    }

    @Test
    fun `write classes survive a serialization round trip`() {
        val json = Json { encodeDefaults = true }
        assertEquals(a, json.decodeFromString(InputTokenStatistics.serializer(), json.encodeToString(InputTokenStatistics.serializer(), a)))
    }
}
