package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.ImageResolution
import kotlin.test.assertEquals

/**
 * Mapping [ImageResolution] → Gemini `imageSize` string. The Gemini image API accepts only
 * `1K/2K/4K`; `512` is invalid and crashes the pipeline with a 400 error (prod incident 2026-07-18:
 * the character token went out with SD_512 → "512" → fatal character creation). SD_512 is clamped to 1K.
 */
class GeminiImageResolutionMappingTest {

    @Test
    fun `SD_512 clamps to 1K (never the invalid 512)`() {
        assertEquals("1K", ImageResolution.SD_512.toGemini())
    }

    @Test
    fun `larger resolutions map straight through`() {
        assertEquals("1K", ImageResolution.HD_1024.toGemini())
        assertEquals("2K", ImageResolution.XHD_2048.toGemini())
        assertEquals("4K", ImageResolution.UHD_4096.toGemini())
    }
}
