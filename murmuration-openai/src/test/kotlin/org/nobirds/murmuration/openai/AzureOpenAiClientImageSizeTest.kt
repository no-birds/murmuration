package org.nobirds.murmuration.openai

import com.openai.models.images.ImageEditParams
import com.openai.models.images.ImageGenerateParams
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.nobirds.murmuration.prompt.ImageAspectRatio
import org.nobirds.murmuration.prompt.ImageResolution
import org.nobirds.murmuration.prompt.ImageSize
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Table-driven checks for [ImageSize.toPixelDimensions] / [toGenerateQuality] / [toEditQuality] —
 * the mapping from the provider-neutral [ImageResolution] + [ImageAspectRatio] pair onto
 * gpt-image-2's `WIDTHxHEIGHT` size strings and quality tiers. See the KDoc on
 * [ImageSize.toPixelDimensions] for the formula these checkpoints exercise.
 */
class AzureOpenAiClientImageSizeTest {

    data class SizeCase(
        val resolution: ImageResolution,
        val aspectRatio: ImageAspectRatio,
        val expectedWidth: Int,
        val expectedHeight: Int,
    ) {
        override fun toString() = "$resolution+$aspectRatio"
    }

    companion object {
        @JvmStatic
        fun checkpoints() = listOf(
            // SD_512² (262,144 px) undershoots the 655,360 px minimum, so the target area clamps
            // up; sqrt(655360)/16 = 50.6 rounds up to k=51 -> 51*16 = 816.
            SizeCase(ImageResolution.SD_512, ImageAspectRatio.SQUARE_1_1, 816, 816),
            // HD_1024² (1,048,576 px) is in range, but the odd `9` in 9:16 forces k up to the next
            // multiple of 16 (96 instead of the raw ~85.3), overshooting the naive 768x1366.
            SizeCase(ImageResolution.HD_1024, ImageAspectRatio.PORTRAIT_9_16, 864, 1536),
            // XHD_2048² (4,194,304 px) is in range and 2048 is already a multiple of 16: no
            // overshoot, k lands exactly on 128.
            SizeCase(ImageResolution.XHD_2048, ImageAspectRatio.SQUARE_1_1, 2048, 2048),
            // UHD_4096² overshoots the 8,294,400 px maximum, so the target area clamps down to
            // exactly the max, which happens to divide out evenly for 16:9 -> 3840x2160.
            SizeCase(ImageResolution.UHD_4096, ImageAspectRatio.LANDSCAPE_16_9, 3840, 2160),
        )
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("checkpoints")
    fun `resolution and aspect ratio map to the expected WxH`(case: SizeCase) {
        val size = ImageSize(case.resolution, case.aspectRatio)
        val (width, height) = size.toPixelDimensions()
        assertEquals(case.expectedWidth, width, "width")
        assertEquals(case.expectedHeight, height, "height")
    }

    @Test
    fun `every mapped size lands within gpt-image-2's legal area and edge-multiple constraints`() {
        for (resolution in ImageResolution.entries) {
            for (aspectRatio in ImageAspectRatio.entries) {
                val (width, height) = ImageSize(resolution, aspectRatio).toPixelDimensions()
                val area = width.toLong() * height.toLong()

                assertEquals(0, width % SIZE_GRID.toInt(), "width not a multiple of 16 for $resolution+$aspectRatio")
                assertEquals(0, height % SIZE_GRID.toInt(), "height not a multiple of 16 for $resolution+$aspectRatio")
                assertTrue(
                    area in MIN_IMAGE_AREA_PX..MAX_IMAGE_AREA_PX,
                    "area $area out of [$MIN_IMAGE_AREA_PX, $MAX_IMAGE_AREA_PX] for $resolution+$aspectRatio",
                )

                // gpt-image-2 requires the aspect ratio between 1:3 and 3:1. Some enum entries
                // (PORTRAIT_8_1 / LANDSCAPE_1_8, ratio 8:1) fall well outside that — this is
                // exactly the case reducedClampedRatio snaps to the nearest legal boundary.
                val longEdge = maxOf(width, height).toDouble()
                val shortEdge = minOf(width, height).toDouble()
                assertTrue(
                    longEdge / shortEdge <= 3.0 + 1e-9,
                    "aspect ratio ${width}x$height exceeds 3:1 for $resolution+$aspectRatio",
                )
            }
        }
    }

    @Test
    fun `extreme 8_1 aspect ratio clamps to 3_1 instead of the raw 8_1`() {
        // PORTRAIT_8_1(8, 1) has widthRatio=8 > heightRatio=1 (the enum's numbers, not its name,
        // define the shape — see ImageAspectRatio's KDoc), i.e. it's nowhere near gpt-image-2's
        // [1:3, 3:1] window on the wide side. Verify it snaps to the 3:1 boundary rather than
        // being passed through (which the API would reject) or silently ignored (wrong shape).
        val (width, height) = ImageSize(ImageResolution.HD_1024, ImageAspectRatio.PORTRAIT_8_1).toPixelDimensions()
        assertEquals(3.0, width.toDouble() / height)
    }

    @Test
    fun `quality tier follows resolution, not a fixed constant`() {
        assertEquals(ImageGenerateParams.Quality.LOW, ImageResolution.SD_512.toGenerateQuality())
        assertEquals(ImageGenerateParams.Quality.MEDIUM, ImageResolution.HD_1024.toGenerateQuality())
        assertEquals(ImageGenerateParams.Quality.HIGH, ImageResolution.XHD_2048.toGenerateQuality())
        assertEquals(ImageGenerateParams.Quality.HIGH, ImageResolution.UHD_4096.toGenerateQuality())

        assertEquals(ImageEditParams.Quality.LOW, ImageResolution.SD_512.toEditQuality())
        assertEquals(ImageEditParams.Quality.MEDIUM, ImageResolution.HD_1024.toEditQuality())
        assertEquals(ImageEditParams.Quality.HIGH, ImageResolution.XHD_2048.toEditQuality())
        assertEquals(ImageEditParams.Quality.HIGH, ImageResolution.UHD_4096.toEditQuality())
    }

    @Test
    fun `generate and edit size mappings agree on the same WxH string`() {
        val size = ImageSize(ImageResolution.HD_1024, ImageAspectRatio.PORTRAIT_3_2)
        assertEquals(size.toImageSize().toString(), size.toImageEditSize().toString())
    }
}
