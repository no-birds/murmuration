package org.nobirds.murmuration.prompt

import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * [InputTokenStatistics] has no constructor defaults; the only place a missing field is tolerated is
 * this surrogate, so dumps written before the cache-write classes existed still decode (missing = 0).
 */
@Serializable
internal class InputTokenStatisticsSurrogate(
    val total: Long = 0,
    val cached: Long = 0,
    val cacheWriteShort: Long = 0,
    val cacheWriteLong: Long = 0,
)

internal object InputTokenStatisticsSerializer : KSerializer<InputTokenStatistics> {
    override val descriptor: SerialDescriptor = InputTokenStatisticsSurrogate.serializer().descriptor

    override fun serialize(encoder: Encoder, value: InputTokenStatistics) = encoder.encodeSerializableValue(
        InputTokenStatisticsSurrogate.serializer(),
        InputTokenStatisticsSurrogate(value.total, value.cached, value.cacheWriteShort, value.cacheWriteLong),
    )

    override fun deserialize(decoder: Decoder): InputTokenStatistics =
        decoder.decodeSerializableValue(InputTokenStatisticsSurrogate.serializer()).let {
            InputTokenStatistics(it.total, it.cached, it.cacheWriteShort, it.cacheWriteLong)
        }
}
