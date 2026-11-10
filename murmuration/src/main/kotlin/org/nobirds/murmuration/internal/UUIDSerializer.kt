package org.nobirds.murmuration.internal

import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import java.util.UUID

internal object UUIDSerializer : KSerializer<UUID> {
    /**
     * Distinct from the bare `"UUID"` other serializers use, so the schema walker can recognise
     * exactly this serializer and attach `format: uuid` without touching third-party UUID
     * descriptors.
     */
    const val SERIAL_NAME: String = "java.util.UUID"

    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor(SERIAL_NAME, PrimitiveKind.STRING)

    override fun serialize(encoder: Encoder, value: UUID) {
        encoder.encodeString(value.toString())
    }

    override fun deserialize(decoder: Decoder): UUID {
        return UUID.fromString(decoder.decodeString())
    }
}
