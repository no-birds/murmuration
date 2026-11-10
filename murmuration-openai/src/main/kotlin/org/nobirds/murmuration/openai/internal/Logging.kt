package org.nobirds.murmuration.openai.internal

import org.slf4j.Logger
import org.slf4j.LoggerFactory

/** SLF4J logger bound to the receiver's runtime class — mirror of the gemini module helper. */
internal inline fun <reified T : Any> T.logger(): Logger = LoggerFactory.getLogger(T::class.java)
