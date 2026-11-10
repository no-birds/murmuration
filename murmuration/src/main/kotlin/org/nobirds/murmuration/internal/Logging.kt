package org.nobirds.murmuration.internal

import org.slf4j.Logger
import org.slf4j.LoggerFactory

internal inline fun <reified T> T.logger(): Logger = LoggerFactory.getLogger(T::class.java)
internal inline fun <reified T> logger(): Logger = LoggerFactory.getLogger(T::class.java)
