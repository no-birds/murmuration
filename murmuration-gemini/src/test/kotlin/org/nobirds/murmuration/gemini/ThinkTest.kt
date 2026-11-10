package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.GenerateContentConfig
import com.google.genai.types.ThinkingConfig
import kotlinx.coroutines.future.await
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    val c = Client.builder().apiKey(System.getenv("GEMINI_API_KEY")).build()
    val config = GenerateContentConfig.builder()
        .thinkingConfig(ThinkingConfig.builder().includeThoughts(true).build())
        .build()
    val stream = c.async.models.generateContentStream("gemini-2.5-flash", "test", config).await()
    for (chunk in stream) {
        println(chunk.text())
    }
}
