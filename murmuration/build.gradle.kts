plugins {
    id("org.nobirds.murmuration.library")
}

description = "Core of Murmuration: a provider-neutral Kotlin/JVM framework for LLM agents, with composable prompts, tool calling, structured output schemas and retries."

dependencies {
    api(libs.kotlinx.serialization.json)
    api(libs.kotlinx.coroutines.core)
    api(libs.slf4j.api)
    implementation(kotlin("reflect", kotlin.coreLibrariesVersion))

    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.logback.classic)
    testRuntimeOnly(libs.junit.platform.launcher)
}
