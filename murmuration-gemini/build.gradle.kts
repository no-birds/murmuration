plugins {
    id("org.nobirds.murmuration.library")
}

description = "Google Gemini and Vertex AI provider for Murmuration: prompt executor with streaming, context caching and a Gemini-specific retry strategy."

dependencies {
    api(project(":murmuration"))

    api(libs.google.genai)

    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.slf4j.api)
    implementation(kotlin("reflect", kotlin.coreLibrariesVersion))

    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.mockk)
    testRuntimeOnly(libs.junit.platform.launcher)
}
