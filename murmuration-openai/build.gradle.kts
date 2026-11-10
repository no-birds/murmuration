plugins {
    id("org.nobirds.murmuration.library")
}

description = "OpenAI provider for Murmuration, built on the OpenAI Java SDK, with Azure Identity for authentication."

dependencies {
    api(project(":murmuration"))

    api(libs.openai.java)
    api(libs.azure.identity)

    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.slf4j.api)

    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.wiremock)
    testRuntimeOnly(libs.junit.platform.launcher)
}
