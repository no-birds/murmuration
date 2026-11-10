plugins {
    id("org.nobirds.murmuration.library")
}

description = "Anthropic provider for Murmuration, built on the official Anthropic Java SDK."

dependencies {
    api(project(":murmuration"))

    api(libs.anthropic.java)

    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.slf4j.api)

    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.wiremock)
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    // Poison values: tests prove the explicit configuration wins and the environment never selects the way of signing in.
    environment("ANTHROPIC_API_KEY", "env-api-key-must-not-be-used")
    environment("ANTHROPIC_AUTH_TOKEN", "env-auth-token-must-not-be-used")
    environment("ANTHROPIC_WORKSPACE_ID", "wrkspc_env_must_not_be_used")
    environment("ANTHROPIC_LOG", "debug")
    environment("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
}
