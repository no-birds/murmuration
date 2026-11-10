// Example module: compiled and tested on the declared minimum (the `kotlin` convention), never published
// (the `library` convention, which adds publication, is deliberately not applied).
plugins {
    id("org.nobirds.murmuration.kotlin")
}

description = "Runnable examples for Murmuration. Not published."

dependencies {
    implementation(project(":murmuration"))
    implementation(libs.kotlinx.coroutines.core)

    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}
