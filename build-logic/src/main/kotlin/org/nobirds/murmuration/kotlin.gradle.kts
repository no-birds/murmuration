package org.nobirds.murmuration

import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion

// Compatibility convention: the declared minimum for consumers lives here and nowhere else.
// Applied to published modules (through `library`) and to non-published ones alike.
plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
}

val minJvm = 17
// Kotlin 2.3, not 2.2: kotlinx-serialization 1.12 and coroutines 1.11 (and the stdlib they pull) carry metadata 2.4.0;
// a compiler reads metadata up to one minor ahead of itself, so the oldest consumer compiler is 2.3 (2.2 fails, verified).
val minKotlin = KotlinVersion.KOTLIN_2_3
// Latest 2.3.x patch: the standard library the published modules are built and tested against.
val coreLibraries = "2.3.21"

repositories {
    mavenCentral()
}

kotlin {
    // The toolchain compiles; the consumer-visible minimum is -Xjdk-release, not the toolchain.
    jvmToolchain(21)
    // Default version of kotlin(...) dependencies declared without a version (stdlib, reflect):
    // bound to the declared minimum, not to the compiler version.
    coreLibrariesVersion = coreLibraries
    compilerOptions {
        jvmTarget = JvmTarget.fromTarget(minJvm.toString())
        languageVersion = minKotlin
        apiVersion = minKotlin
        freeCompilerArgs.addAll("-opt-in=kotlin.time.ExperimentalTime", "-Xjdk-release=$minJvm")
    }
}

java {
    sourceCompatibility = JavaVersion.toVersion(minJvm)
    targetCompatibility = JavaVersion.toVersion(minJvm)
}

tasks.test {
    useJUnitPlatform()
    // Compatibility matrix: -Pmurmuration.testJvm=17|21 runs tests on that JVM (the toolchain stays 21 for compilation).
    providers.gradleProperty("murmuration.testJvm").orNull?.let { jvm ->
        javaLauncher = javaToolchains.launcherFor {
            languageVersion = JavaLanguageVersion.of(jvm.toInt())
        }
    }
}

tasks.register("printCompatibility") {
    group = "help"
    description = "Prints the declared minimum (JVM, Kotlin language/API, core libraries)."
    val lines = listOf("JVM: $minJvm", "Kotlin: ${minKotlin.version}", "Kotlin standard library: $coreLibraries")
    doLast { lines.forEach(::println) }
}
