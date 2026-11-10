rootProject.name = "murmuration-build"

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
    includeBuild("build-logic")
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

include(":murmuration", ":murmuration-gemini", ":murmuration-openai", ":murmuration-anthropic", ":murmuration-samples")
