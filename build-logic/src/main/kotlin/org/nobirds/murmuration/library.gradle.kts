package org.nobirds.murmuration

import com.vanniktech.maven.publish.JavadocJar
import com.vanniktech.maven.publish.KotlinJvm

// Published library = `kotlin` convention + Maven Central publication.
// Group and version are set by the root project (version from axion).
plugins {
    id("org.nobirds.murmuration.kotlin")
    id("org.jetbrains.dokka")
    id("com.vanniktech.maven.publish")
}

// SHA of the commit being built, written to POM scm.tag (V-9).
// Order: explicit property, then GITHUB_SHA, then the local checkout.
val buildSha: Provider<String> = providers.gradleProperty("murmuration.buildSha")
    .orElse(providers.environmentVariable("GITHUB_SHA"))
    .orElse(
        providers.exec {
            commandLine("git", "rev-parse", "HEAD")
            workingDir = rootProject.projectDir
        }.standardOutput.asText.map { it.trim() }
    )

mavenPublishing {
    configure(KotlinJvm(javadocJar = JavadocJar.Dokka("dokkaGeneratePublicationHtml"), sourcesJar = true))

    pom {
        name = project.name
        description = project.provider { project.description ?: error("${project.path}: description is required for the POM") }
        url = "https://github.com/no-birds/murmuration"
        licenses {
            license {
                name = "The Apache License, Version 2.0"
                url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
            }
        }
        developers {
            developer {
                id = "boris.vanin"
                name = "Boris Vanin"
                email = "boris.vanin@nobirds.org"
            }
        }
        scm {
            url = "https://github.com/no-birds/murmuration"
            connection = "scm:git:git://github.com/no-birds/murmuration.git"
            developerConnection = "scm:git:ssh://git@github.com/no-birds/murmuration.git"
            tag = buildSha
        }
    }

    // Without a key (a local run) signing is skipped; CI passes ORG_GRADLE_PROJECT_signingInMemoryKey*.
    if (providers.gradleProperty("signingInMemoryKey").isPresent) {
        signAllPublications()
    }

    // The Central repository exists only on GitHub Actions: a laptop has neither credentials nor the right to upload.
    // Mode is set by gradle properties: mavenCentralAutomaticPublishing, mavenCentralDeploymentValidation.
    if (System.getenv("GITHUB_ACTIONS") == "true") {
        publishToMavenCentral()
    }
}

// Staging repository for local gates and the `stage` job.
providers.gradleProperty("murmuration.testRepo").orNull?.let { dir ->
    publishing {
        repositories {
            maven(uri("file://" + File(dir).absolutePath)) {
                name = "Test"
            }
        }
    }
}

tasks.withType<PublishToMavenLocal>().configureEach {
    enabled = false
}
