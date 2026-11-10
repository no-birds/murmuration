plugins {
    alias(libs.plugins.axion.release)
}

scmVersion {
    // The highest reachable release tag, not the nearest one: the version is computed from tags only.
    useHighestVersion = true
    tag {
        prefix = "v"
    }
    // The version in Gradle is a plain release value: no -SNAPSHOT and no branch decoration (a pre-release or
    // SNAPSHOT version must not exist). The publish job builds BEFORE the tag is created, so the
    // version it publishes comes from `-Prelease.forceVersion=X` (computed once by `guard`).
    snapshotCreator { _, _ -> "" }
    branchVersionCreator.put(".*", "simple")
}

// Version comes from axion alone (tags `v<semver>`); no file and no -Pversion holds it.
// `-Prelease.forceVersion=X` is axion's own override, used by the release workflow and local gates.
version = scmVersion.version

subprojects {
    group = "org.nobirds.murmuration"
    version = rootProject.version
}
