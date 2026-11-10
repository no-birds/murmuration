#!/usr/bin/env python3 -I
"""Check of the publication against a file:// repository.

Usage: python3 -I check-poms.py <testRepoDir> [<expectedBuildSha>]
Checks: 4 artifacts x (.pom .module .jar -sources.jar -javadoc.jar), each with a .asc signature file;
(a) every <dependency> in all POMs has a <version>;
(b) dependencies on murmuration modules use org.nobirds.murmuration:* coordinates;
(c) the .module has no project references (project links instead of coordinates);
(d) every POM has name, description, url, Apache-2.0 license, developer Boris Vanin, scm;
(e) the -javadoc.jar holds Dokka HTML (index.html and a non-trivial number of html files);
(f) bytecode major == 61 (Java 17) in every class of the main jar, org.gradle.jvm.version == 17 in the .module;
(i) when <expectedBuildSha> is given, scm.tag of every POM equals it.
Signature validity (gpg --verify) is checked by scripts/stage-publish.sh; here only presence.
"""
import json
import struct
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

GROUP = "org.nobirds.murmuration"
ARTIFACTS = ["murmuration", "murmuration-gemini", "murmuration-openai", "murmuration-anthropic"]
NS = "{http://maven.apache.org/POM/4.0.0}"
JAVA_17_MAJOR = 61
JVM_MINIMUM = 17
MIN_JAVADOC_HTML = 10
SUFFIXES = (".pom", ".module", ".jar", "-sources.jar", "-javadoc.jar")


def find_files(repo: Path, name: str, suffix: str):
    return sorted(repo.joinpath(*GROUP.split(".")).joinpath(name).rglob(f"*{suffix}"))


def main_jar(repo: Path, a: str):
    return [f for f in find_files(repo, a, ".jar") if not f.name.endswith(("-sources.jar", "-javadoc.jar"))]


def check_pom_fields(pom: Path, expected_sha, errors):
    root = ET.parse(pom).getroot()
    for field in ("name", "description", "url"):
        if not (root.findtext(f"{NS}{field}") or "").strip():
            errors.append(f"{pom.name}: <{field}> missing or empty (d)")
    licenses = [(lic.findtext(f"{NS}name") or "") for lic in root.iter(f"{NS}license")]
    if not any("Apache License, Version 2.0" in n for n in licenses):
        errors.append(f"{pom.name}: license is not Apache-2.0: {licenses} (d)")
    developers = [(d.findtext(f"{NS}name") or "") for d in root.iter(f"{NS}developer")]
    if "Boris Vanin" not in developers:
        errors.append(f"{pom.name}: developer Boris Vanin missing: {developers} (d)")
    scm = root.find(f"{NS}scm")
    if scm is None or not (scm.findtext(f"{NS}url") or "").strip() or not (scm.findtext(f"{NS}connection") or "").strip():
        errors.append(f"{pom.name}: <scm> missing url/connection (d)")
    elif expected_sha is not None and scm.findtext(f"{NS}tag") != expected_sha:
        errors.append(f"{pom.name}: scm.tag is {scm.findtext(f'{NS}tag')}, expected {expected_sha} (i)")


def check_javadoc(jar: Path, errors):
    with zipfile.ZipFile(jar) as z:
        names = z.namelist()
    html = [n for n in names if n.endswith(".html")]
    if "index.html" not in names or len(html) < MIN_JAVADOC_HTML:
        errors.append(f"{jar.name}: no Dokka HTML (index.html present: {'index.html' in names}, html files: {len(html)}) (e)")


def check_bytecode(jar: Path, errors):
    majors = set()
    with zipfile.ZipFile(jar) as z:
        for n in z.namelist():
            if n.endswith(".class") and not n.startswith("META-INF/"):
                head = z.read(n)[:8]
                majors.add(struct.unpack(">H", head[6:8])[0])
    if majors != {JAVA_17_MAJOR}:
        errors.append(f"{jar.name}: class major versions {sorted(majors)}, expected only {JAVA_17_MAJOR} (f)")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: check-poms.py <testRepoDir> [<expectedBuildSha>]")
        return 2
    repo = Path(sys.argv[1])
    expected_sha = sys.argv[2] if len(sys.argv) == 3 else None
    errors = []
    for a in ARTIFACTS:
        base = repo.joinpath(*GROUP.split(".")) / a
        for suffix, label in ((".pom", "pom"), (".module", "module")):
            if len(find_files(repo, a, suffix)) != 1:
                errors.append(f"{a}: expected exactly 1 {label} in {base}")
        jars = main_jar(repo, a)
        if len(jars) != 1:
            errors.append(f"{a}: expected exactly 1 main jar, found {len(jars)}")
        sources = list(base.rglob("*-sources.jar"))
        javadocs = list(base.rglob("*-javadoc.jar"))
        if len(sources) != 1:
            errors.append(f"{a}: expected exactly 1 -sources.jar")
        if len(javadocs) != 1:
            errors.append(f"{a}: expected exactly 1 -javadoc.jar")
        for s in SUFFIXES:
            for f in find_files(repo, a, s):
                if not f.with_name(f.name + ".asc").is_file():
                    errors.append(f"{f.name}: signature .asc missing")
        for jar in javadocs:
            check_javadoc(jar, errors)
        for jar in jars:
            check_bytecode(jar, errors)

        for pom in find_files(repo, a, ".pom"):
            check_pom_fields(pom, expected_sha, errors)
            root = ET.parse(pom).getroot()
            for dep in root.iter(f"{NS}dependency"):
                g = dep.findtext(f"{NS}groupId")
                art = dep.findtext(f"{NS}artifactId")
                ver = dep.findtext(f"{NS}version")
                if not ver:
                    errors.append(f"{pom.name}: dependency {g}:{art} without <version> (a)")
                if art in ARTIFACTS and (g != GROUP or not ver):
                    errors.append(f"{pom.name}: dependency on {art} is not of the form {GROUP}:{art}:<version> (b): {g}:{art}:{ver}")

        for mod in find_files(repo, a, ".module"):
            data = json.loads(mod.read_text())
            for variant in data.get("variants", []):
                for dep in variant.get("dependencies", []):
                    if "project" in json.dumps(dep).lower() and ("project" in dep or dep.get("module") is None):
                        errors.append(f"{mod.name}: project reference in {variant.get('name')}: {dep} (c)")
                    if dep.get("group") == GROUP and not dep.get("version", {}).get("requires"):
                        errors.append(f"{mod.name}: {dep.get('module')} without a version (b)")
                jvm = variant.get("attributes", {}).get("org.gradle.jvm.version")
                if variant.get("name") in ("apiElements", "runtimeElements") and jvm != JVM_MINIMUM:
                    errors.append(f"{mod.name}: {variant.get('name')} org.gradle.jvm.version={jvm}, expected {JVM_MINIMUM} (f)")
            if '"project"' in mod.read_text().replace(" ", ""):
                errors.append(f"{mod.name}: the .module text contains a project key (c)")

    if errors:
        print("FAIL:")
        for e in dict.fromkeys(errors):
            print(" -", e)
        return 1
    sha_note = ", scm.tag == build SHA" if expected_sha else ""
    print(f"OK: {len(ARTIFACTS)} artifacts, 5 files each with .asc, POM fields, Dokka javadoc, bytecode 61, JVM 17{sha_note}, versions and coordinates correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
