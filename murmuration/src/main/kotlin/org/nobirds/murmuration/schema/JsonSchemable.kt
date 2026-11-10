package org.nobirds.murmuration.schema

/**
 * Marker for a type shown to the model as a JSON Schema (a *display* schema), via
 * [inputSchema] / [jsonWithSchema]. The bound on those builders means only marked types can
 * reach the model as schema documentation, which makes the set of display types reflectively
 * enumerable: a CI test scans every `JsonSchemable` implementor and asserts each builds a
 * valid strict schema (no empty object, no Map), catching an under-annotated DTO before it
 * ships rather than crashing at agent setup in prod.
 *
 * Output schemas (tool params/returns) are deliberately NOT marked — they already have a
 * reflective handle through `@Instrument` (the InstrumentFactory enumerates them) and their
 * own smoke coverage, so a marker there would be redundant and unenforceable.
 */
interface JsonSchemable
