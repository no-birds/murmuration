# Tools and schemas

## Declaring a tool

A tool is a plain Kotlin object. `InstrumentFactory().create(obj)` (done for you by `toolAgent`) reflects over it
and builds one declaration per tool function.

```kotlin
class WeatherTool {
    @Instrument("Get the current temperature for a city, in Celsius.")
    fun temperature(@InstrumentParam("City name") city: String): Int = lookup(city)
}
```

- `@Instrument(description, name)` on a function exposes it to the model; the name defaults to the function name.
  On a class it exposes every public function of the class (except `equals`, `hashCode`, `toString`).
- `@InstrumentParam(description)` describes a parameter; the description goes into the schema.
- `@NotAnInstrument` keeps a public function of an `@Instrument` class out of the tool set.
- Parameters and return values are ordinary Kotlin types. Parameter types must be serializable with
  kotlinx.serialization (`@Serializable` classes, enums, collections, primitives). Parameters with a Kotlin
  default are optional in the schema.
- A function may be `suspend`.
- Tools are listed in alphabetical order of their names, so the tool set is deterministic across runs.

## Agent-level annotations

| Annotation | Effect |
|---|---|
| `@Phases("A", "B")`, `@AllPhases` | Restrict the tool to state-machine phases, or keep it available in every phase. See [agents.md](agents.md#state-machines). |
| `@NeverSkip` | Opt out of the repeated-call de-duplication; use it for gates and positional mutators. |
| `@LeavesRecord(RecordReader.X)` | Declares who reads the effect of the call (world, player, calling model, bookkeeping, own phase). |
| `@CantCloseTask` | The tool does not get the `closesTaskId` parameter that `TaskClosing` owners add to every tool. |
| `@CallValidatedBy(MyValidator::class)` | Cross-parameter validation, see below. |

## Validation and transformation

Validators run before the tool body. A rejection becomes the tool result the model reads, so the model can fix the
call on the next round-trip. Each built-in validator also projects its constraint into the schema and into the
parameter description, so the model sees the rule up front.

Built-in annotations: `@NotBlank`, `@NotEmpty`, `@MinLength`, `@MaxLength`, `@Pattern`, `@NotContains`, `@MinSize`,
`@MaxSize`, `@Min`, `@Max`. Compose shortcuts with `@ValidationGroup`:

```kotlin
@ValidationGroup
@NotBlank
@MaxLength(50)
annotation class ShortText
```

On a constructor property of a data class use the explicit `@property:` use-site (`@property:NotBlank val id: String`).
Without it Kotlin attaches the annotation to the constructor parameter, the serial descriptor never sees it, and
the validator silently does nothing.

Transformers run before validators: `@Trim`, `@NormalizeNewlines`, groups through `@TransformationGroup`.

Extension points:

- `ParameterValidator` and `ValidatorRegistry` (annotation type to validator factory);
- `ParameterTransformer` and `TransformerRegistry`;
- `CallValidator` with `@CallValidatedBy` for checks across several parameters;
- `InstrumentDecorator` and `DecoratorRegistry` for arbitrary aspects; pass your own registry to
  `InstrumentFactory(decorators)`.

Argument decoding is strict. Values that do not fit the declared types (unknown enum value, wrong shape) are
refused with one message listing all problems (`ToolArgumentException`). Enum values are matched
case-insensitively, `@SerialName` and `@JsonNames` aliases are honoured, and an enum whose constants differ only
by case is refused when the tool is mounted (`AllowedValues`).

## JSON schemas

The core has its own provider-neutral schema model, `JsonSchema` (`ObjectSchema`, `ArraySchema`, `StringSchema`,
`IntegerSchema`, `NumberSchema`, `BooleanSchema`, `AnyOfSchema`).

- `KSerializer<T>.toJsonSchema()` walks a kotlinx descriptor and builds the schema.
- `@SchemaInfo(description, format, example, minLength, maxLength, minValue, maxValue, pattern)` adds metadata to a
  property or class.
- `@SchemaGeneratorMeta(generator = ...)` plugs in a custom `SchemaGenerator` (for example
  `PrimitiveUnionGenerator`); on a constructor property use `@property:SchemaGeneratorMeta`.
- `jsonSchema { ... }` is a small DSL for hand-written schemas.
- `InstrumentDeclaration.toFunctionSchema()` gives the `FunctionSchema` of a tool: name, description and the
  parameter object with validator constraints applied.

Providers translate the neutral model into their own format:
`JsonSchema.toGeminiSchema()` (`murmuration-gemini`), `JsonSchema.toOpenAiJson(strict)` and
`FunctionSchema.toOpenAiFunctionJson()` (`murmuration-openai`). The Anthropic module checks a schema against
the subset of JSON Schema that Claude accepts before the request leaves the process.

### Structured output

`textAgent<In, Out>(llm, executor) { ... }` generates the schema from a `@Serializable` `Out`, asks the provider
for constrained output and decodes the answer.

### Showing a schema to the model

Types that implement the marker interface `JsonSchemable` can be shown as schema documentation inside a prompt:
`inputSchema<T>()` adds the schema as text, `jsonWithSchema<T>(value)` adds a schema and a concrete value. The
schema part is cached for the pipeline by default and the value part is not (see [caching.md](caching.md)).

## Keeping Gemini function-calling schemas small

Gemini on Vertex AI enforces tool schemas with a finite-state transducer built per request from the whole tool
set. When it outgrows the service limit, the request fails with `Constraint is too tall: <N> vs max of ...`. The
limit has been observed at 5888 states; treat it as service behaviour that can change. What drives the size
(observed):

- long enum values (every value is a branch of a character trie), long property names;
- `maxLength`, `minLength`, `minimum`, `maximum` (a counter automaton per property);
- deep nesting, long arrays, big `anyOf` unions, `format`s such as `date-time`;
- many optional properties.

What helps: short `@SerialName` codes for enum constants (keep the readable Kotlin names, and persist the serial
name), dropping bounds that are recommendations rather than enforced rules (move them into the description text),
and hosting an enum-bearing field on a single dedicated tool instead of in a DTO reused by many tools. Making an
optional parameter `required + nullable` does not change the size.

`FstStateModel.states(declarations)` in `murmuration-gemini` computes the same state count locally, without calling
the service, for a list of Gemini `FunctionDeclaration`s. It is meant for tests that guard the size of a tool set.
Features it has no measured model for fail loudly. Keep such a test in `src/test`.

`GeminiWireKeyLayout` and `GeminiWireKeyTable` can additionally move chosen argument fields to the tail of the
schema under short wire names (reversed automatically when calls are decoded). `GeminiWireKeyLayout.NONE` leaves
every schema canonical; that is what you want unless you tune schema size.
