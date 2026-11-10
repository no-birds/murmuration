package org.nobirds.murmuration.openai

import com.openai.client.OpenAIClient
import com.openai.core.MultipartField
import com.openai.models.Reasoning
import com.openai.models.ReasoningEffort
import com.openai.models.responses.FunctionTool
import com.openai.models.responses.ResponseCreateParams
import com.openai.models.responses.ResponseFormatTextJsonSchemaConfig
import com.openai.models.responses.ResponseIncludable
import com.openai.models.responses.ResponseTextConfig
import com.openai.models.responses.Tool
import com.openai.models.embeddings.EmbeddingCreateParams
import com.openai.models.images.ImageEditParams
import com.openai.models.images.ImageGenerateParams
import com.openai.models.images.ImagesResponse
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.future.await
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.serializer
import org.nobirds.murmuration.openai.internal.logger
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.DataPromptPart
import org.nobirds.murmuration.prompt.EmbeddingPromptSettings
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.ImageAspectRatio
import org.nobirds.murmuration.prompt.ImagePromptSettings
import org.nobirds.murmuration.prompt.ImageResolution
import org.nobirds.murmuration.prompt.ImageSize
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.MusicPromptSettings
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.ParsePhase
import org.nobirds.murmuration.agent.AgentExecutionIdentity
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ThinkingLevel
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.TypedPromptSettings
import org.nobirds.murmuration.prompt.dump
import org.nobirds.murmuration.prompt.dumpSettings
import org.nobirds.murmuration.schema.toFunctionSchema
import org.nobirds.murmuration.schema.toJsonSchema
import java.util.Base64
import kotlin.jvm.optionals.getOrNull
import kotlin.math.ceil
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Builds the strict JSON-Schema `JsonObject` used for Azure OpenAI structured output from a
 * Kotlin type's kotlinx serializer — the OpenAI analogue of `GeminiJsonSchemaProvider`.
 */
@OptIn(ExperimentalSerializationApi::class)
class OpenAiJsonSchemaProvider {
    fun generateSchemaFromType(type: KClass<*>): JsonObject =
        serializer(type.java).toJsonSchema().toOpenAiJson(strict = true)
}

/**
 * Azure OpenAI [PromptExecutor] — the openai-family mirror of `GeminiClient`. Serves the
 * GPT-5.4 chat family (text + tool-calling + structured output), gpt-image-2 (Images API) and
 * text-embedding-3-large (Embeddings API).
 *
 * [deploymentResolver] maps a canonical model name ([Llm.model], e.g. `gpt-5.4`) to the Azure
 * **deployment name**, sent as the `model` field of the request body; the client talks to the
 * unified `/openai/v1` endpoint (`api-version=preview`) — the Responses API is not served on
 * the legacy `/openai/deployments/{deployment}` path.
 *
 * There is no context-cache path here: Azure exposes only implicit prompt caching, so this
 * executor never emits `CacheCreated` and carries no caching provider.
 */
class AzureOpenAiClient(
    private val client: OpenAIClient,
    private val responseStrategy: GptResponseStrategy,
    /** Maps canonical model name ([Llm.model]) to the Azure deployment name used in requests. */
    private val deploymentResolver: (String) -> String = { it },
    private val schemaProvider: OpenAiJsonSchemaProvider = OpenAiJsonSchemaProvider(),
) : PromptExecutor {

    private val logger = logger()

    override fun supports(llm: Llm): Boolean = llm.provider == "azure"

    override fun execute(prompt: Prompt): Flow<PromptExecutionEvent> =
        when (prompt.settings) {
            is EmbeddingPromptSettings -> executeEmbedding(prompt)
            is MusicPromptSettings -> executeUnsupported(prompt, "Music generation is served by the Vertex provider")
            is ImagePromptSettings -> executeImage(prompt)
            else -> executeText(prompt)
        }

    // Modalities Azure OpenAI doesn't serve in our stack. Emits a Failed + throws (rather than
    // erroring on the eager path) to keep the executor contract: every failure flows through
    // the event stream first.
    private fun executeUnsupported(prompt: Prompt, reason: String): Flow<PromptExecutionEvent> = flow {
        val requestMessages = prompt.content.toMessages()
        emit(PromptStarted(prompt.llm.model, prompt.llm.model, requestMessages))
        val error = InvalidRequestException(prompt, httpCode = null, message = reason, cause = null)
        emit(Failed(error))
        throw error
    }

    private fun executeText(prompt: Prompt): Flow<PromptExecutionEvent> = flow {
        val settings = prompt.settings
        val input = prompt.content.toResponseInput()
        val requestMessages = prompt.content.toMessages()
        val deployment = deploymentResolver(prompt.llm.model)

        val builder = ResponseCreateParams.builder()
            .model(deployment)
            .inputOfResponse(input)
            // Stateless reasoning transport: don't persist server-side, but ask the API to return
            // reasoning items with `encrypted_content` so we can feed them back on the next
            // round-trip (the model continues its thought instead of re-deriving it). This is the
            // whole point of the Responses migration — chat completions dropped reasoning between
            // tool-calling round-trips.
            .store(false)
            .addInclude(ResponseIncludable.REASONING_ENCRYPTED_CONTENT)

        // Output-token cap (content + reasoning combined) — INCOMPLETE/max_output_tokens surfaces
        // as MaxTokensException so a runaway loop fails fast. `null` = provider default.
        settings?.maxOutputTokens?.let { builder.maxOutputTokens(it.toLong()) }
        // Reasoning config: `summary=auto` is always requested so the model returns readable
        // reasoning summaries (our only window into GPT-5.x thoughts — the raw CoT never leaves
        // the server; "auto" because GPT-5.x rejects "concise"). The effort tier is added only
        // when the caller set one (reasoning models otherwise use their default); the SDK builder
        // treats both fields as independent, so summary-only Reasoning is valid.
        builder.reasoning(
            Reasoning.builder()
                .summary(Reasoning.Summary.AUTO)
                .apply { settings?.thinkingLevel?.let { effort(it.toReasoningEffort()) } }
                .build()
        )
        // Azure has no numeric thinking budget — only reasoning_effort tiers.
        settings?.thinkingBudget?.let {
            logger.debug("thinkingBudget ({}) is not supported by Azure OpenAI; use thinkingLevel — ignoring", it)
        }
        // temperature / top_p are NOT sent: reasoning models reject them. parallel_tool_calls is
        // pinned to the API default explicitly — batching several calls per round-trip is the
        // behaviour the tool-calling loop is built around.
        builder.parallelToolCalls(true)

        // Implicit-cache routing hint: the ambient [AgentExecutionIdentity] (see its KDoc), if the
        // executing agent published one into the coroutine context around this flow's collection,
        // lets Azure's router consistently send this agent invocation's repeated, growing-prefix
        // requests to the same warm replica instead of round-robining them across the deployment
        // (implicit prompt caching is per-replica, NOT sticky by itself). The identity's agentName
        // (the agent displayName) is stable across one invocation's round-trips and distinct
        // across agents/invocations — exactly the shard-key shape `prompt_cache_key` wants.
        // Absent when no identity is in context — the request is sent exactly as before (no field
        // on the wire), so this is purely additive.
        currentCoroutineContext()[AgentExecutionIdentity]?.agentName?.let { builder.promptCacheKey(it) }

        when (settings) {
            is ToolsPromptSettings -> {
                // toolChoice is left at the API default (auto when tools are present).
                val tools = settings.tools.flatMap { it.instruments }.map { Tool.ofFunction(it.toOpenAiTool()) }
                if (tools.isNotEmpty()) builder.tools(tools)
            }
            is TypedPromptSettings<*> -> builder.text(buildTextConfig(settings.type))
            else -> {}
        }

        // Full prompt dump — TRACE only: the content already lands in
        // `llm_statistics.raw_request_prompt`, no point duplicating it in normal logs.
        if (logger.isTraceEnabled) {
            logger.trace(
                "Sending Azure OpenAI request [deployment={}, settings={}]\n{}",
                deployment, settings.dumpSettings(), prompt.dump(),
            )
        }

        emitAll(responseStrategy.execute(prompt, builder.build(), requestMessages, prompt.llm.model))
    }

    // Images API (gpt-image-2). A separate API method, hence branched above `chat.completions`.
    // Emits the SAME events (PromptStarted / PartReceived / UsageDelta / Completed) so cost and
    // statistics stay uniform with text. A prompt WITHOUT inline input images goes to
    // `images.generate`; a prompt WITH them (reference portraits, style samples) goes to
    // `images.edit`, which is how gpt-image models take input images.
    private fun executeImage(prompt: Prompt): Flow<PromptExecutionEvent> = flow {
        val settings = prompt.settings as ImagePromptSettings
        val requestMessages = prompt.content.toMessages()
        val deployment = deploymentResolver(prompt.llm.model)
        // gpt-image-2 takes a single text prompt: SYSTEM text as a prefix, then USER text.
        val promptText = listOf(promptTextOf(prompt, PromptContentType.SYSTEM), promptTextOf(prompt, PromptContentType.USER))
            .filter { it.isNotBlank() }
            .joinToString("\n\n")

        // Input reference images: inline image parts from the request side of the prompt.
        // File-uri parts are NOT resolved here — materializing them into bytes is the calling
        // layer's concern (the application wraps this executor in a materializing decorator), so
        // whatever file-uri parts are left at this point are skipped.
        val requestParts = prompt.content
            .filter { it.type == PromptContentType.SYSTEM || it.type == PromptContentType.USER }
            .flatMap { it.parts }
        val inputImages = requestParts.filterIsInstance<DataPromptPart>()
            .filter { it.mimeType.startsWith("image/") }
        requestParts.filterIsInstance<FileDataPromptPart>().takeIf { it.isNotEmpty() }?.let { skipped ->
            logger.debug(
                "Skipping {} file-uri part(s) in the image prompt — the Images API takes inline bytes only; " +
                    "materialization is the caller's concern",
                skipped.size,
            )
        }

        // The images response may not carry a model name, so report the requested one.
        emit(PromptStarted(prompt.llm.model, prompt.llm.model, requestMessages))
        try {
            val response = if (inputImages.isEmpty()) {
                client.async().images().generate(buildGenerateParams(deployment, promptText, settings)).await()
            } else {
                client.async().images().edit(buildEditParams(deployment, promptText, settings, inputImages)).await()
            }
            emit(StreamResponded(servedModelName = prompt.llm.model))
            emitImageResult(response)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            val mapped = e.toPromptExecutionException(prompt)
            emit(Failed(mapped))
            throw mapped
        }
    }

    private fun buildGenerateParams(
        deployment: String,
        promptText: String,
        settings: ImagePromptSettings,
    ): ImageGenerateParams = ImageGenerateParams.builder()
        .model(deployment)
        .prompt(promptText)
        .size(settings.size.toImageSize())
        // Quality tier follows the requested resolution (see toGenerateQuality) — there is no
        // separate quality knob on ImagePromptSettings. No responseFormat: gpt-image always
        // returns b64_json.
        .quality(settings.size.resolution.toGenerateQuality())
        .n(1L)
        .build()

    // Images EDIT: the same text prompt plus input images. gpt-image models accept up to 16
    // input images, serialized as multipart `image[]` parts by the SDK, IN THE ORDER GIVEN here —
    // `images` arrives from `executeImage` already order-preserved from the prompt's part list
    // (filter/flatMap/filterIsInstance are all order-preserving, and the file-materializing
    // decorator upstream replaces FileDataPromptPart 1:1 in place), so whichever image comes
    // first in the prompt's DSL is the one the edit API treats as the primary/first reference.
    // That matters: the API grants the FIRST input image special weight (a mask, if any, only
    // ever applies to it) — callers that care which image is "the one being edited" (e.g. the
    // map skeleton) must put it first, not the style sample (lesson 2026-07-04, see
    // TurnImageGenerator's map prompt). The multipart field carries the first image's mime type
    // and a filename with a matching extension — like the SDK's `image(path)` overload does — so
    // the server can detect the file format.
    private fun buildEditParams(
        deployment: String,
        promptText: String,
        settings: ImagePromptSettings,
        images: List<DataPromptPart>,
    ): ImageEditParams {
        val imageField = MultipartField.builder<ImageEditParams.Image>()
            .value(ImageEditParams.Image.ofInputStreams(images.map { it.data.inputStream() }))
            .contentType(images.first().mimeType)
            .filename("image.${images.first().mimeType.substringAfter('/')}")
            .build()
        return ImageEditParams.builder()
            .image(imageField)
            .prompt(promptText)
            .model(deployment)
            .size(settings.size.toImageEditSize())
            .quality(settings.size.resolution.toEditQuality())
            // Unconditionally HIGH: every edit-path use case we have (map skeleton, portrait→
            // token, style-referenced generation) needs the input geometry/likeness preserved,
            // never redrawn loosely — there's no case in this codebase that wants the "low"
            // (cheaper, looser) default, so this isn't wired as a per-call config knob (that
            // would be a channel with exactly one value ever used). The SDK's doc comment on
            // `InputFidelity` names `gpt-image-1`/`gpt-image-1.5`+ as supported and
            // `gpt-image-1-mini` as not; our deployment is Azure's `gpt-image-2`, which isn't
            // named either way — left in on the assumption it's a newer/renamed sibling of the
            // supported family, to be confirmed by an approved live smoke test (the API should
            // simply reject the field if it's truly unsupported, not corrupt output).
            .inputFidelity(ImageEditParams.InputFidelity.HIGH)
            .n(1L)
            .build()
    }

    // Shared tail of both Images API calls (generate and edit): decode the b64 payload into an
    // InlineData part, report usage, complete.
    private suspend fun FlowCollector<PromptExecutionEvent>.emitImageResult(response: ImagesResponse) {
        val b64 = response.data().getOrNull()?.firstOrNull()?.b64Json()?.getOrNull()
            ?: throw MalformedResponseException(ParsePhase.IMAGE_BLOB_MISSING, "Image response had no b64_json payload", null)
        val bytes = Base64.getDecoder().decode(b64)
        emit(PartReceived(MessagePart.InlineData(mimeType = "image/png", sizeBytes = bytes.size.toLong(), data = bytes)))
        response.usage().getOrNull()?.let { usage ->
            emit(
                UsageDelta(
                    input = InputTokenStatistics(total = usage.inputTokens(), cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = usage.outputTokens()),
                )
            )
        }
        emit(Completed())
    }

    // Embeddings API (text-embedding-3-large). A separate API method (`embeddings.create`),
    // hence branched above `responses` — the openai-family mirror of GeminiClient's
    // `executeEmbedding`. Emits the SAME events (PromptStarted / UsageDelta / PartReceived /
    // Completed) so cost and statistics stay uniform with text/image. Embeddings bill input
    // tokens only, so no output-token usage is reported.
    private fun executeEmbedding(prompt: Prompt): Flow<PromptExecutionEvent> = flow {
        val settings = prompt.settings as EmbeddingPromptSettings
        val text = prompt.content
            .flatMap { it.parts }
            .filterIsInstance<TextPromptPart>()
            .joinToString("\n") { it.text }
        val requestMessages = prompt.content.toMessages()
        val deployment = deploymentResolver(prompt.llm.model)

        emit(PromptStarted(prompt.llm.model, prompt.llm.model, requestMessages))
        try {
            // `dimensions` shrinks text-embedding-3's native 3072-d vector to the requested size
            // (768 here) — the openai analogue of Gemini's `outputDimensionality`. taskType has no
            // OpenAI counterpart and is ignored. The `model` field carries the Azure deployment name.
            val params = EmbeddingCreateParams.builder()
                .model(deployment)
                .input(text)
                .dimensions(settings.outputDimensionality.toLong())
                .build()
            val response = client.async().embeddings().create(params).await()
            val values = response.data().firstOrNull()?.embedding()?.toFloatArray()
                ?: throw MalformedResponseException(
                    ParsePhase.EMBEDDING_MISSING,
                    "Embedding response had no vector for deployment '$deployment'",
                    null,
                )
            emit(StreamResponded(servedModelName = prompt.llm.model))
            // Embeddings bill prompt (input) tokens only; report as an input-usage signal for cost.
            emit(UsageDelta(input = InputTokenStatistics(total = response.usage().promptTokens(), cached = 0, cacheWriteShort = 0, cacheWriteLong = 0)))
            emit(PartReceived(MessagePart.Embedding(values.size, values)))
            emit(Completed())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            val mapped = e.toPromptExecutionException(prompt)
            emit(Failed(mapped))
            throw mapped
        }
    }

    // Structured output on the Responses API rides the `text.format` json_schema config (the
    // analogue of chat completions' `response_format`). Schema goes through the strict emitter.
    private fun buildTextConfig(type: KClass<*>): ResponseTextConfig {
        val schema = ResponseFormatTextJsonSchemaConfig.Schema.builder()
            .putAllAdditionalProperties(schemaProvider.generateSchemaFromType(type).toJsonValueMap())
            .build()
        val format = ResponseFormatTextJsonSchemaConfig.builder()
            .name(type.simpleName ?: "response")
            .schema(schema)
            .strict(true)
            .build()
        return ResponseTextConfig.builder().format(format).build()
    }
}

/** Concatenates the text of one role's parts into a single prompt string. */
private fun promptTextOf(prompt: Prompt, type: PromptContentType): String =
    prompt.content
        .filter { it.type == type }
        .flatMap { it.parts }
        .filterIsInstance<TextPromptPart>()
        .joinToString("\n") { it.text }

/** Maps the provider-neutral reasoning tier onto OpenAI's `reasoning_effort`. */
internal fun ThinkingLevel.toReasoningEffort(): ReasoningEffort = when (this) {
    ThinkingLevel.MINIMAL -> ReasoningEffort.MINIMAL
    ThinkingLevel.LOW -> ReasoningEffort.LOW
    ThinkingLevel.MEDIUM -> ReasoningEffort.MEDIUM
    ThinkingLevel.HIGH -> ReasoningEffort.HIGH
}

/**
 * Builds a Responses API [FunctionTool] from an instrument. The parameters are a raw JSON Schema
 * object (non-strict regime) fed into the SDK's untyped [FunctionTool.Parameters] bag.
 */
private fun InstrumentDeclaration<*>.toOpenAiTool(): FunctionTool {
    val schema = toFunctionSchema()
    val parameters = FunctionTool.Parameters.builder()
        .putAllAdditionalProperties(schema.parameters.toOpenAiJson(strict = false).toJsonValueMap())
        .build()
    return FunctionTool.builder()
        .name(schema.name)
        .description(schema.description)
        .parameters(parameters)
        .strict(false)
        .build()
}

// gpt-image-2's hard pixel-area bounds. Confirmed against the SDK's own doc comment on
// `ImageGenerateParams.Size` / `ImageEditParams.Size` (byte-for-byte identical on both — so the
// two endpoints share one sizing contract): arbitrary `WIDTHxHEIGHT` strings, both edges
// divisible by 16, aspect ratio between 1:3 and 3:1, maximum supported resolution 3840x2160
// (area 8,294,400 px; above 2560x1440 is "experimental" but documented as working). The lower
// bound (655,360 px) is the model's documented minimum requested area.
internal const val MIN_IMAGE_AREA_PX = 655_360L
internal const val MAX_IMAGE_AREA_PX = 8_294_400L

/** Both edges of a gpt-image-2 request must land on this grid. */
internal const val SIZE_GRID = 16L

/**
 * Maps a provider-neutral [ImageSize] (resolution tier + aspect ratio) onto the `WIDTHxHEIGHT`
 * pixel dimensions gpt-image-2 accepts, on BOTH the generate and edit endpoints (see
 * [MIN_IMAGE_AREA_PX] doc — the SDK's `Size` type on either params class is a bare string
 * wrapper with the identical documented contract, so one formula serves both call sites via
 * [toImageSize] / [toImageEditSize]).
 *
 * ## Formula
 * 1. **Target area** = [ImageResolution.pixels]² clamped into `[MIN_IMAGE_AREA_PX,
 *    MAX_IMAGE_AREA_PX]`. [ImageResolution] itself only promises `pixels` as the long-edge
 *    reference for a 1:1 image (see [ImageSize.pixelWidth]); squaring it to get a target area is
 *    the natural generalization to non-square aspects, and clamping guards the two tiers that
 *    fall outside the model's legal area range on their own (SD_512² undershoots the minimum;
 *    UHD_4096² overshoots the maximum).
 * 2. The aspect ratio is reduced to lowest terms, then clamped to gpt-image-2's supported
 *    `[1/3, 3]` range by snapping to the nearest boundary (3:1 or 1:3) — this is what keeps an
 *    extreme ratio like `PORTRAIT_8_1` legal.
 * 3. Width and height are both scaled from that SAME reduced `(widthRatio, heightRatio)` pair by
 *    one integer multiplier `k`, the smallest multiple of 16 for which
 *    `k² * widthRatio * heightRatio >= target area`. Scaling both edges off one integer pair —
 *    instead of rounding width and height to the nearest 16 independently — guarantees the
 *    OUTPUT ratio is bit-for-bit the requested one; independent rounding drifts the ratio at
 *    skewed aspects. `k` must be a multiple of 16 (not merely "big enough"): since the ratio is
 *    reduced to lowest terms, at least one of the two numbers is odd, and an odd factor only
 *    lands on the 16-grid when `k` itself is a multiple of 16.
 *
 * Worked examples (verified in [AzureOpenAiClientImageSizeTest]):
 * - `SD_512` + `SQUARE_1_1` → 816×816 (512² = 262,144 undershoots the minimum, so the target
 *   area clamps up to 655,360; `sqrt(655360)/16 = 50.6` rounds up to k=51 → 816).
 * - `HD_1024` + `PORTRAIT_9_16` → 864×1536 (1024² = 1,048,576 is in range, but the odd `9`
 *   forces k up to the next multiple of 16 — 96 instead of the raw ~85.3 — overshooting the
 *   naive 768×1366 by about 12%).
 * - `XHD_2048` + `SQUARE_1_1` → 2048×2048 (2048 is already a multiple of 16, no overshoot).
 * - `UHD_4096` + `LANDSCAPE_16_9` → 3840×2160 (4096² overshoots the maximum, so the target area
 *   clamps down to exactly 8,294,400, which happens to divide out evenly for this ratio).
 */
internal fun ImageSize.toPixelDimensions(): Pair<Int, Int> {
    val targetArea = resolution.pixels.toLong().let { it * it }.coerceIn(MIN_IMAGE_AREA_PX, MAX_IMAGE_AREA_PX)
    val (wRatio, hRatio) = aspectRatio.reducedClampedRatio()
    val rawK = sqrt(targetArea.toDouble() / (wRatio * hRatio)) / SIZE_GRID
    var k = SIZE_GRID * ceil(rawK).toLong().coerceAtLeast(1L)
    // Rounding UP to the next 16-multiple guarantees area >= target, which is what we want for
    // the lower tiers (never render smaller than asked). But for UHD_4096 — where the target
    // area is already pinned to MAX_IMAGE_AREA_PX because 4096² alone overshoots it — that same
    // ceiling can push the rounded area just past the model's hard cap, especially for skewed
    // aspect ratios where one step of `k` moves the area by a wide margin. Legality of the
    // request wins over hugging the exact target: step back down to the largest 16-multiple that
    // still fits under the cap.
    while (k > SIZE_GRID && k * k * wRatio * hRatio > MAX_IMAGE_AREA_PX) {
        k -= SIZE_GRID
    }
    return (k * wRatio).toInt() to (k * hRatio).toInt()
}

/**
 * [ImageAspectRatio.widthRatio]/[ImageAspectRatio.heightRatio] reduced to lowest terms (some
 * enum entries, e.g. `PORTRAIT_21_9(21, 9)`, aren't already reduced) and clamped into
 * gpt-image-2's supported `[1/3, 3]` range by snapping to the nearest boundary.
 */
internal fun ImageAspectRatio.reducedClampedRatio(): Pair<Long, Long> {
    val g = gcd(widthRatio.toLong(), heightRatio.toLong())
    val w = widthRatio / g
    val h = heightRatio / g
    return when {
        w > 3 * h -> 3L to 1L
        h > 3 * w -> 1L to 3L
        else -> w to h
    }
}

internal tailrec fun gcd(a: Long, b: Long): Long = if (b == 0L) a else gcd(b, a % b)

/** Resolution → gpt-image-2 quality tier. There is no separate quality knob on
 * [ImagePromptSettings]; resolution alone drives it. */
internal fun ImageResolution.toGenerateQuality(): ImageGenerateParams.Quality = when (this) {
    ImageResolution.SD_512 -> ImageGenerateParams.Quality.LOW
    ImageResolution.HD_1024 -> ImageGenerateParams.Quality.MEDIUM
    ImageResolution.XHD_2048, ImageResolution.UHD_4096 -> ImageGenerateParams.Quality.HIGH
}

/** Same tiering as [toGenerateQuality], onto the edit endpoint's own [ImageEditParams.Quality]. */
internal fun ImageResolution.toEditQuality(): ImageEditParams.Quality = when (this) {
    ImageResolution.SD_512 -> ImageEditParams.Quality.LOW
    ImageResolution.HD_1024 -> ImageEditParams.Quality.MEDIUM
    ImageResolution.XHD_2048, ImageResolution.UHD_4096 -> ImageEditParams.Quality.HIGH
}

/** Maps [ImageSize] onto the generate endpoint's `WIDTHxHEIGHT` size string. See [toPixelDimensions]. */
internal fun ImageSize.toImageSize(): ImageGenerateParams.Size {
    val (w, h) = toPixelDimensions()
    return ImageGenerateParams.Size.of("${w}x${h}")
}

/** Maps [ImageSize] onto the edit endpoint's `WIDTHxHEIGHT` size string. See [toPixelDimensions]. */
internal fun ImageSize.toImageEditSize(): ImageEditParams.Size {
    val (w, h) = toPixelDimensions()
    return ImageEditParams.Size.of("${w}x${h}")
}
