package org.nobirds.murmuration.agent

import java.util.UUID
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

/**
 * Publishes the ALREADY-EXISTING execution identity — which agent of which pipeline is
 * executing — into the coroutine context, so lower layers (prompt executors) can see who is
 * executing without a new parameter channel through the prompt/executor APIs. This is not a
 * new concept: [pipelineId] is [ExecutionContext.pipelineId] and [agentName] is the agent's
 * `displayName` ([ExecutionContext.getDisplayName]) — the same identity every
 * `reportAgentExecution` entry already carries; this element merely makes it visible DOWN the
 * call stack instead of only up the event stream.
 *
 * ## Propagation contract (cold flows)
 *
 * [org.nobirds.murmuration.prompt.PromptExecutor.execute] returns a cold
 * [kotlinx.coroutines.flow.Flow]: its suspend body runs in the COLLECTOR's coroutine context
 * at the point of `.collect`, not the context that was active when the prompt/flow was built
 * (standard cold-flow context preservation). So the agent must wrap the actual `.collect`
 * (directly or via an aggregator) in `withContext(AgentExecutionIdentity(...)) { ... }` —
 * wrapping prompt construction alone has no effect. [org.nobirds.murmuration.agent.TextPromptAgent]
 * and [org.nobirds.murmuration.tooling.ToolCallingAgent] do exactly that.
 *
 * A provider that supports identity-aware request handling reads
 * `currentCoroutineContext()[AgentExecutionIdentity]` from inside its own flow body and maps
 * it onto its own mechanism (e.g. the Azure OpenAI executor forwards [agentName] as
 * `prompt_cache_key`, keeping one agent invocation's requests on one warm cache replica);
 * providers without an equivalent concept simply never look it up.
 */
class AgentExecutionIdentity(
    val pipelineId: UUID,
    val agentName: String,
    /**
     * Composition path of the [step] frame this invocation runs in (see
     * [ExecutionContext.stepPath]), outermost first; empty at the pipeline root. Published for
     * the same reason as [agentName] — so a layer below the agent can tell *which* invocation of
     * a shared module it is serving, not just which agent.
     */
    val stepPath: List<String> = emptyList(),
) : AbstractCoroutineContextElement(AgentExecutionIdentity) {
    companion object Key : CoroutineContext.Key<AgentExecutionIdentity>
}
