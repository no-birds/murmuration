package org.nobirds.murmuration.gemini

import com.google.genai.types.ClientOptions
import com.google.genai.types.GenerateContentConfig
import com.google.genai.types.HttpOptions
import okhttp3.Call
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import java.time.Duration
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * Gives the streaming path a handle on the OkHttp [Call] behind a `generateContentStream` request,
 * so a timed-out stream can really be torn down instead of living on as a zombie.
 *
 * The google-genai SDK never exposes the `Call` (and its future's `cancel` does not reach it), so the
 * handle is obtained from the transport side: [clientOptions] installs a custom `OkHttpClient` whose
 * application [Interceptor] reads a per-request ticket header (stamped by [StreamCall.applyTo]),
 * strips it before the request goes out, and registers `chain.call()` on the matching [StreamCall].
 *
 * One tracker per genai `Client`: build the client with `.clientOptions(tracker.clientOptions())` and
 * hand the SAME tracker to [StreamingResponseStrategy]. The strategy fails loudly on the first request
 * if the interceptor never fired (tracker not wired into the client).
 */
class StreamCallTracker {

    private val tickets = ConcurrentHashMap<String, StreamCall>()

    private val interceptor = Interceptor { chain ->
        val request = chain.request()
        val ticketId = request.header(TICKET_HEADER)
        val ticket = ticketId?.let(tickets::get)
        if (ticket == null) {
            chain.proceed(request)
        } else {
            ticket.register(chain.call())
            chain.proceed(request.newBuilder().removeHeader(TICKET_HEADER).build())
        }
    }

    /**
     * Client options carrying the tracking `OkHttpClient`. A custom client replaces the SDK's own
     * default one, so its defaults are reproduced here: the SDK builds its client with connect/read/
     * write timeouts of 0 (no socket-level limit; liveness is bounded by the executor's idle/overall
     * timeouts, never by OkHttp's 10s defaults which would kill a thinking-silent stream).
     */
    fun clientOptions(): ClientOptions = ClientOptions.builder()
        .customHttpClient(
            OkHttpClient.Builder()
                .connectTimeout(Duration.ZERO)
                .readTimeout(Duration.ZERO)
                .writeTimeout(Duration.ZERO)
                .addInterceptor(interceptor)
                .build(),
        )
        .build()

    /** Opens a ticket for ONE streaming request. Always pair with [StreamCall.abort] in a `finally`. */
    internal fun open(): StreamCall = StreamCall(UUID.randomUUID().toString()) { id -> tickets.remove(id) }
        .also { tickets[it.id] = it }


    /** Number of open, not yet released tickets (test/diagnostic visibility). */
    internal val openCount: Int get() = tickets.size

    internal companion object {
        const val TICKET_HEADER = "X-Murmuration-Stream-Ticket"
    }
}

/**
 * Handle on one streaming request's OkHttp [Call]. [abort] is idempotent and safe at any point:
 * before the interceptor registered the call (the call is cancelled the moment it registers), while
 * the call is hanging, or after it completed normally (cancelling a finished call is a no-op).
 */
internal class StreamCall(
    val id: String,
    private val release: (String) -> Unit,
) {
    private val lock = Any()
    private var call: Call? = null
    private var aborted = false

    /** True once the tracking interceptor has seen this request. */
    val isRegistered: Boolean get() = synchronized(lock) { call != null }

    /** The registered call, if any (test visibility). */
    val registeredCall: Call? get() = synchronized(lock) { call }

    /** Stamps the ticket header into [config]'s per-request [HttpOptions], preserving what is there. */
    fun applyTo(config: GenerateContentConfig): GenerateContentConfig {
        val existing = config.httpOptions().orElse(HttpOptions.builder().build())
        val headers = existing.headers().orElse(emptyMap()) + (StreamCallTracker.TICKET_HEADER to id)
        return config.toBuilder().httpOptions(existing.toBuilder().headers(headers).build()).build()
    }

    fun register(registered: Call) {
        val cancelNow = synchronized(lock) {
            call = registered
            aborted
        }
        if (cancelNow) {
            registered.cancel()
            release(id)
        }
    }

    /**
     * Cancels the call and releases the ticket. If the call has not registered yet (abort raced the
     * request's start), the ticket stays until [register] cancels the late call — releasing it now
     * would let that call live on as a zombie. The only residual is a request that never reaches the
     * network at all, which leaves one small ticket object behind.
     */
    fun abort() {
        val registered = synchronized(lock) {
            aborted = true
            call
        }
        if (registered != null) {
            registered.cancel()
            release(id)
        }
    }
}
