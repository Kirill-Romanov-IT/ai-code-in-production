# AI-Generated Code vs Human Code: Why the Difference Matters and When It Becomes Critical

> An analytical article based on real code and current research, 2024–2026

---

## Table of Contents

- [Introduction: The Question Nobody Wants to Ask](#introduction)
- [Part I. What the Data Says](#part-i-what-the-data-says)
- [Part II. Real Code Under the Microscope](#part-ii-real-code-under-the-microscope)
- [Part III. Mechanics and Mathematics](#part-iii-mechanics-and-mathematics)
- [Part IV. Locally — Fine, Production — Risky](#part-iv-locally--fine-production--risky)
- [Part V. Task Taxonomy](#part-v-task-taxonomy)
- [Part VI. How to Work with AI-Generated Code](#part-vi-how-to-work-with-ai-generated-code)
- [Part VII. Where the Industry Is Heading](#part-vii-where-the-industry-is-heading)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

According to the Stack Overflow Developer Survey 2025, **84% of developers** regularly use AI tools for writing code. GitHub reports that in some repositories more than **30% of code** is written with AI assistance. Impressive numbers. But behind them hides a question the industry prefers not to ask out loud:

> **What exactly happens to code quality when AI writes it? And does the answer change depending on what that code is intended for?**

This article is an honest attempt to answer. Not from a position of "AI is bad" or "AI is good", but from an engineering perspective: what AI does well, what it does poorly, why, and when the difference stops being academic and becomes a production problem.

The material is built on three layers of evidence:
- Real code from a real task (a CLI multi-agent application)
- Analysis against SOLID and Clean Code principles
- Data from 30+ studies from 2024–2026 — CodeRabbit, Veracode, METR, GitClear, Google DORA, arxiv

The target audience is developers of any level and technical leaders who make decisions about how and where to use AI in their teams.

---

## Part I. What the Data Says

### 1.1. Key Numbers

| Research / Source | Key Metric | What Was Measured |
|---|---|---|
| CodeRabbit, Dec 2025 | **1.7× more defects** | 470 real open-source PRs |
| Veracode, 2025 | **45% of code has vulnerabilities** | 100+ LLMs on 80 security tasks |
| METR, July 2025 | **−19% speed for experienced devs** | RCT, 16 senior devs, 246 tasks |
| GitClear, 2025 | **4–8× growth in code duplication** | 211M lines, 2020–2024 |
| Anthropic, Jan 2026 | **−17% depth of understanding** | 52 junior devs, RCT |
| Google DORA, 2025 | **+30% change failure rate** | ~5,000 respondents |
| Cortex, 2026 | **+23.5% incidents per PR** | Enterprise telemetry |
| arxiv 2508.21634, 2025 | **Java: +22K defective samples** | 500K+ code samples |

### 1.2. Critical Caveat: No Direct Comparison with Mid-Level Developers Exists

This is important to establish honestly: **none of the studies found conduct a controlled experiment** of the format "give the same task to AI and a group of mid-level developers, compare the result". All data compares AI code with "human code" without breakdown by seniority level.

Indirect data allows us to build an approximate picture: AI code without human revision corresponds roughly to the **junior+ / low-mid level** — above a beginner developer on routine tasks, but below a confident mid-level on tasks requiring context, defensive programming, and architectural decisions.

> **Key finding:** AI does not write bad code. AI writes code for a vacuum — without accounting for the fact that the system will run under load, in a team, with an unreliable network and real users.

### 1.3. What AI Does Well — and Why

For the analysis to be honest, let us start with where AI is genuinely strong. These are tasks with **high pattern repeatability**: boilerplate, CRUD, regular expressions, SQL queries, unit tests, documentation. AI has seen millions of examples of these patterns in training data and reproduces them reliably.

- CodeRabbit: AI code has **1.32× fewer testability issues** than human code
- Spelling errors in human code occur **1.76× more often**
- GitHub: **53.2% higher probability** of passing all unit tests with Copilot

The mathematical reason is simple: for these tasks, the solution space is small and well-covered by training data. AI finds the pattern almost deterministically.

---

## Part II. Real Code Under the Microscope

### 2.1. The Task

For the analysis we used a real CLI application written by AI based on the following specification:

```
AI Team Chat — a CLI application simulating a team room.

Participants:
  - Director (human, enters the idea once at the start)
  - Junior (human, main user, writes messages)
  - Product Manager (AI agent — decomposes tasks)
  - Senior Developer (AI agent — designs architecture)

Stack: Python, requests, python-dotenv, OpenRouter API

Requirements:
  - Shared history for all agents
  - Sequential responses (Product first, then Senior)
  - In-memory storage per session
```

This is not a textbook exercise. It is a typical prototype that teams build for internal tools. Size: ~150 lines, 10 classes. Small enough to fully dissect, large enough for problems to become visible.

### 2.2. Full Source Code (Written by AI)

```python
import sys          # <- imported, never used
import os
import requests
import json
from dotenv import load_dotenv
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from abc import ABC, abstractmethod

load_dotenv()

class SenderType(Enum):
    DIRECTOR = "Director"
    JUNIOR   = "Junior"
    PRODUCT  = "Product"
    SENIOR   = "Senior"

@dataclass
class Message:
    sender: SenderType
    text: str
    timestamp: datetime = field(default_factory=datetime.now)  # stored, never used

class HistoryManager:
    def __init__(self):
        self.history: List[Message] = []  # grows unbounded — no TTL

    def add(self, message: Message) -> None:
        self.history.append(message)

    def get_all(self) -> List[Message]:
        return self.history

    def get_formatted(self) -> str:
        lines = []
        for message in self.history:
            lines.append(f"{message.sender.value}: {message.text}")
        return "\n".join(lines)

class AnthropicClient:               # <- name lies: this hits OpenRouter
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "qwen/qwen3-coder"

    def send(self, system: str, history: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": history}
                ]
            }
        )
        # no try/except, no status_code check, no timeout
        return response.json()["choices"][0]["message"]["content"]

class BaseAgent(ABC):
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.api_client = AnthropicClient()  # <- DIP violated: hard dependency

    @abstractmethod
    def respond(self, system: str, history: str) -> Message:
        ...

class ProductAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Product Manager",
            system_prompt="You are an experienced product manager. You decompose tasks."
        )

    def respond(self, system: str, history: str) -> Message:
        text = self.api_client.send(system, history)
        return Message(sender=SenderType.PRODUCT, text=text)

class SeniorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Senior Developer",
            system_prompt="You are an experienced Senior developer. You design architecture."
        )

    def respond(self, system: str, history: str) -> Message:
        text = self.api_client.send(system, history)
        return Message(sender=SenderType.SENIOR, text=text)

class ContextBuilder:
    def __init__(self, director_idea: str):
        self.idea = director_idea

    def build(self, agent: BaseAgent) -> str:   # accepts whole agent, needs only .system_prompt
        return f"{agent.system_prompt}\n\nDirector's idea: {self.idea}"

class AgentOrchestrator:
    def __init__(self, agents: List[BaseAgent], context_builder: ContextBuilder):
        self.agents = agents
        self.context_builder = context_builder

    def trigger(self, history: HistoryManager) -> List[Message]:  # name is unclear
        results = []
        for agent in self.agents:
            system = self.context_builder.build(agent)
            message = agent.respond(system, history.get_formatted())
            results.append(message)
        return results

class InputHandler:
    def __init__(self, history_manager: HistoryManager, orchestrator: AgentOrchestrator):
        self.history_manager = history_manager
        self.orchestrator = orchestrator

    def handle(self, text: str, sender: SenderType) -> None:
        message = Message(sender=sender, text=text)
        self.history_manager.add(message)
        messages = self.orchestrator.trigger(self.history_manager)
        for msg in messages:
            self.history_manager.add(msg)

class CLI:                           # <- God Object: init + I/O + display logic
    def __init__(self):
        director_idea = input("Enter the director's idea: ")
        context_builder = ContextBuilder(director_idea)
        history_manager = HistoryManager()
        orchestrator = AgentOrchestrator(
            agents=[ProductAgent(), SeniorAgent()],
            context_builder=context_builder
        )
        self.input_handler = InputHandler(
            history_manager=history_manager,
            orchestrator=orchestrator
        )
        self.history_manager = history_manager

    def run(self) -> None:
        print("\nChat started! You play the Junior. Type 'exit' to quit.\n")
        while True:
            text = input("Junior: ")
            if text == "exit":
                break
            self.input_handler.handle(text=text, sender=SenderType.JUNIOR)
            messages = self.history_manager.get_all()
            last_two = messages[-2:]   # <- magic number: assumes exactly 2 agents
            for msg in last_two:
                if msg.sender in [SenderType.PRODUCT, SenderType.SENIOR]:
                    print(f"\n{msg.sender.value}: {msg.text}\n")

CLI().run()
```

### 2.3. Analysis Against SOLID Principles

#### S — Single Responsibility Principle ✅ (mostly)

The principle is met for most classes. `HistoryManager` is responsible only for history. `AnthropicClient` — only for HTTP. `ContextBuilder` — only for prompt assembly.

**Violation:** `CLI` combines three responsibilities — initialising the entire system, managing user input, and filtering/displaying messages. This is a God Object in miniature.

#### O — Open/Closed Principle ⚠️ (partial)

`BaseAgent` is abstract — a new agent can be added without modifying existing code. ✅

**Violations:**
- `AnthropicClient` is named after one provider but works with another. Adding a second provider = modifying the class. ❌
- Agent order (Product→Senior) is determined by the order agents are passed in — fragile. ❌

#### L — Liskov Substitution Principle ✅

Fully met. `ProductAgent` and `SeniorAgent` are completely interchangeable through `BaseAgent`. The signature `respond(system, history) -> Message` is maintained in both subclasses.

#### I — Interface Segregation Principle ⚠️ (partial)

Interfaces are small — `BaseAgent` has one method. ✅

`HistoryManager` is passed around in its entirety everywhere, even though `InputHandler` only uses `add()` and `get_all()`. In Python without strict interfaces this does not create a syntax problem, but creates an architectural coupling.

#### D — Dependency Inversion Principle ❌ (critical violation)

```python
# How AI wrote it — DIP violation:
class BaseAgent(ABC):
    def __init__(self, name, system_prompt):
        self.api_client = AnthropicClient()  # creates its own dependency

# How it should be — with dependency injection:
class BaseAgent(ABC):
    def __init__(self, name: str, system_prompt: str, api_client: LLMClient):
        self.api_client = api_client  # dependency passed from outside

# Now in tests:
mock_client = MockLLMClient(response="test response")
agent = ProductAgent(api_client=mock_client)
# No real HTTP request needed
```

**Practical consequence:** writing a unit test for any agent without a real API call is impossible. Testing requires spinning up the entire system — unacceptable in production.

### 2.4. Clean Code Issues

#### Names That Lie

| Name in Code | What It Actually Does | What It Should Be Called |
|---|---|---|
| `AnthropicClient` | Sends requests to OpenRouter | `OpenRouterClient` or `LLMClient` |
| `trigger()` | Collects agent responses | `collect_responses()` |
| `build(agent)` | Uses only `agent.system_prompt` | `build(system_prompt: str)` |

#### Magic Numbers

```python
# AI code:
last_two = messages[-2:]
for msg in last_two:
    if msg.sender in [SenderType.PRODUCT, SenderType.SENIOR]:
        ...

# Problem: '-2' is implicit knowledge about the number of agents.
# Add a third agent — output silently breaks.

# Correct approach:
agent_senders = {SenderType.PRODUCT, SenderType.SENIOR}
new_messages = self.orchestrator.collect_responses(history)
for msg in new_messages:
    if msg.sender in agent_senders:
        print(f"{msg.sender.value}: {msg.text}")
```

#### No Error Handling

```python
# AI code — one line, no protection:
return response.json()["choices"][0]["message"]["content"]

# What can go wrong:
# 1. requests.post()          → ConnectionError, Timeout
# 2. response.json()          → JSONDecodeError (if 502 returns HTML)
# 3. ["choices"][0]           → KeyError (if API returns {"error": "rate limit"})
# 4. ["message"]["content"]   → KeyError for streaming responses

# Minimal correct version:
try:
    response = requests.post(..., timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
except requests.Timeout:
    raise LLMTimeoutError("API did not respond within 30 seconds")
except requests.HTTPError as e:
    raise LLMAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
except (KeyError, IndexError) as e:
    raise LLMResponseError(f"Unexpected API response format: {e}")
```

### 2.5. AI Code vs Mid+ Code: Side-by-Side Comparison

To illustrate the difference clearly, here is a payment processing function — a task in the "partial" category: AI knows the pattern but not the context.

#### AI code

```javascript
async function processPayment(userId, amount, cardNumber) {
  if (!userId || !amount || !cardNumber) {
    throw new Error("Missing fields");
  }

  const user = await db.users.findOne({ id: userId });

  const result = await stripe.charge({
    amount: amount,          // ❌ amount can be a string, negative, or Infinity
    currency: "usd",
    source: cardNumber,      // ❌ raw PAN in memory and logs — PCI DSS violation
  });

  await db.payments.create({
    userId: userId,
    amount: amount,
    stripeId: result.id,
    status: "success",       // ❌ always "success" — even if stripe returned "pending"
  });
  // ❌ no idempotency: double click = double charge
  // ❌ no transaction: stripe passed, DB failed = split state

  return { success: true, result };
}
```

#### Mid+ code

```javascript
async function processPayment({
  userId,
  amountCents,        // ✅ integers — no float issues, no negative values
  paymentMethodId,    // ✅ token, not PAN — PCI compliant
  idempotencyKey,     // ✅ repeated calls return same result
}) {
  validatePaymentInput({ userId, amountCents, paymentMethodId, idempotencyKey });

  // ✅ idempotency check
  const existing = await payments.findByIdempotencyKey(idempotencyKey);
  if (existing) return existing;

  const user = await users.findByIdOrThrow(userId);
  await checkPaymentLimits(user, amountCents);

  let stripeIntent;
  try {
    stripeIntent = await stripe.paymentIntents.create({
      amount: amountCents,
      currency: "usd",
      payment_method: paymentMethodId,
      confirm: true,
      idempotency_key: idempotencyKey,
    });
  } catch (err) {
    handleStripeError(err); // ✅ separates card_declined, network_error, insufficient_funds
  }

  // ✅ atomic: if DB fails after stripe succeeds — transaction rolls back
  const payment = await db.transaction(async (trx) => {
    return payments.create({
      userId,
      amountCents,
      stripeIntentId: stripeIntent.id,
      status: mapStripeStatus(stripeIntent.status), // ✅ honest status: pending, requires_action
      idempotencyKey,
    }, { trx });
  });

  await auditLog.record({ event: "payment.processed", userId, paymentId: payment.id });
  return payment;
}
```

**Key observation:** AI *knows* about idempotency, PCI DSS, and transactions. If explicitly asked — it will write all of that. But a mid+ developer applies these patterns automatically, without a reminder. The difference is not in knowledge — it is in **intuition formed through experiencing consequences**.

---

## Part III. Mechanics and Mathematics

### 3.1. The Fundamental Idea: Local Optimum ≠ Global Optimum

AI solves problems **at a point**. It sees a fragment of context, one moment in time, one user, one session. The solution at this point may be perfect. But production is a multi-dimensional space with time, concurrency, users, and dependencies.

**Mathematically:** AI always finds a local minimum in the neighbourhood of a given context. The global minimum for a system with N components, M users, and K external dependencies is a problem of a completely different dimensionality.

> **Physics analogy:** gradient descent converges to a local minimum. AI code is gradient descent in solution space with a single starting point. Production requires the global minimum in a multi-dimensional error landscape.

### 3.2. Twenty-Five Real Reasons

#### Category I. Mathematics of Complexity

**1. State space dimensionality grows exponentially.**
Locally: 1 user × 1 session × 1 state = trivial. Production: N users × M sessions × K agent states. AI optimises for N=1. At N=10⁶, intersections of states AI did not anticipate are combinatorially guaranteed.

**2. Dependency graph — acyclic locally, cyclic in the system.**
A single-user script has a linear graph (DAG). In production, components form a graph with cycles through queues, caches, events. AI builds a DAG without seeing that other system components will close the cycle. Hence deadlocks in production that do not exist locally.

**3. Number of unknowns exceeds number of equations.**
Production unknowns: latency, throughput, SLA, compliance, team conventions, data contracts, upstream API changes. This is an underdetermined system — infinitely many formally "correct" solutions, none optimal for the specific context.

**4. No idempotency — O(N) error instead of O(1).**
In production, retry logic, message queues, and network partitions guarantee repeated calls. Code without idempotency produces errors not once, but proportionally to users × retries.

**5. Tail statistics: p99 latency is invisible at N=1.**
With one user you always observe the median. With a million — 10,000 people always land in p99. AI optimises for the average case. Tail latencies (slow DNS, GC pause, lock contention) only manifest at scale.

#### Category II. System Mechanics

**6. Script as a system component changes data contracts.**
AI writes code with implicit data contracts. When this script becomes part of a pipeline, its output becomes another component's input. An upstream change breaks all downstream without an explicit contract.

**7. No backpressure — producer faster than consumer.**
AI scripts write to queues or DBs without checking that the consumer keeps up. If producer rate > consumer rate, the buffer overflows. Basic mechanics, not implemented by default.

**8. No circuit breaker — cascading failure of the entire system.**
If a dependent service degrades, AI code will wait for timeout, blocking threads. One slow node → thread pool exhaustion → the whole system hangs. Circuit breaker breaks this chain.

**9. Memory leak with in-memory storage and no TTL.**
`HistoryManager` grows indefinitely. Locally: 10 KB per session. In production, a session may last hours. At N=10⁶ concurrent sessions × unbounded history = OOM.

**10. Synchronous I/O blocks the event loop under concurrency.**
One HTTP call to an LLM = blocked thread for 500ms–30s. In production: 1,000 concurrent requests × 5 sec = 5,000 seconds of blocked CPU. Async/await or thread pool required.

**11. No observability — the system becomes a black box.**
AI does not add metrics, traces, or structured logging. During a production incident there is no data: when did it start, which agent failed, how long did each step take. MTTR grows proportionally.

**12. No graceful shutdown — data is lost during deployment.**
AI scripts do not handle `SIGTERM`. In-flight requests are aborted, uncommitted data is lost. OS sends SIGTERM → 30 sec → SIGKILL. Without a handler = total loss.

#### Category III. AI's Own Limitations

**13. Context window — AI does not see the entire system.**
Even a large model's context window is 128K–200K tokens. A real production codebase runs into millions of lines. AI writes code seeing only a fragment. Local optimum ≠ global optimum.

**14. Data freshness: the model does not know APIs changed over the past year.**
The model is trained on data up to its cutoff. Provider APIs change, endpoints get deprecated, rate limits change. AI confidently writes code based on outdated documentation.

**15. Non-determinism: one prompt produces different code each run.**
Temperature > 0 means AI generates different solutions for identical tasks. In a team of five, each person gets slightly different code for the same problem. Architectural decisions become inconsistent in random places.

**16. Confidence without verification — hallucination in critical paths.**
AI generates syntactically correct code with non-existent methods, wrong signatures. In production, if a code path is rarely executed (error handling, edge cases) — the bug lives for months.

**17. No domain knowledge — AI does not know business invariants.**
Business rules are not inferable from code. AI does not know them unless they are in the prompt. Violation of an invariant by technically correct code is the most expensive class of production bugs.

#### Category IV. Security

**18. Secrets in code and logs — attack surface grows with scale.**
An API key from `.env` locally risks only your machine. In production: the key lives in environment variables on multiple servers, CI/CD systems, potentially in logs.

**19. Prompt injection through user input.**
In the example code: user text goes directly into history → into the agent prompt without sanitisation. A user can write an instruction overriding agent behaviour. In production — a vector to attack other users' data.

**20. No rate limiting — DoS by a single user.**
One user can send thousands of messages. Each message = HTTP request to LLM = money + latency for everyone. Without rate limiting, one abuser exhausts the API quota for the entire production system. Attacker cost: zero.

**21. All users' history in one context — data leak.**
`HistoryManager` is not isolated per user. Moving to a multi-user environment — the agent sees all users' history. Classic mistake when porting a script to a web environment.

**22. Dependency on a third party with no fallback — single point of failure.**
100% of logic depends on one external provider. If the provider changes terms or goes down — the system is completely dead. AI does not design fallbacks because the API works at the time of writing.

#### Category V. Organisational Risks

**23. Untestability — regressions cannot be caught automatically.**
Hard dependencies (DIP violated) make unit tests impossible without a real API. Every change requires manual verification. The number of missed regressions grows linearly with development velocity.

**24. Technical debt compounds — refactoring cost grows exponentially.**
Every new component built on a bad foundation inherits its problems and adds its own. The cost of refactoring grows as O(N²) with the number of components. AI writes "works now" code — each such component increases future change costs non-linearly.

**25. System knowledge is not transferred — bus factor = 0.**
If the system was written by AI, nobody on the team understands why a given decision was made. Architecture Decision Records (ADRs) are absent. During a 3am incident — the knowledge lives only in a prompt history that no longer exists.

---

## Part IV. Locally — Fine, Production — Risky

### 4.1. Why the Same Code Has Different Risk Levels

| Problem in Code | Locally (1 user) | Production (10K+ users) |
|---|---|---|
| `HistoryManager` without TTL | 10 KB per session, unnoticeable | N sessions × unbounded = OOM |
| No HTTP retry | Crashed — restart | 10K requests/sec × failure rate |
| `messages[-2:]` | Works, always 2 agents | Added agent — output silently broke |
| No rate limiting | You don't attack yourself | One abuser = exhausted API quota |
| DIP violated, no DI | No tests needed, run manually | Regressions not caught automatically |
| Prompt injection | You attack yourself | Attack vector on other users' data |
| No graceful shutdown | Ctrl+C, restart | In-flight request loss during deployment |

### 4.2. What Actually Hurts Even Locally

One problem exists regardless of scale and requires fixing even for a personal tool — **absent HTTP error handling**. If OpenRouter returns 429 or 500, or the network drops — the application crashes with an ugly Python traceback.

```python
def send(self, system: str, history: str) -> str:
    try:
        response = requests.post(
            self.url,
            headers=self.headers,
            json=self.payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.Timeout:
        raise RuntimeError("LLM API did not respond within 30 seconds")
    except requests.HTTPError as e:
        raise RuntimeError(f"API returned error: {e.response.status_code}")
    except (KeyError, IndexError):
        raise RuntimeError("Unexpected response format from API")
```

### 4.3. Scale Thresholds: When Problems Become Critical

| Problem | Threshold of Impact | Nature of Risk |
|---|---|---|
| No HTTP error handling | Immediately (1 user) | UX: ugly crash |
| HistoryManager without TTL | ~100 long sessions | OOM, service restart |
| No rate limiting | ~1,000 users/day | API quota exhaustion |
| No idempotency | When retry logic is added | Data duplication |
| No circuit breaker | ~10K requests/day | Cascading system failure |
| Prompt injection | When multi-tenant | Data leak across users |
| DIP violated | At first refactoring | Inability to write tests |
| No observability | At first production incident | MTTR hours instead of minutes |

---

## Part V. Task Taxonomy

### 5.1. Engineering Classification by Task Type

| Task Category | AI Performance | Why | Examples |
|---|---|---|---|
| Boilerplate and CRUD | ✅ Excellent | High pattern repeatability, no context | Getters, REST endpoints, DTOs |
| Regex and parsers | ✅ Excellent | Strict formal grammar | Email validation, CSV parser |
| SQL queries | ✅ Good | Declarative, no system context | Aggregations, JOINs |
| Unit tests | ✅ Good | Typical assert patterns | Happy path tests |
| Documentation | ✅ Good | Language task, no context | JSDoc, README, comments |
| Error handling | ⚠️ Partial | Knows patterns, not the domain | try/catch yes, custom errors no |
| Integration code | ⚠️ Partial | Does not know real API and config | Stripe, AWS SDK |
| Concurrency | ⚠️ Partial | Knows mutexes, not execution graph | Race conditions, deadlocks |
| Auth and authorisation | ⚠️ Partial | Typical patterns yes, custom rules no | JWT yes, custom RBAC no |
| System architecture | ❌ Poor | Does not understand trade-offs | Microservices vs monolith |
| Production debugging | ❌ Poor | No access to real runtime state | Race condition under load |
| Security design | ❌ Poor | 45% of code has vulnerabilities (Veracode) | Threat modelling, cryptography |
| Business context | ❌ Poor | Does not know domain and business rules | Financial invariants, compliance |

### 5.2. Why the Boundary Is Here

The boundary between "good" and "poor" passes not along syntactic complexity, but along the **type of knowledge required**:

**Pattern knowledge** — "how to write a FOR EACH loop correctly", "what a typical JWT authorisation looks like". This is what AI extracts from training data. AI is strong here.

**Contextual knowledge** — "why in our system authorisation works exactly this way", "which invariant will break if this flag is changed", "why Redis was chosen over Memcached three years ago". This knowledge does not exist on the internet — it lives in the minds of the people who built the system. AI has no access to it.

This is why AI writes good code for isolated tasks and poor code for tasks requiring understanding of the system as a whole. This is **not a temporary limitation** that will disappear with more powerful models — it is a structural limitation: information about a specific system simply does not exist in training data.

---

## Part VI. How to Work with AI-Generated Code

### 6.1. Interaction Model: AI as Intern, Not as Senior

The most accurate metaphor for AI code is **a short-term contractor unfamiliar with the project**. They write quickly, know all the textbook patterns, but do not know why your project is structured the way it is. GitClear uses exactly this formulation to describe the character of AI code in real projects.

This implies a concrete interaction model: **AI generates a draft, a human reviews it with an understanding of system context**. Not the other way around.

### 6.2. AI Code Review Checklist

- [ ] Are all failure paths handled (network, timeout, unexpected response format)?
- [ ] Are there magic numbers that encode implicit assumptions about the system?
- [ ] Do class and method names match what they actually do?
- [ ] Can a unit test be written without spinning up real dependencies?
- [ ] Is there user input that goes directly into a system prompt or SQL query?
- [ ] Are there limits on data sizes (lists, buffers, history)?
- [ ] Does the code comply with relevant regulations (PCI, GDPR, SOC2) if applicable?
- [ ] Does at least one person on the team understand why the code is written the way it is?

### 6.3. Where AI Adds Maximum Value

| Use AI for | Retain human control for |
|---|---|
| Generating boilerplate and CRUD | Architectural decisions |
| Writing documentation and comments | Security-critical code |
| Generating typical tests | Business logic with invariants |
| SQL queries and regular expressions | Handling sensitive data |
| First draft of a new component | Code review with system context |
| Refactoring formatting and naming | Decisions about scaling |

---

## Part VII. Where the Industry Is Heading

### 7.1. The Speed Paradox

Data from 2025–2026 records an unexpected pattern — the **speed paradox**: individual productivity grows, organisational quality metrics do not.

Faros AI (telemetry from 10,000+ developers):

| Metric | Change |
|---|---|
| Merged PRs | **+98%** |
| Completed tasks | **+21%** |
| Time spent on review | **+91%** |
| PR size | **+154%** |
| Bugs per developer | **+9%** |

Cortex Engineering Benchmark 2026: with a 20% growth in PRs per author — incidents per PR grew by **23.5%**, change failure rate by **30%**.

> **Key insight for 2025–2026:** the industry is shifting from the narrative "AI accelerates development" to "AI accelerates writing code, but slows down delivery of quality software."

### 7.2. The Verification Tax

Time spent verifying AI code is becoming the dominant cost item. An experienced developer with AI works fast at the file level, but spends disproportionately more time reviewing AI-generated PRs from other team members.

METR found that experienced developers with AI access spent **6.5% more time on code review**.

This creates a structural paradox: AI benefits the person writing, and creates load for the person reviewing. In teams where reviewer and author are different people, this load transfer is invisible in individual metrics but visible in team metrics.

### 7.3. Skill Formation Risk

Anthropic's study (January 2026): the group with AI access scored **50%** on the understanding test versus **67%** for the control group — a gap of **17 percentage points**. The largest gap: debugging questions.

This matters because debugging is exactly the skill needed to **validate what AI generates**. Developers who use AI for learning become less capable of checking what AI produces. This is a positive feedback loop in an undesirable direction.

### 7.4. Security Does Not Improve with Larger Models

Veracode's Spring 2026 GenAI Code Security Update noted that security metrics "have barely moved" despite growth in model size and capability.

The reason — security requires understanding of the **threat model and context**, not just knowledge of vulnerability syntax. This is a structural limitation, not a capability gap that scales away.

---

## Conclusion

AI code is neither good nor bad. It is **optimal for a task at a point**, and suboptimal for the system as a whole. This is a fundamental property, not a temporary shortcoming.

The analysis of the specific code in this article shows: AI wrote a structurally readable application, correctly applied `dataclass`, `Enum`, and `ABC`, and complied with LSP. This is a good prototype. But the same code contains:
- A DIP violation making it untestable
- Absent error handling making it brittle
- A God Object pattern making it hard to extend

**Mathematically:** AI always finds a local minimum. Production requires the global one. The gap between them is the verification cost borne by the human.

### Three Practical Conclusions

1. **Use AI as a draft generator, not as a final arbiter of quality.** Code review with system context is mandatory — not optional.

2. **For local use, the code from the example works.** Add only HTTP error handling — this is the only problem that hurts immediately. Everything else is technical debt that manifests at scale.

3. **For production, refactor before adding new components.** DI, graceful shutdown, rate limiting, observability — these are not options, they are the foundation. The cost of adding them after the system has grown is quadratic.

> **The best code reviewer in the AI era is not someone who can quickly read AI code. It is someone who knows how to ask the questions AI never asks:**
> *"What happens when this fails?"*
> *"Who owns this invariant?"*
> *"How will this behave with 10,000 concurrent users?"*

---

## References

### Academic Papers

- Cotroneo, Improta, Liguori. *Human-Written vs. AI-Generated Code: A Large-Scale Study of Defects, Vulnerabilities, and Complexity.* arXiv:2508.21634, August 2025.
- Perry et al. *Do Users Write More Insecure Code with AI Assistants?* arXiv:2211.03622, ACM CCS 2023.
- Pearce et al. *Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions.* IEEE S&P 2022.
- Xu et al. *AI-Assisted Programming Decreases the Productivity of Experienced Developers.* arXiv:2510.10165, October 2025.
- CMU. *Speed at the Cost of Quality: How Cursor AI Increases Short-Term Velocity and Long-Term Complexity.* arXiv:2511.04427, November 2025.
- Anthropic. *How AI assistance impacts the formation of coding skills.* anthropic.com/research, January 2026.

### Industry Reports

- CodeRabbit. *State of AI vs Human Code Generation Report.* December 2025.
- Veracode. *GenAI Code Security Report 2025.* July 2025.
- Veracode. *Spring 2026 GenAI Code Security Update.* March 2026.
- GitClear. *AI Copilot Code Quality: 2025 Data Suggests 4x Growth in Code Clones.* February 2025.
- Google DORA. *Accelerate State of DevOps Report 2025.* Google Cloud.
- Cortex. *Engineering in the Age of AI: 2026 Benchmark Report.*
- METR. *Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity.* July 2025.
- Apiiro. *4x Velocity, 10x Vulnerabilities: AI Coding Assistants Are Shipping More Risks.* September 2025.
- Faros AI. *The AI Productivity Paradox Research Report.* 2026.
- Sonar. *State of Code Developer Survey Report 2026.*
- Stack Overflow. *2025 Developer Survey.* December 2025.
- Qodo. *State of AI Code Quality 2025.* June 2025.

---