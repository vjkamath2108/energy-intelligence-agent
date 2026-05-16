# Sovereign Inference Architecture
## A Decision Framework for UK Energy Operators

> This document is not a prescription. It is a decision framework.  
> The right answer depends on your client's regulatory obligations, risk appetite,  
> and operational maturity — not on a universal rule.

---

## 1. What Data Sovereignty Actually Means

### The misconception that costs money

Ask most engineers where their data lives and they'll point to their S3 bucket.
"Our data is in eu-west-2. We're compliant."

They're half right.

Data **at rest** in an eu-west-2 S3 bucket does stay in London. That part is
correct. But data sovereignty is not just about where data is stored. It is about
where data is **processed**, by **whom**, under **what jurisdiction**, with what
**audit trail**.

The moment you call an inference API, you are not querying the warehouse.
You are taking data out of the warehouse, putting it in an envelope, and
sending it to wherever the inference endpoint lives. The question is not
where the envelope came from. The question is where it goes, who handles it,
and what happens to it when it gets there.

### The envelope analogy — and its limits

Consider a UK energy operator storing grid operational data in S3 eu-west-2.
They call a `us.*` cross-region Bedrock inference profile to generate a dispatch
recommendation.

What actually happens:

1. The prompt — containing operational data about UK grid state — leaves eu-west-2
2. AWS routes it to whichever US region has capacity: us-east-1, us-west-2, or us-east-2
3. The data is loaded into GPU memory, processed, and a response is generated
4. The response travels back to eu-west-2
5. The prompt data is not written to disk — AWS does not store it after inference

The envelope goes to America. It comes back. Nothing is left behind.

So is that a problem?

**That depends on a question only your client can answer.**

### The transient processing question

AWS's Bedrock data privacy commitments are explicit: prompt data is not used
for training, not stored after inference, not accessible to AWS employees.
These commitments are real, contractually binding, and auditable to a degree
via AWS CloudTrail.

But UK GDPR Article 4 defines "processing" to include use and consultation —
not just storage. The moment your data is loaded into memory in us-east-1 and
processed, that is a restricted international transfer under UK GDPR,
regardless of whether it is subsequently stored.

So the legal answer is: **transient processing in a foreign jurisdiction is
still processing in a foreign jurisdiction.**

The practical answer is: **most enterprise customers accept this trade-off,
with appropriate legal cover.**

The question your client needs to answer is not "is this technically a data
transfer?" — it is — but "what level of assurance do we need, and what are
we willing to trade for it?"

### Three types of customers

In practice, you will encounter three types of customers when this question
comes up:

**The pragmatic customer** has reviewed AWS's data processing terms, understands
the data is transient, has accepted the UK GDPR transfer risk under Standard
Contractual Clauses, and their legal team is comfortable. They want Bedrock in
eu-west-2 and they want to move. This is the majority of enterprise customers.

**The sovereignty-first customer** will not accept any processing outside their
own infrastructure, transient or not. They need to tell their regulator that
the data never left their VPC — full stop. Contractual commitments from AWS are
not sufficient; they need technical proof. This is the NIS2-obligated energy
operator, the nuclear-adjacent client, the organisation whose legal team has
drawn a hard line.

**The confused customer** believes that storing data in eu-west-2 means their
AI system is compliant. They have not yet separated the question of data at rest
from data in transit and data in processing. They need the envelope explained
before they can make an informed decision about which of the above two categories
they actually belong in.

---

## 2. Why UK Energy Operators Face Specific Sovereignty Requirements

### Not all industries are equal under data law

GDPR applies broadly across sectors. But UK energy operators face a layer of
obligation that most industries do not — one that makes the data sovereignty
question sharper, more urgent, and less negotiable.

That layer is NIS2.

### NIS2 — the regulation most AI architects haven't read

The Network and Information Security Directive 2 (NIS2), transposed into UK law
post-Brexit as the Network and Information Systems Regulations, classifies energy
operators as **critical national infrastructure**. This has three specific
implications for AI system design that go beyond GDPR:

**1. Supply chain security obligations**

NIS2 requires operators to assess and manage security risks in their supply chain.
An external inference API endpoint is a supply chain dependency. If Anthropic's
API or AWS Bedrock experiences an outage, a security incident, or a policy change,
your client's AI system is affected. NIS2 obligates the operator to have assessed
that risk and documented their mitigation.

Self-hosted inference eliminates this dependency entirely. The inference endpoint
is inside the client's own infrastructure — they control uptime, security patching,
and availability.

**2. Incident reporting obligations**

NIS2 imposes strict incident reporting timelines. If a security incident involves
data that was processed by an external provider, the operator needs to be able to
describe exactly what data was exposed, where it was, and who had access. "It was
in Bedrock's inference pipeline transiently" is a harder answer to give a regulator
than "it never left our VPC."

**3. Proportionate security measures**

NIS2 requires operators to implement security measures proportionate to the risk.
For operational data about UK grid dispatch decisions — which assets are online,
which are being shed, what the reserve margin is — the risk classification is high.
Sending this data to an external API, even transiently, requires a proportionate
justification.

### GDPR vs NIS2 — the key distinction

These two frameworks ask different questions:

| Framework | Primary question | Unit of concern |
|---|---|---|
| UK GDPR | Does this involve personal data? | The individual |
| NIS2 | Does this involve critical infrastructure operations? | National security |

An energy operator's dispatch recommendations may contain **no personal data
whatsoever** — no names, no account numbers, no consumption data tied to
individuals. They are still subject to NIS2 obligations because the data concerns
the operation of critical national infrastructure.

This matters because many operators — and their legal teams — approach AI
compliance purely through a GDPR lens. They ask "does this contain personal data?"
find the answer is no, and conclude they are compliant. NIS2 asks a different
question entirely.

### The sub-processor question revisited

Under GDPR, when you call Bedrock, **AWS is the data processor** — not Anthropic.
The data is processed on AWS infrastructure, in an AWS region, under AWS's data
processing terms. Anthropic provides the model weights; they do not see your
inference data.

This is materially different from calling `api.anthropic.com` directly, where
Anthropic becomes the data processor and a separate DPA with Anthropic is required.

For most enterprise clients with existing AWS agreements, Bedrock's data processing
terms are already covered. The question is not "do we need a DPA?" — you do, and
it's in your AWS agreement — but "does processing in a US region, transiently,
under AWS's terms, satisfy our NIS2 obligations?"

For the pragmatic customer: probably yes, with appropriate documentation.

For the sovereignty-first customer: no. They need to eliminate the external
dependency entirely, not negotiate better terms with it.

### What energy operators are actually worried about

In practice, when energy clients push back on external inference APIs, the concern
is rarely a precise citation of NIS2 Article 21. It is usually one of three things:

**"Our data about grid operations cannot leave our network."** This is the NIS2
supply chain concern expressed in plain English. The answer is EKS self-hosting
or a private Bedrock endpoint — depending on their definition of "our network."

**"We don't want our data training future models."** This is training data anxiety.
AWS's Bedrock terms explicitly prohibit this. But for a sovereignty-first customer,
a contractual commitment is not the same as a technical guarantee. Self-hosting
provides the technical guarantee.

**"Our regulator will ask us where our AI is running."** This is the audit
question. Bedrock in eu-west-2 with CloudTrail gives a defensible answer.
VPC Flow Logs on an EKS cluster give a verifiable answer. The difference matters
to some regulators and not to others — your client's compliance team needs to
make that call.

---

## 3. The Three-Way Decision Framework

There are four realistic inference deployment options for an organisation
building an AI system today. Each makes a different set of trade-offs. None is
universally correct.

| | Bedrock (AWS-managed) | EKS cloud-sovereign (vLLM) | Air-gapped K8s (on-prem) | Ollama (dev only) |
|---|---|---|---|---|
| Data processor | AWS | None (client owns) | None (client owns) | None |
| Data residency | AWS region (configurable) | Client VPC | Client data centre | Client machine |
| Transient cross-boundary processing | Yes, if cross-region profile | No | No | No |
| Audit proof | AWS CloudTrail | VPC Flow Logs | Physical network isolation | Not applicable |
| Model quality | Full Claude Sonnet | Open-source (size-dependent) | Open-source (size-dependent) | Open-source |
| JSON validity (this benchmark) | 100% | — | — | phi4-mini: 100%, qwen3:14b: 51%* |
| Severity accuracy (this benchmark) | 82% | — | — | phi4-mini: 96%, llama3.2:3b: 17% |
| Latency P50 (this benchmark) | 5145ms | ~200-500ms on GPU node | ~200-500ms on GPU node | phi4-mini: 1559ms† |
| Latency use case | General purpose | Voice bots, real-time | Voice bots, real-time | Dev evaluation only |
| Cost crossover | Baseline (pay per token) | Depends on volume, model, utilisation | Depends on hardware amortisation | ~$0 |
| Ops burden | Low | High | Very high | Very low |
| Usage governance | CloudTrail + CloudWatch + resource tags | LiteLLM Proxy (VPC-bound) | LiteLLM Proxy (air-gapped cluster) | Local only; not scalable |
| NIS2 supply chain dependency | Yes | No | No | No |
| Suits | GDPR-comfortable, cloud-native | NIS2-obligated, cloud-connected | Hard air-gap requirement | Development only |
| Production-ready | Yes | Yes | Yes | No |

*qwen3:14b JSON validity affected by thinking-mode token budget — see benchmark report.
†On CPU/Metal. On GPU node expected ~200-500ms.

### Reading the table

The table does not have a winning column. It has a column that is right for
each client depending on where they sit in the decision framework from Section 1.

What follows is the reasoning behind each option — not a recommendation,
but the questions that determine whether it is the right choice.

---

### Option 1: Bedrock (AWS-managed)

**What it is:** Claude Sonnet running on AWS-managed infrastructure. You call an
endpoint, AWS processes the inference, you get a response. No infrastructure to
manage.

**The residency question:** Bedrock supports region-specific endpoints. Calling
bedrock-runtime.eu-west-2.amazonaws.com with a provisioned throughput endpoint
keeps inference in London. Using a us.* cross-region inference profile routes
to US regions for capacity — data crosses jurisdictional boundaries transiently.

For UK energy clients, the choice of endpoint matters more than the choice of
Bedrock itself. A eu-west-2 provisioned throughput endpoint with an eu.*
inference profile is a materially different compliance posture than a us.*
cross-region profile.

**The right questions for your client:**
- Does your legal team accept transient processing in AWS eu-west-2 under AWS DPA?
- Are your prompts free of personal data, or do you have SCCs in place for international transfers?
- Can you document the Bedrock endpoint configuration and region to satisfy a NIS2 audit?
- Is the supply chain dependency on AWS acceptable given your NIS2 obligations?

**When Bedrock is the right answer:** The client has reviewed AWS data processing
terms, selected an appropriate region, and their legal team is comfortable with
AWS as data processor. They want production-grade model quality without
infrastructure overhead. Their compliance posture does not require eliminating
the external dependency — only controlling where it sits.

---

### Option 2: EKS Cloud-Sovereign (vLLM)

**What it is:** An open-source model running on the client's own Kubernetes
cluster, inside their own VPC in AWS. The inference endpoint is
http://inference-svc.prod.svc.cluster.local — it never resolves outside
the cluster. No data leaves the client's VPC boundary. The cluster is
cloud-connected but network-isolated for inference traffic.

**The sovereignty argument:** This is the only cloud option that provides a
**technical** guarantee of data residency rather than a contractual one.
VPC Flow Logs give auditable proof that no inference traffic crossed the VPC
boundary. A regulator can verify this independently — they are not relying on
AWS's contractual commitment, they are reading network logs.

**The latency argument:** Sovereignty is not the only reason to choose this
option. For latency-sensitive applications — voice bots, real-time dispatch
interfaces, streaming responses — Bedrock's P50 of ~5s is a non-starter.
A GPU-hosted model running inside the same VPC as your application eliminates
the internet round-trip and cross-region routing entirely. Inference latency
drops to ~200-500ms on a GPU node. This is a performance architecture
argument entirely independent of compliance.

**The application code does not change.** The inference endpoint URL is the
only difference between development (Ollama on localhost) and EKS sovereign
mode (vLLM inside the VPC). Same Pydantic output schema. Same application
flag. The sovereignty boundary is a deployment configuration, not an
architecture change.

**The model quality question:** This benchmark tested phi4-mini (3.8B) locally.
phi4-mini achieved 96% severity accuracy — higher than Bedrock at 82% on this
task. Sovereign inference does not require sacrificing quality. A well-chosen,
instruction-tuned open-source model on appropriate hardware is competitive with
cloud inference on structured tasks.

**The cost crossover — how to think about it:**

EKS self-hosting has a fixed daily cost — the GPU node runs whether you make
one inference call or a million. Bedrock has a variable cost that scales
linearly with usage. These two cost curves cross at some volume that depends
entirely on your specific situation.

The variables that determine where your crossover sits:
- **Bedrock cost per token** — check current pricing for your model and region
- **GPU node cost** — instance type, On-Demand vs Spot, Reserved commitment
- **Node utilisation** — 20% utilisation vs 80% changes effective cost per token significantly
- **Model size** — phi4-mini runs on a smaller, cheaper node than qwen3:14b
- **Engineering overhead** — model updates, security patching, on-call, capacity planning

To work through your specific volume: take your current Bedrock cost as baseline,
model the EKS alternative at your required latency and utilisation, add engineering
overhead honestly, and compare total cost of ownership — not just compute cost.
If EKS is cost-competitive, sovereignty becomes a free benefit. If it is
materially more expensive and sovereignty is not a hard requirement, Bedrock
with an appropriate regional endpoint is the rational choice.

**The right questions for your client:**
- Do you have a platform team capable of managing GPU nodes, model updates, and vLLM?
- Can you accept open-source model quality, or do you need Claude Sonnet specifically?
- Have you modelled your inference volume against EKS node costs at your required utilisation? Is self-hosting cost-competitive given your specific volume and engineering overhead?
- Does your regulator require technical proof of data residency rather than a contractual commitment?
- Do you have latency requirements that Bedrock cannot meet?

**When EKS cloud-sovereign is the right answer:** The client has NIS2 obligations
that preclude external API dependencies, or requires technical proof of data
residency, or has latency requirements that cloud inference cannot satisfy —
**and** has the operational maturity to run GPU infrastructure in their VPC.

The last condition is non-negotiable. Self-hosting for a client without a
mature platform team is not sovereignty — it is a different kind of risk.

---

### Option 3: Air-Gapped Kubernetes (On-Premise)

**What it is:** An open-source model running on a Kubernetes cluster in the
client's own data centre or private cloud, with no internet connectivity.
The same vLLM stack as Option 2 — the difference is physical: there is no
network path to the outside world. Sovereignty is enforced by the absence
of connectivity, not by network policy.

**When the air-gap is required:** For organisations where cloud connectivity
is contractually or regulatorily prohibited — certain FSI contexts, gambling,
defence-adjacent, critical national infrastructure with the highest sensitivity
classifications — the cloud-sovereign EKS option is not sufficient. The
regulator or legal constraint is not "data must stay in our VPC" but "data
must never traverse a network we do not physically control."

An air-gapped Kubernetes cluster satisfies this requirement. The sovereignty
argument is the strongest available: there is no network, therefore no
network-based data exposure is possible.

**The operational reality:** This is the highest-burden option. Model updates
cannot be pulled from a registry — they must be distributed via internal
tooling or physical media. The cluster has no access to external package
repositories, security feeds, or vendor support APIs. The platform team
must be fully self-sufficient.

Kubernetes is still the right foundation for this environment — self-healing,
scaling, GPU management, and rolling deployments are just as valuable in an
air-gapped cluster as in EKS. The difference is everything that normally
comes from external connectivity must be sourced internally.

**The right questions for your client:**
- Is cloud connectivity genuinely prohibited, or is VPC-isolated EKS actually sufficient?
- Do you have the operational maturity to run a fully self-sufficient Kubernetes platform?
- How will you manage model updates, security patches, and capacity changes without internet access?
- Is the compliance benefit worth the significantly higher operational cost?

**When air-gapped Kubernetes is the right answer:** The client's regulatory or
contractual constraints explicitly prohibit cloud connectivity, not just
external data processing. They have a mature platform team capable of operating
without external dependencies. The compliance requirement cannot be satisfied
by network isolation alone — it requires physical isolation.

---

### Option 4: Ollama (Development Only)

**What it is:** An open-source model running on a developer's local machine.
Zero cloud calls. Zero external network dependency. Zero production relevance.

Ollama's role is specific: it proves the sovereign inference pattern works
before committing to Kubernetes infrastructure. The same application code
that calls Ollama on localhost will call vLLM on EKS or an air-gapped cluster
— the endpoint URL is the only difference.

This benchmark ran llama3.2:3b, phi4-mini, and qwen3:14b locally via Ollama.
The findings directly inform model selection for Options 2 and 3. phi4-mini
at 96% accuracy and ~1.5s latency on a laptop is a strong candidate for
GPU-node deployment — fast, accurate, and small enough to run cost-effectively
on shared hardware.

**When Ollama is the right answer:** Local development and model evaluation.
Proving the sovereign inference pattern before infrastructure commitment.
Never in production — not because Ollama cannot run a model reliably, but
because it provides none of the operational guarantees (self-healing, scaling,
governance, auditability) that production requires.

---

## 4. What Sovereign Inference Actually Looks Like

### The key principle

Sovereign inference is a deployment decision, not an application decision.

The application code does not change between a Bedrock deployment and a
sovereign self-hosted deployment. The model, the prompt, the output schema,
the retry logic — all identical. What changes is a single configuration:
the endpoint URL the application calls.

In development, that URL points to Ollama on a developer's laptop.
In production sovereign mode, it points to a vLLM service running inside
the client's own infrastructure — their VPC or their data centre.
The application has no awareness of which environment it is in.

This matters for the architecture conversation: adopting sovereign inference
does not require rewriting applications. It requires deciding where the
inference endpoint lives and who operates it.

### What "proof of sovereignty" actually means

This is where Bedrock and self-hosted inference differ in a way that matters
to regulators and compliance teams — not just architects.

With Bedrock, the proof that data stayed within your chosen region is
**contractual**. AWS commits in their data processing terms that inference
data is not stored, not used for training, and processed in the region you
configured. That commitment is real and legally binding. But it is
AWS's word, verified by AWS's logs.

With self-hosted inference inside your own infrastructure, the proof is
**technical**. Network traffic logs show every connection that entered and
left your environment. A regulator can verify independently — from your
own logs, not from a vendor's assurance — that no inference traffic crossed
your boundary. There is no trust relationship required. The logs speak for
themselves.

For most organisations, contractual proof is sufficient. For organisations
with the highest sensitivity classifications — certain FSI contexts, critical
national infrastructure, defence-adjacent — technical proof is the only
proof that satisfies the requirement.

### The operational question you must answer first

Before choosing a sovereign deployment option, there is one question that
determines whether it is viable:

**Do you have a platform team capable of owning this?**

Self-hosted inference on Kubernetes — whether cloud-sovereign EKS or
air-gapped on-premise — requires a team that can manage GPU infrastructure,
handle model updates (model weights for a 14B model are 8-28GB and cannot
be updated the way you update application code), maintain the inference
service through hardware failures, and operate without the safety net of
a managed service.

This is not a reason to avoid self-hosting. It is a prerequisite to be
honest about before the decision is made. An organisation that chooses
EKS sovereign mode without a capable platform team has not gained
sovereignty — it has traded one risk for a larger one.

If the platform team does not exist today, the pragmatic path is Bedrock
with an appropriate regional endpoint while the team and processes are built.
Sovereignty is a destination, not a switch.

### What the governance boundary looks like

In a sovereign deployment, every component that touches inference data must
sit inside the sovereignty boundary — not just the model itself.

Observability tooling, usage logging, cost tracking — if these write to an
external SaaS platform, data about your inference calls is leaving your
environment. For a sovereignty-first client, the observability stack must
run inside the same boundary as the inference service.

This is a common oversight. Organisations invest in self-hosted inference
and then route their telemetry to an external monitoring platform, recreating
the data egress they set out to eliminate. The sovereignty boundary is only
as strong as its weakest egress path.

---

## 5. Making the Decision — A Diagnostic Framework

The sections above have laid out what data sovereignty means, why UK energy
operators face specific obligations, and what each deployment option provides.
This section puts it together as a decision framework.

Work through the questions in order. The first question that produces a clear
answer determines your path. If you reach the end without a clear answer,
the default is Bedrock with an appropriate regional endpoint — the lowest
operational risk while the remaining questions are resolved.

---

### Question 1: Is cloud connectivity an option at all?

**If no** — your regulatory or contractual constraints prohibit sending data
to any external network, regardless of region or provider — the answer is
**Option 3: Air-gapped Kubernetes**. No further questions needed on the
sovereignty dimension. The remaining questions to answer are operational:
do you have the platform team, and how will you manage model updates?

**If yes** — proceed to Question 2.

---

### Question 2: Does your data contain personal data, or does it concern critical national infrastructure operations?

This question determines which legal framework is primary.

**Personal data only (UK GDPR applies):** The question is whether you can
establish a lawful basis for international transfers if using a cross-region
inference profile. A eu-west-2 provisioned throughput endpoint with an eu.*
inference profile keeps processing in the UK and sidesteps the international
transfer question entirely. Standard Contractual Clauses cover the residual
risk. Bedrock with the right endpoint configuration is likely sufficient.

**Critical national infrastructure operational data (NIS2 applies):** The
question is whether you can document and justify the supply chain dependency
on an external inference provider to your regulator. If yes — Bedrock with
documented endpoint configuration and a NIS2 supply chain risk assessment
may be sufficient. If no — proceed to Question 3.

**Both:** Apply the more stringent framework. Proceed to Question 3.

**Neither:** Your data sovereignty obligations are minimal. Bedrock with
standard configuration is appropriate. Focus your energy on model quality
and latency rather than sovereignty architecture.

---

### Question 3: Does your regulator or legal team require technical proof of data residency, or is contractual assurance sufficient?

**Contractual assurance is sufficient:** AWS's data processing terms,
combined with a eu-west-2 or eu.* inference profile, provide the
contractual basis. **Bedrock (Option 1)** with documented endpoint
configuration is your answer. Ensure your compliance team has reviewed
and accepted AWS's DPA and that your endpoint configuration is locked
down — cross-region profiles should be explicitly disabled if data
residency is a requirement.

**Technical proof is required:** You need network-level evidence that
data did not leave your environment — evidence that lives in your own
logs, not in a vendor's assurance. Proceed to Question 4.

---

### Question 4: Do you have, or can you build, a platform team capable of operating self-hosted inference?

This is the operational maturity question. Be honest. Self-hosted inference
on Kubernetes requires GPU infrastructure management, model update processes,
capacity planning, and on-call for inference service failures. These are not
insurmountable — but they are real, and underestimating them is the most
common reason sovereign deployments fail.

**Yes, we have the capability:** Proceed to Question 5.

**Not yet, but we are building it:** The interim answer is **Bedrock with
a eu-west-2 provisioned throughput endpoint** while the team and processes
are established. Set a concrete milestone for when the sovereign deployment
will be ready, and use that time to evaluate models via Ollama so the model
selection decision is already made when the platform is ready.

**No, and we are unlikely to build it:** If technical proof of sovereignty
is required but the operational capability does not exist, this is a
strategic gap that needs to be resolved at the organisational level before
the architecture question can be answered. Managed sovereign inference
services — where a third party operates the self-hosted stack inside your
VPC — are an emerging option worth investigating.

---

### Question 5: What are your latency requirements?

**Latency is not a constraint (>3s acceptable):** Both Bedrock and
EKS cloud-sovereign are viable. The decision is driven by sovereignty
and cost considerations from the earlier questions.

**Sub-3s required (voice bots, real-time interfaces):** Bedrock's
current P50 latency makes it unsuitable for these use cases regardless
of sovereignty considerations. **EKS cloud-sovereign (Option 2)** with
a GPU node in the same VPC as your application is the path. Model
selection matters here — phi4-mini's performance profile in this benchmark
makes it a strong candidate for latency-sensitive deployments.

**Sub-1s required:** This requires careful hardware sizing and model
selection. Smaller, well-tuned models on dedicated GPU nodes. Evaluate
via Ollama first to establish the latency floor before committing to
infrastructure.

---

### The decision at a glance

| If your situation is... | The right option is... |
|---|---|
| Cloud prohibited by contract or regulation | Air-gapped Kubernetes (Option 3) |
| No personal data, no NIS2, cloud acceptable | Bedrock with standard configuration (Option 1) |
| GDPR personal data, cloud acceptable, eu-west-2 sufficient | Bedrock with eu-west-2 provisioned throughput (Option 1) |
| NIS2 obligations, contractual assurance sufficient, platform team exists | Bedrock with eu.* profile + documented risk assessment (Option 1) |
| Technical proof required, platform team exists, latency flexible | EKS cloud-sovereign (Option 2) |
| Technical proof required, platform team exists, sub-3s latency needed | EKS cloud-sovereign with GPU nodes (Option 2) |
| Technical proof required, platform team does not yet exist | Bedrock interim + build toward EKS |
| Model evaluation and development | Ollama (Option 4) |

---

### A note on starting points

Most organisations do not arrive at this framework before they have already
started building. They embrace AI, connect to Bedrock, and later discover
that their compliance team has questions. That is a normal sequence, not a
failure.

The value of this framework is not that it prevents those conversations —
it is that it structures them. A compliance team asking "where is our data
being processed?" deserves a precise answer, not a reassurance. This
framework gives you the vocabulary and the decision path to provide one.