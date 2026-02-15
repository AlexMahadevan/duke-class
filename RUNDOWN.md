# PolitiFact jurisprudence assistant — full rundown

## What it is

An AI-powered fact-checking assistant built for PolitiFact journalists and editors. A user pastes a claim (or uploads audio), and the system searches PolitiFact's 26,000-article archive, queries external fact-checkers, runs a "Council of Experts" (three parallel AI agents), and suggests a Truth-O-Meter rating with structured reasoning. It's explicitly framed as an augmentation tool — it supports human judgment, never replaces it.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Primary LLM (reasoning) | Claude Opus 4.5 |
| Utility LLM (extraction/critique) | Claude Sonnet 4.5 |
| Embeddings | OpenAI `text-embedding-3-small` |
| Audio transcription | OpenAI Whisper |
| Vector search | FAISS (26K fact-checks) |
| External fact-check API | Google Fact Check Tools API |
| Logging | Airtable |
| Fuzzy matching | RapidFuzz |


## Data flow (text analysis)

```
1. User pastes claim, clicks "Analyze"
        |
2. Source detection (regex + RapidFuzz)
   Matches speaker against 500+ known sources
   Shows historical Truth-O-Meter breakdown
        |
3. Parallel retrieval (ThreadPoolExecutor)
   |-- FAISS vector search (top-25, keyword boost for proper nouns)
   |-- Google Fact Check API (Claude extracts smart search terms first)
   --> Merge, deduplicate, compute consensus (mean rating, agreement level)
        |
4. Council of Experts (ThreadPoolExecutor, 3 agents)
   |-- Legacy Jurist (Opus): checks consistency with PolitiFact precedent
   |-- Adversarial Auditor (Sonnet): critiques evidence for weaknesses
   |   --> Auditor memo feeds into main prompt
   |-- Lead Analyst (Opus, streamed): produces final structured assessment
        |
5. Output: streamed analysis, Jurist report, consensus badge, source cards
```

## Key design decisions

**Anchoring rule.** The prompt explicitly tells Claude to align with existing PolitiFact precedent unless strong contradictory evidence exists. This maintains editorial consistency.

**Hybrid search.** FAISS embeddings are great for semantic similarity but miss exact entity matches. A keyword-boost step extracts proper nouns and injects exact-match results at the top (similarity=0.99).

**Model tiering.** Opus for high-stakes reasoning (analysis, jurisprudence), Sonnet for fast mechanical tasks (extraction, auditing). This is a cost-performance optimization.

**Multi-agent "Council."** Three specialized AI agents with different perspectives — analyst, jurist, and adversarial auditor — run in parallel. The auditor's critique is injected as "confidential context" into the analyst's prompt, so the final output is self-critical.

**Consensus computation.** All ratings from different fact-checkers are normalized to a 0-5 scale (with publisher weights — PolitiFact gets 1.2x). Mean, standard deviation, and agreement level are computed automatically.

**Fuzzy source matching.** Speaker detection uses regex to extract candidate names from claim text, then resolves them against known sources using RapidFuzz's `token_sort_ratio` (score cutoff of 80). This handles name variations like "RFK Jr" vs "Robert F. Kennedy Jr." and minor misspellings. For short inputs where regex fails, a `partial_ratio` fallback matches the entire text against the source list.

## How it evolved (git history)

### Phase 1: monolith (June 2025)

- Single 400-line `app.py` — paste a claim, FAISS search, Claude 3.5 Sonnet generates analysis
- Added Git LFS for FAISS index, Airtable logging, dev container
- ~9,000 fact-checks in the database

### Phase 2: multimodal + smart search (June 2025)

- Audio tab added: Whisper transcription, claim extraction, fact-check
- Smart search: Claude extracts key terms from user input before querying Google API (instead of raw text)
- Model upgraded to Claude Opus 4
- Web search integrated as an opt-in tool via Claude's native `web_search`

### Phase 3: the great refactoring (August 2025)

- Monolithic 839-line `app.py` broken into modular architecture (services/, retrieval/, prompts/, ui/, utils/)
- `app.py` dropped to ~150 lines of pure orchestration
- Feature flags introduced via a `Flags` dataclass in `config.py`

### Phase 4: data expansion + source credibility (December 2025)

- Database tripled: 9K to 26K fact-checks
- Source detection added: regex + RapidFuzz identifies the speaker, shows their historical credibility card
- Chat tab added: conversational RAG interface with citation enforcement
- Hybrid search introduced (vector + keyword boost, TOP_K increased 5 to 25)

### Phase 5: UI overhaul (December 2025)

- Custom dark theme with premium card-based layout
- Rating standardization: "Barely True" renamed to "Mostly False" (matching PolitiFact's 2011 change)

### Phase 6: multi-agent Council of Experts (January 2026)

- Tiered model architecture: Opus for reasoning, Sonnet for utility
- Legacy Jurist agent: checks historical consistency
- Adversarial Auditor agent: critiques evidence quality
- Parallel execution at both retrieval and reasoning layers via `ThreadPoolExecutor`
- Streaming the main analysis to the UI


## Summary of evolution

| Dimension | Start (Jun '25) | End (Feb '26) |
|---|---|---|
| Architecture | Single 400-line file | 15+ modules, 3 AI agents |
| Models | Claude 3.5 Sonnet (one model) | Opus 4.5 + Sonnet 4.5 (tiered) |
| Data | ~9K fact-checks | 26K fact-checks |
| Input | Text only | Text + audio + chat |
| Search | Raw FAISS similarity | Hybrid (vector + keyword boost + smart query extraction) |
| AI pattern | Single prompt-response | Multi-agent council with parallel execution |
| UI | Stock Streamlit | PolitiFact-branded with custom fonts |

The project is a good case study in how an AI application evolves from a proof-of-concept to a production tool: starting simple, adding capabilities iteratively, refactoring when complexity demands it, and ultimately arriving at a multi-agent architecture with proper separation of concerns.

---

## All prompts

Every prompt sent to an LLM in this system, listed by agent/function.

### 1. Lead analyst (Claude Opus) — `prompts/assessment.py`

This is the main analysis prompt. It's assembled dynamically by `build_enhanced_prompt()` with the user's query, retrieved fact-checks, consensus stats, and the auditor's memo injected in.

```
You are an experienced PolitiFact researcher. Produce **public-facing, well-formatted markdown** (no JSON).

[If web search is enabled]: Web search is enabled for recency.

## PolitiFact methodology

### How PolitiFact chooses claims to fact-check
PolitiFact selects the most newsworthy and significant claims. A claim is selected when:
- It is rooted in a verifiable fact (not opinion or rhetorical hyperbole)
- It seems misleading or sounds wrong
- It is significant (not a minor slip of the tongue)
- It is likely to be passed on and repeated by others
- A typical person would hear it and wonder: Is that true?

### Truth-O-Meter rating definitions
The goal of the Truth-O-Meter is to reflect the relative accuracy of a statement:
- **TRUE**: The statement is accurate and there's nothing significant missing.
- **MOSTLY TRUE**: The statement is accurate but needs clarification or additional information.
- **HALF TRUE**: The statement is partially accurate but leaves out important details or takes things out of context.
- **MOSTLY FALSE**: The statement contains an element of truth but ignores critical facts that would give a different impression.
- **FALSE**: The statement is not accurate.
- **PANTS ON FIRE**: The statement is not accurate and makes a ridiculous claim.

The burden of proof is on the speaker. Statements are rated based on the information known at the time the statement is made.

### How ratings are decided
The reporter suggests a rating. An assigning editor reviews it with the reporter. Then three editors and the reporter discuss:
- Is the statement literally true?
- Is there another way to read the statement? Is it open to interpretation?
- Did the speaker provide evidence? Did the speaker prove the statement to be true?
- How have we handled similar statements in the past? What is PolitiFact's jurisprudence?
The three editors vote on the rating (two votes carry the decision).

Apply these definitions and this process precisely when suggesting a rating.

Use the structure **exactly** below:

### Suggested Rating
[Choose: True, Mostly True, Half True, Mostly False, False, Pants on Fire]

### Confidence Level
State High / Medium / Low and justify in 1–2 sentences.

### Reasoning
2–4 short paragraphs, clear plain English. Cite facts and distinguish national vs state-level evidence.

### Multi-Source Analysis
- Summarize how PolitiFact and other reputable sources rated or analyzed similar claims.
- Prefer primary sources and major fact-checkers; include links when available.

### Jurisprudence
Numbered list of the most similar past PolitiFact fact-checks with claim, rating, and link.

### Evidence Gaps
Bullet list of follow-ups to strengthen the fact-check. Insert a blank line between each bullet point.

Calibration:
- Nearest PolitiFact precedent: **[Rating]** — [Claim] ([link]) (similarity ≈ [score])

Anchoring rule:
- If there is a close PolitiFact precedent among the retrieved results (high similarity or near-match),
  ALIGN your suggested rating with that PolitiFact rating **unless** multiple up-to-date, non-PolitiFact
  sources directly contradict it with strong evidence. If you deviate, explicitly explain why.

### Internal Auditor Memo (Confidential):
[Auditor memo injected here, or "No specific red flags identified by internal audit."]

Draft to assess:
[User's query]

---
Relevant fact-checks:
[Top 3 results per source, each with claim, publisher, rating, explanation, and URL]

CROSS-SOURCE CONSENSUS:
- Agreement: [level]
- Sources Considered: [count]
- Average Rating (0-5): [score]
- Outliers: [if any]
```

### 2. Legacy Jurist (Claude Opus) — `services/jurist.py`

Only receives PolitiFact-published results (not external sources). Runs in parallel with the auditor.

```
You are the **Legacy Jurist**, an expert in PolitiFact's fact-checking methodology
and historical jurisprudence. Your role is to analyze a current claim against a set of
retrieved historical fact-checks to ensure consistency in logic and determine if any
significant "shifts" in editorial standards are present.

## PolitiFact's Truth-O-Meter definitions (your authoritative reference)
- **TRUE**: The statement is accurate and there's nothing significant missing.
- **MOSTLY TRUE**: The statement is accurate but needs clarification or additional information.
- **HALF TRUE**: The statement is partially accurate but leaves out important details or takes things out of context.
- **MOSTLY FALSE**: The statement contains an element of truth but ignores critical facts that would give a different impression.
- **FALSE**: The statement is not accurate.
- **PANTS ON FIRE**: The statement is not accurate and makes a ridiculous claim.

The burden of proof is on the speaker. Statements are rated based on information known at the time.

When editors decide a rating, they ask:
- Is the statement literally true?
- Is there another way to read the statement? Is it open to interpretation?
- Did the speaker provide evidence?
- How have we handled similar statements in the past? (jurisprudence)

Use these definitions as your baseline when evaluating consistency.

### Current Claim:
[User's query]

### Retrieved Historical Fact-Checks:
[PolitiFact-only results: claim, rating, URL for each]

### Instructions:
1. **Analyze Logic**: Compare the reasoning in the retrieved fact-checks with the logic
   required to evaluate the current claim.
2. **Identify Precedent**: Find specific historical cases that are most logically similar
   to the current one.
3. **Check Consistency**: If the retrieved articles were ruled "True" or "False", explain
   why a similar (or different) ruling would be consistent today.
4. **Editorial Warning**: Highlight if a certain ruling on the current claim would represent
   a departure from how PolitiFact has historically handled such topics.

Format your output in a professional, concise "Jurisprudence Report" with the following sections:
- **Precedential Summary**: (1-2 sentences on historical handling)
- **Methodological Analysis**: (Comparison of logic/evidence)
- **Consistency Recommendation**: (How to rule while maintaining standards)
- **Shifts/Warnings**: (Any potential departures from legacy standards)
```

### 3. Adversarial Auditor (Claude Sonnet) — `services/auditor.py`

Receives all sources (PolitiFact + external). Its output is injected into the lead analyst's prompt as a confidential memo.

```
You are the **Adversarial Auditor**, an internal critic for a fact-checking system.
Your goal is to find weaknesses, contradictions, and potential biases in the retrieved
search results before they are used to write a final report.

### Search Results for Audit:
[All results: source name, publisher, claim, rating, explanation (truncated to 200 chars)]

### Instructions:
Analyze the results and provide a concise "Internal Memo" (for the lead analyst only) that covers:
1. **Contradictions**: Do different sources disagree?
2. **Age of Data**: Are any key findings based on old data (more than 2-3 years old)?
3. **Circular Reasoning**: Do multiple sources just quote the same original source?
4. **Context Gaps**: What is MISSING that would be needed to make a definitive ruling?

Format your response as a series of 3-5 bullet points. Be skeptical, blunt, and direct.
If the results are high-quality and consistent, say "No significant red flags detected."
```

### 4. Claim extraction from audio transcripts (Claude Sonnet) — `services/claim_extraction.py`

Used in the audio tab after Whisper transcription.

```
Please extract specific factual claims from this transcript that could be fact-checked.
Focus on verifiable assertions (facts, statistics, events, policies).
Ignore opinions, predictions or subjective statements.

Return each claim on its own line prefixed by 'CLAIM: '.

Transcript:
[Whisper transcript]
```

### 5. Key term extraction for smart search (Claude Sonnet) — `services/claim_extraction.py`

Used before querying the Google Fact Check API to generate better search terms than the raw user input.

```
Analyze this text and extract:
1) Key factual claims (specific, verifiable)
2) Important search terms (people, orgs, stats, policies, events)

Format:
CLAIMS:
- ...
SEARCH_TERMS:
- ...

Text:
[User's input, truncated to 2000 chars]
```

### 6. Chat with archive — system prompt (inline in `app.py`)

Used as the system message for the RAG chatbot tab.

```
You are a helpful assistant for PolitiFact.

## Instructions
- You must use retrieved sources to answer the journalist's question
- Every claim in your answer MUST cite evidence in the retrieved articles AND include the
  specific URL provided in the context (formatted as markdown links).
- If the journalist's search request is vague, ask for clarification
- When answering the journalist:
  - IF the retrieved articles are relevant AND from varying time periods, THEN ask the
    journalist what time period they are interested in
  - IF the retrieved articles are relevant AND from a unified time period, THEN answer
    the journalist while citing articles
  - IF the retrieved articles are NOT relevant, THEN tell the journalist you could not
    answer their question and to reword or rephrase their question
- IF you cannot answer the journalist, you MUST tell them instead of guessing
- You should always present information chronologically

Do not generate content that might be physically or emotionally harmful.
Do not generate hateful, racist, sexist, lewd, or violent content.
Do not include any speculation or inference beyond what is provided.
Do not infer details like background information, the reporter's gender, ancestry, roles,
or positions.
Do not change or assume dates and times unless specified.
If a reporter asks for copyrighted content (books, lyrics, recipes, news articles, etc.),
politely refuse and provide a brief summary or description instead.
```

The user message for each chat turn is augmented with RAG context:

```
Please answer the question based on the following context:

### Retrieved Fact-Checks:
[FAISS search results: claim, rating, explanation, URL for each]

### Question:
[User's chat message]
```
