# Argos-Agent Architecture, Backed by Tailwag Identity and Memory

## 1. Context

The Argos-agent architecture turns a Unitree Go2 robot into a voice-first, context-aware companion. It has two parts that should stay distinct: the realtime model, which handles conversation and reasoning, and the local harness around it, which owns robot-facing control. Together, they sit between people, the robot body, and the surrounding identity and memory systems.

That matters because a robot conversation is not just a chat session with speakers attached. The local harness has to handle mic admission, wake word and attention signals, face and voice ownership, playback, display state, tool calls, and physical motion without letting those concerns blur together. The full Argos-agent architecture is the realtime model plus that control harness around the persistent realtime session.

Figure 1. The Argos-agent architecture connects human interaction, the realtime model, robot behavior, identity, memory, and operator visibility.

![Argos system map](assets/argos_system_map.svg){width=6.8 height=3.8}

## 2. What Argos Can Do

The Argos-agent architecture combines conversation, perception, memory, embodiment, navigation, and tools into one interaction loop.

| Capability | What it does | Why it matters |
|---|---|---|
| Natural voice interaction | A person can speak to the robot and hear a spoken response from the realtime model. | Creates a direct conversational interface instead of a menu-driven one. |
| Local listening control | The robot uses wake word, voice activity, attention, and interaction state before it records a turn. | The robot is less likely to respond to random background speech. |
| Face awareness | The robot tracks whether people are visible and whether someone appears oriented toward it. | Presence can shape greetings, listening, and social context. |
| Speaker ownership | Completed speech can be compared against saved voice references and combined with face evidence. | The harness can decide whose conversation and memories are in scope. |
| Person memory | Owner-scoped context can be pulled from the memory system and new recognized conversations can be recorded. | The architecture can support memory without mixing people together. |
| Robot actions and tools | The model can call profile-enabled local tools for posture, gestures, short motion, camera capture, identity enrollment, employee lookup, and semantic memory search. | The model can act through the robot, but only through bounded interfaces. |
| Interaction display | A screen can show face state, subtitles, thinking/recording state, and enrollment review. | People can understand what the robot is doing and approve sensitive captures. |
| Observability | The harness records timing and usage markers around listening, model response, playback, tools, and memory queries. | Operators can debug whether the robot is slow, blocked, or using the wrong context. |
| Navigation | The navigation profile can expose named-location goals, relative movement, waypoint routes, cancellation, current-pose lookup, patrol stop, and charging dock tools. | Mobility is available through profile-gated tools, but the main readiness question remains interaction reliability. |

## 3. How A Conversation Turn Works

The realtime model is central to language and reasoning, but the local harness around it owns the robot-facing control loop.

Before the model responds, the harness has already made several concrete decisions. It evaluates local audio admission, captures and commits the intended speech turn, resolves the likely speaker, selects the owner-scoped history, builds the dynamic turn instructions, and then asks the model to answer. During and after the response stream, the harness owns playback, display updates, function-call execution, follow-up response creation, navigation event handling, and memory recording.

Figure 2. A human turn moves through local harness control before the model response is triggered.

![Argos turn flow](assets/argos_turn_flow.svg){width=6.8 height=3.35}

The harness keeps explicit state for engagement, capture, playback, tool calls, navigation, battery, display, and identity ownership. The model generates speech and tool intent; the harness decides when a turn exists, which context is eligible, which registered tools are available in the active profile, and how results become robot behavior.

## 4. Identity, Memory, And Trust

The trust problem for the Argos-agent architecture is not simply whether the robot recognizes someone. The harder problem is whether it attributes the conversation to the right person. A wrong answer can be corrected. A wrong memory attached to the wrong person is much more damaging.

The harness separates the identity problem into four parts:

| Layer | What it owns | Trust boundary |
|---|---|---|
| Face identity | Visual recognition from the camera, with optional depth gating and strict single-face owner selection; enrollment adds stronger quality gates and optional display review. | A face alone is not enough to decide who spoke. |
| Speaker identity | Voice matching from the completed spoken turn. | Voice evidence can win when strong; when weak, strict single-face ownership may still resolve the turn. |
| Person record | Name, aliases, and employee-style metadata for the person. | Identity data stays separate from social memory. |
| Social memory | Preferences, notes, follow-ups, and conversation episodes when memory is enabled. | Memory is scoped to the resolved speaker, not just whoever is visible. |

That separation is important. If one person is visible but another speaks off-camera, the harness should not attach the conversation to the visible person by default. If two faces are visible, it should avoid pretending the scene is simple. If voice evidence is weak and there is no strict single-face owner candidate, it is better to mark ownership as unknown than to save the wrong memory.

Enrollment follows the same trust logic. The robot can enroll a visible person, but the capture path is quality-gated, and the display can require an accept/reject review before saving a face reference. That makes enrollment feel less like a hidden background process and more like an explicit human-facing workflow.

## 5. Readiness And Evidence

The current Argos-agent architecture is broad enough to show the full product direction: realtime speech, attention-aware listening, face and voice identity, owner-scoped memory, display feedback, robot actions, optional navigation, provider-backed tools, and latency logging. The important next question is not "can another feature be added?" It is "which parts are reliable enough to trust repeatedly?"

Table 1. The next steps are to move these areas from described behavior to measured evidence.

| Area | What should be proven next | What good evidence looks like |
|---|---|---|
| Turn reliability | The robot consistently opens, commits, responds, and finishes voice turns without blocking. | Latency summaries across repeated interactions, including time to first audio response. |
| Attention and face handling | The robot listens when someone is plausibly addressing it and ignores nearby side conversations. | Face and attention evaluation across distances, lighting, and multi-person scenes. |
| Owner resolution | The harness correctly resolves the speaker or safely stays unknown. | A face-plus-voice evaluation set with correct, rejected, and ambiguous cases. |
| Memory safety | Person context and new memory stay attached to the resolved owner when memory is enabled. | Prompt/context snapshots and memory episode checks from repeated interactions. |
| Tool and motion safety | Model-requested actions stay within approved robot capabilities, including navigation only when the profile enables it. | Approved-action checks, navigation-result traces, battery/blocking behavior, and manual robot safety checks for live motion. |
| Operator experience | People can tell when the robot is listening, thinking, speaking, or asking for enrollment approval. | Display screenshots or recordings from representative interactions. |

The next stage should focus on maturity rather than breadth. The Argos-agent architecture already touches many of the pieces required for a believable robot companion. The highest-leverage work is to make the existing loop easier to measure, explain, and operate.

## 7. Key Risks

Perception can be confident and still wrong. Face recognition, head pose, and speaker matching depend on lighting, camera angle, audio quality, distance, and scene complexity. The mitigation is conservative ownership policy and evaluation on real collected interactions.

Memory mistakes can damage trust. A robot that remembers the wrong thing about the wrong person will feel unsafe. The mitigation is owner-scoped memory, explicit uncertainty, and checks that memory is only written when the speaker is resolved.

The robot body raises the stakes. A wrong chatbot answer is annoying; a wrong robot action can be unsafe or disruptive. The mitigation is a bounded action list, capability checks, and operator approval for live motion testing.

Demo polish can hide reliability gaps. A strong demo can still depend on favorable lighting, clean audio, or manual setup. The mitigation is to report measured results separately from demo behavior and make repeatability the next review standard.
