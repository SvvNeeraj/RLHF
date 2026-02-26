import re


def _clean_topic(text: str) -> str:
    t = text.strip().rstrip("?.!")
    t = re.sub(r"^(what is|define|explain|describe|how does|how do)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bhow\s+can\s+it\s+be\s+prevented\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bwith\s+example(s)?\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\band\s+explain\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" ,.-")
    return t or "the topic"


def _split_sections(template: str) -> dict[str, str]:
    headings = [
        "Short Summary:",
        "Detailed Explanation:",
        "Real-Life Example:",
        "Key Points:",
        "Conclusion:",
    ]
    out: dict[str, str] = {h: "" for h in headings}
    pattern = r"(Short Summary:|Detailed Explanation:|Real-Life Example:|Key Points:|Conclusion:)"
    parts = re.split(pattern, template)
    if len(parts) < 3:
        return out

    for i in range(1, len(parts), 2):
        head = parts[i]
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if head in out:
            out[head] = body
    return out


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {x.strip()}" for x in items if x and x.strip())


def _main_points_for_key(topic: str, key: str) -> list[str]:
    if key == "machine_learning":
        return [
            "Definition and objective of machine learning.",
            "End-to-end workflow: data prep, training, validation, and testing.",
            "Generalization behavior: overfitting, underfitting, and bias-variance balance.",
            "Deployment concerns: model drift, monitoring, and retraining.",
        ]
    if key in {"osi_tcpip", "computer_networks", "routing"}:
        return [
            f"Core role of {topic} in communication systems.",
            "Layer/protocol responsibilities and data flow path.",
            "Reliability, addressing/routing, and performance factors.",
            "Troubleshooting approach using protocol-level reasoning.",
        ]
    if key in {"sql_injection", "cyber_security"}:
        return [
            f"Security objective and threat model for {topic}.",
            "Attack surface and common failure points.",
            "Preventive controls and secure implementation practices.",
            "Detection, response, and hardening workflow.",
        ]
    if key in {"operating_system", "cpu_scheduling", "deadlock"}:
        return [
            f"Core concept and purpose of {topic} in OS behavior.",
            "How resources/processes are coordinated during execution.",
            "Performance-reliability tradeoffs and common bottlenecks.",
            "Practical strategy to avoid failure states and improve stability.",
        ]
    if key in {"dbms", "normalization"}:
        return [
            f"Data design objective of {topic}.",
            "Internal mechanism: dependencies, constraints, and integrity flow.",
            "Performance implications in reads/writes and joins.",
            "Production design balance between correctness and speed.",
        ]
    if key in {"ece_eee", "power_factor", "control_stability"}:
        return [
            f"Physical meaning and engineering objective of {topic}.",
            "Model behavior from input to output response.",
            "Key design constraints: stability, efficiency, and safety.",
            "Practical tuning/correction strategy in real systems.",
        ]
    return [
        f"Definition and purpose of {topic}.",
        f"Working flow of {topic} from input to output.",
        f"Where {topic} is applied in engineering systems.",
        f"Limitations and best-practice usage of {topic}.",
    ]


def _key_points_for_key(topic: str, key: str, needs_improvement: bool, variation_id: int) -> list[str]:
    if key == "machine_learning":
        base = [
            "Model quality depends more on data quality than model complexity alone.",
            "Separate training, validation, and test stages to avoid leakage.",
            "Use metrics aligned to task goals (for example, precision/recall for imbalance).",
            "Control overfitting with regularization, cross-validation, and monitoring.",
            "Re-evaluate model behavior after deployment to catch drift early.",
        ]
    elif key in {"osi_tcpip", "computer_networks", "routing"}:
        base = [
            "Layered design simplifies protocol debugging and system integration.",
            "Addressing and routing logic directly affects latency and reliability.",
            "Transport behavior (TCP/UDP) must match application requirements.",
            "Encapsulation/decapsulation explains packet movement across layers.",
            "Network performance tuning should balance throughput, delay, and loss.",
        ]
    elif key in {"sql_injection", "cyber_security"}:
        base = [
            "Treat all external input as untrusted by default.",
            "Use secure-by-default controls such as parameterization and least privilege.",
            "Combine prevention with monitoring and incident response.",
            "Patch and configuration hygiene reduce known exploit paths.",
            "Security awareness and process discipline are as important as tools.",
        ]
    elif key in {"operating_system", "cpu_scheduling", "deadlock"}:
        base = [
            "OS decisions are tradeoffs between fairness, latency, and throughput.",
            "Scheduling and synchronization policies shape practical system behavior.",
            "Resource contention must be analyzed before performance tuning.",
            "Deadlock/starvation risks should be handled by clear policy design.",
            "Explain mechanism first, then discuss measurable impact.",
        ]
    elif key in {"dbms", "normalization"}:
        base = [
            "Good schema design reduces anomalies and improves maintainability.",
            "Normalization should be balanced with query performance requirements.",
            "Indexes improve reads but add write/storage overhead.",
            "ACID and concurrency control are essential for reliable transactions.",
            "Design decisions must match real workload patterns.",
        ]
    elif key in {"ece_eee", "power_factor", "control_stability"}:
        base = [
            "Always connect equations to physical meaning.",
            "Input conditions determine response behavior and design margins.",
            "Stability and efficiency should be evaluated together.",
            "Protection limits and safe operation are non-negotiable in practice.",
            "Explain both theoretical model and practical implementation tradeoffs.",
        ]
    else:
        base = [
            f"Clear definition and scope of {topic}.",
            f"Working mechanism of {topic} from input, process, to output.",
            f"Practical engineering use of {topic}.",
            f"One realistic limitation of {topic}.",
            f"Interpret {topic} using measurable behavior rather than vague statements.",
        ]

    if needs_improvement:
        variants = [
            f"Add one additional concrete practical scenario for {topic}.",
            f"Connect assumptions to observed outcomes while explaining {topic}.",
            f"Include one common pitfall of {topic} and its correction.",
        ]
        base[-1] = variants[variation_id]

    if topic.lower() not in " ".join(base).lower():
        base[0] = f"Explain {topic} with clear definition, mechanism, and outcome."

    return base


def _upgrade_template(template: str, topic: str, key: str, flags: dict | None = None) -> str:
    flags = flags or {}
    needs_improvement = bool(flags.get("needs_improvement", False))
    variation_id = int(flags.get("variation_id", 0)) % 3
    sections = _split_sections(template)
    if not sections["Short Summary:"]:
        return template.strip()

    summary = sections["Short Summary:"].strip()
    if needs_improvement:
        summary_variants = [
            f"{topic.capitalize()} explains a core engineering principle used to model behavior and solve practical problems.",
            f"{topic.capitalize()} defines how a system behaves under specific inputs and constraints.",
            f"{topic.capitalize()} links concept, mechanism, and practical system outcome in a structured way.",
        ]
        summary = summary_variants[variation_id]

    detailed = sections["Detailed Explanation:"]
    detail_addon = (
        f"\n\n{topic.capitalize()} is best understood by linking input conditions, internal mechanism, and output behavior. "
        f"A complete discussion also includes one practical limitation or tradeoff."
    )
    if "input conditions" not in detailed.lower():
        detailed = (detailed + detail_addon).strip()
    if needs_improvement:
        detail_variants = [
            f" Include one focused scenario showing how {topic} behaves when operating conditions change.",
            f" Include assumptions used in {topic} and the effect when those assumptions are violated.",
            f" Include one practical improvement approach used when {topic} shows limitations.",
        ]
        detailed = (detailed + detail_variants[variation_id]).strip()
    example_body = sections["Real-Life Example:"]
    cleaned_example = re.sub(r"\s+", " ", example_body).strip()
    if not cleaned_example:
        cleaned_example = f"A practical use of {topic} in a real engineering workflow."
    ex1_variants = [
        cleaned_example,
        f"In a classroom lab, students apply {topic} step by step and validate each stage with measured output.",
        f"In exam-based problem solving, {topic} is used to map the given data to the correct method and final result.",
    ]
    ex2_variants = [
        f"In a lab experiment, {topic} helps students measure performance systematically and reduce analysis mistakes.",
        f"In a mini project, {topic} helps split a complex task into clear steps and produce more reliable output.",
        f"In semester practicals, applying {topic} improves debugging speed and result accuracy.",
    ]
    ex3_variants = [
        f"In industry systems, teams apply {topic} to reduce failures and improve performance under real load.",
        f"In production environments, {topic} is used to improve stability, consistency, and service reliability.",
        f"In enterprise workflows, {topic} supports scalable operation and better fault handling.",
    ]
    examples = (
        f"- Example 1: {ex1_variants[variation_id]}\n"
        f"- Example 2: {ex2_variants[variation_id]}\n"
        f"- Example 3: {ex3_variants[variation_id]}"
    )

    key_points = _format_bullets(_key_points_for_key(topic, key, needs_improvement, variation_id))

    conclusion = sections["Conclusion:"]
    conclusion_addon = (
        f" In short, {topic} is important for both exam performance and practical problem-solving. "
        "A clear understanding helps you explain the concept confidently, choose the right method, and avoid common mistakes."
    )
    if "In short" not in conclusion:
        conclusion = (conclusion + conclusion_addon).strip()
    if needs_improvement:
        conc_variants = [
            f" Writing {topic} as concept-to-application flow makes answers clearer and easier to remember.",
            f" A strong explanation of {topic} includes both mechanism and limitation together for balanced understanding.",
            f" Pairing theory with focused examples improves revision speed and answer quality for {topic}.",
        ]
        conclusion = (conclusion + conc_variants[variation_id]).strip()

    return (
        "Short Summary:\n"
        f"{summary}\n\n"
        "Detailed Explanation:\n"
        f"{detailed}\n\n"
        "Real-Life Example:\n"
        f"{examples}\n\n"
        "Key Points:\n"
        f"{key_points}\n\n"
        "Conclusion:\n"
        f"{conclusion}"
    ).strip()


def detect_topic_key(query: str, topic: str) -> str:
    text = f"{query} {topic}".lower()

    # Cyber / DB / OS focused
    if "backpropagation" in text or "vanishing gradient" in text or "gradient flow" in text:
        return "backpropagation"
    if "data analytics" in text:
        return "data_analytics"
    if "mysql" in text:
        return "mysql"
    if "sql injection" in text:
        return "sql_injection"
    if "deadlock" in text:
        return "deadlock"
    if "cpu scheduling" in text or "round robin" in text or "fcfs" in text or "sjf" in text:
        return "cpu_scheduling"
    if "normalization" in text or "1nf" in text or "2nf" in text or "3nf" in text or "bcnf" in text:
        return "normalization"
    if "stress-strain" in text or "stress strain" in text or "young's modulus" in text or "hooke's law" in text:
        return "stress_strain"

    # Networks
    if "routing" in text or "ospf" in text or "rip" in text or "distance vector" in text:
        return "routing"
    if "osi" in text or "tcp/ip" in text or "transport layer" in text:
        return "osi_tcpip"

    # AIML
    if "transformer" in text or "attention mechanism" in text or "self-attention" in text:
        return "transformers"
    if "overfitting" in text or "underfitting" in text or "bias variance" in text:
        return "overfitting"

    # EEE/ECE
    if "power factor" in text:
        return "power_factor"
    if "stability" in text and "control" in text:
        return "control_stability"

    # General domain templates
    if "operating system" in text or re.search(r"\bos\b", text):
        return "operating_system"
    if "dbms" in text or "database" in text or "mysql" in text or "sql server" in text or "postgres" in text:
        return "dbms"
    if "computer network" in text or "network protocol" in text or re.search(r"\bcn\b", text):
        return "computer_networks"
    if "machine learning" in text or "ml" in text or "deep learning" in text or "neural" in text:
        return "machine_learning"
    if "cyber security" in text or "cybersecurity" in text or "cryptography" in text or "malware" in text:
        return "cyber_security"
    if "signal" in text or "power system" in text or "control system" in text or "digital electronics" in text or "microprocessor" in text:
        return "ece_eee"

    return "generic"


def _sql_injection_template() -> str:
    return """
Short Summary:
SQL injection is a security attack where malicious input changes the intended SQL query. If input is not handled safely, attackers can read, modify, or delete sensitive database data.

Detailed Explanation:
SQL injection occurs when user input is directly concatenated into SQL statements. In that case, attacker-supplied text can alter query logic and bypass checks such as authentication.

The main prevention rule is to separate SQL code from user data. Use parameterized queries (prepared statements) so user input is treated only as data, not executable SQL. Add server-side validation, least-privilege database roles, and safe error messages that do not leak schema details.

For stronger protection, combine secure coding with logging, query monitoring, and regular security testing. This layered defense reduces both successful attacks and detection delay.

Real-Life Example:
A login query built by string concatenation can be bypassed with input like `' OR '1'='1`. With prepared statements, the same input is treated as plain text and the bypass fails.

Key Points:
- Never concatenate raw user input into SQL strings.
- Use prepared statements for all dynamic query values.
- Validate inputs on server side.
- Restrict database privileges.
- Avoid exposing raw SQL errors.

Conclusion:
SQL injection is preventable when query design is secure by default. Parameterized queries are the most important first step.
""".strip()


def _mysql_template() -> str:
    return """
Short Summary:
MySQL is an open-source relational database management system (RDBMS) that stores and manages structured data using SQL.

Detailed Explanation:
MySQL organizes data into related tables with rows and columns, allowing efficient storage, retrieval, and updates through SQL queries. It supports constraints, joins, indexing, and transactions to maintain data consistency and query performance.

In practical applications, MySQL is used as the backend database for web apps, ERP systems, and analytics workflows. Core strengths include reliability, scalability for moderate-to-large workloads, and strong ecosystem support.

Real-Life Example:
A college portal stores student, course, attendance, and marks data in MySQL. SQL queries are used to generate reports and dashboards.

Key Points:
- MySQL is an RDBMS that uses SQL.
- Data is stored in normalized relational tables.
- Indexes and query optimization improve performance.
- Transactions and constraints protect data integrity.

Conclusion:
MySQL is widely used for structured data workloads and remains a practical database choice for academic and industry projects.
""".strip()


def _backpropagation_template() -> str:
    return """
Short Summary:
Backpropagation is the training algorithm used in neural networks to update weights by propagating prediction error from output layer to earlier layers.

Detailed Explanation:
During forward propagation, the network computes predictions from input through hidden layers to output. A loss function (such as cross-entropy or MSE) measures prediction error. Backpropagation then applies chain rule to compute gradients of loss with respect to each weight.

In gradient flow, these gradients move from output to input-side layers. Parameters are updated using an optimizer (for example SGD or Adam) so loss decreases iteratively. Vanishing gradients occur when gradient magnitudes become very small in deep networks, making early layers learn very slowly.

Common solutions are ReLU-family activations, proper weight initialization (He/Xavier), batch normalization, residual/skip connections, and gated architectures (LSTM/GRU for sequence tasks). Gradient clipping is also used to stabilize training.

Real-Life Example:
In image classification, backpropagation updates convolutional and dense layer weights after each batch so the model gradually improves accuracy.

Key Points:
- Backpropagation computes weight gradients using chain rule.
- Gradient flow quality controls learning speed in deep layers.
- Vanishing gradient slows or stalls early-layer learning.
- Architectural and optimization choices reduce vanishing effects.

Conclusion:
Backpropagation is central to deep learning because it turns prediction error into parameter updates that improve model performance.
""".strip()


def _data_analytics_template() -> str:
    return """
Short Summary:
Data analytics is the process of examining data to discover useful patterns, insights, and trends for better decision making.

Detailed Explanation:
Data analytics typically includes data collection, cleaning, transformation, exploration, modeling, and visualization. The goal is to convert raw data into actionable information. Depending on objective, analytics may be descriptive (what happened), diagnostic (why it happened), predictive (what may happen), or prescriptive (what should be done).

In engineering and business systems, data analytics is used for quality monitoring, anomaly detection, demand forecasting, and performance optimization. Tools such as SQL, Python, dashboards, and statistical methods are commonly used in end-to-end workflows.

Reliable analytics requires good data quality, clear metrics, and domain context. Without these, insights can be misleading even when calculations are correct.

Real-Life Example:
A college uses data analytics on attendance and marks to identify at-risk students early and provide targeted academic support.

Key Points:
- Data analytics transforms raw data into decision-ready insights.
- Workflow includes cleaning, analysis, and visualization.
- Different analytics types answer different decision questions.
- Data quality and context are critical for trustworthy results.

Conclusion:
Data analytics improves decision quality by turning large data volumes into clear, actionable understanding.
""".strip()


def _stress_strain_template() -> str:
    return """
Short Summary:
Stress-strain behavior explains how a material deforms under load and is fundamental for selecting safe materials in engineering design.

Detailed Explanation:
Stress is internal force per unit area, while strain is deformation per unit original length. In the elastic region, stress and strain are proportional (Hooke's law), and the slope of the curve gives Young's modulus.

As loading increases, materials reach yield point, plastic deformation region, ultimate tensile strength, and finally fracture. These points define strength, ductility, and safe operating limits for components.

In design practice, stress-strain data is used to choose materials, determine factor of safety, and prevent structural failure under expected service loads.

Real-Life Example:
In a tensile test, a steel specimen is stretched and its stress-strain curve is recorded to identify yield strength and ultimate strength before using it in beams or shafts.

Key Points:
- Stress and strain quantify load response of materials.
- Elastic region is recoverable; plastic region is permanent.
- Young's modulus indicates stiffness.
- Yield and ultimate strength guide safe design limits.

Conclusion:
Understanding stress-strain behavior is essential for reliable mechanical and civil design, especially in safety-critical applications.
""".strip()


def _deadlock_template() -> str:
    return """
Short Summary:
Deadlock is a state where processes wait forever because each process holds a resource needed by another. As a result, no process can continue execution.

Detailed Explanation:
Deadlock appears when four conditions exist together: mutual exclusion, hold-and-wait, no preemption, and circular wait. If all are true at the same time, system progress can stop completely.

Operating systems handle deadlock using prevention, avoidance, detection, or recovery. Prevention breaks at least one deadlock condition. Avoidance checks resource allocation safety before granting resources. Detection allows deadlock to occur and later identifies cycles in resource-allocation graphs.

Recovery methods include terminating one or more processes, rolling back process state, or forcefully reclaiming resources. Each strategy has tradeoffs in performance and reliability.

Real-Life Example:
Process A holds printer and waits for scanner, while Process B holds scanner and waits for printer. Both wait forever, so neither completes.

Key Points:
- Deadlock means permanent waiting among processes.
- Four Coffman conditions are required.
- Prevention and avoidance act before deadlock.
- Detection and recovery act after deadlock.
- Resource allocation policy strongly affects deadlock risk.

Conclusion:
Deadlock is a core OS reliability issue. Understanding conditions and handling strategies is essential for exam answers and system design.
""".strip()


def _cpu_scheduling_template() -> str:
    return """
Short Summary:
CPU scheduling decides which process gets CPU time next. Good scheduling improves throughput, response time, and fairness.

Detailed Explanation:
The scheduler selects processes from the ready queue using algorithms such as FCFS, SJF, Priority, and Round Robin. Each algorithm optimizes different metrics.

FCFS is simple but may cause convoy effect. SJF minimizes average waiting time when burst estimates are accurate. Priority scheduling can cause starvation if lower-priority processes never get CPU. Round Robin improves responsiveness in time-sharing systems using fixed time quantum.

Important evaluation metrics are waiting time, turnaround time, response time, throughput, and CPU utilization. In practice, modern systems use hybrid policies to balance latency and fairness.

Real-Life Example:
In an interactive lab system, Round Robin gives each student process a short CPU slice, so terminal response feels smooth for all users.

Key Points:
- Scheduling controls CPU allocation order.
- Different algorithms optimize different goals.
- Time quantum selection affects Round Robin performance.
- Starvation and fairness must be handled.
- Metrics should guide algorithm choice.

Conclusion:
CPU scheduling is central to OS performance. Choosing the right algorithm depends on workload type and responsiveness requirements.
""".strip()


def _normalization_template() -> str:
    return """
Short Summary:
Normalization organizes database tables to reduce redundancy and prevent update anomalies. It improves data integrity and schema clarity.

Detailed Explanation:
Normalization decomposes relations based on functional dependencies. In 1NF, attributes are atomic. 2NF removes partial dependency on composite keys. 3NF removes transitive dependencies. BCNF further strengthens dependency constraints.

The goal is to avoid insertion, update, and deletion anomalies while keeping relations logically consistent. However, heavy decomposition can increase join cost in read-heavy workloads.

In practical DB design, normalization is balanced with performance. Systems often normalize for correctness first, then selectively denormalize specific paths for speed.

Real-Life Example:
If student and department details are stored in one table, repeated department info causes redundancy. Splitting into Student and Department tables removes repetition and maintains consistency.

Key Points:
- Normalization reduces redundancy.
- 1NF, 2NF, 3NF, BCNF address dependency issues.
- Improves consistency and anomaly resistance.
- Excessive normalization may increase join overhead.
- Use workload-aware balance in production.

Conclusion:
Normalization is a foundational DBMS concept for clean schema design and reliable data management.
""".strip()


def _routing_template() -> str:
    return """
Short Summary:
Routing determines the path packets follow from source to destination across networks. Efficient routing improves reliability and end-to-end performance.

Detailed Explanation:
Routers use routing tables built by static rules or dynamic protocols. Dynamic routing protocols exchange network state and update paths automatically when topology changes.

Distance-vector protocols use neighbor-based hop information, while link-state protocols build a broader network view and compute shortest paths. Metrics such as hop count, cost, delay, and bandwidth influence route selection.

Convergence speed, loop prevention, and scalability are key design concerns. Practical routing design balances fast recovery with control-plane overhead.

Real-Life Example:
When one link in a campus network fails, dynamic routing recalculates alternate paths so traffic continues with minimal interruption.

Key Points:
- Routing decides packet forwarding path.
- Static and dynamic routing are major approaches.
- Link-state and distance-vector differ in topology knowledge.
- Metrics influence path quality.
- Fast convergence improves network availability.

Conclusion:
Routing is core to network resilience and performance. Clear understanding of protocol behavior is essential for exams and troubleshooting.
""".strip()


def _osi_tcpip_template() -> str:
    return """
Short Summary:
OSI and TCP/IP models explain networking through layers, where each layer has a specific responsibility. Layered design simplifies protocol development and troubleshooting.

Detailed Explanation:
The OSI model has seven layers, while TCP/IP is commonly represented with four or five layers. At a high level, application handles user services, transport manages end-to-end delivery, network handles addressing/routing, and link/physical handle frame transmission.

Encapsulation is the key concept: each layer adds its header while data moves downward, then headers are removed at receiver side in reverse order. This modular separation allows interoperability and easier fault isolation.

In real systems, TCP/IP is the practical standard, but OSI remains valuable for conceptual understanding and exam explanations.

Real-Life Example:
During web access, HTTP works at application layer, TCP provides reliable transport, IP handles routing, and Ethernet/Wi-Fi transmits frames.

Key Points:
- Layering separates protocol responsibilities.
- TCP/IP is practical implementation model.
- Encapsulation/decapsulation explains data flow.
- Troubleshooting uses layer-wise diagnosis.
- Interoperability depends on standard protocols.

Conclusion:
OSI/TCP-IP layering is fundamental for understanding protocol behavior and solving real network issues systematically.
""".strip()


def _transformers_template() -> str:
    return """
Short Summary:
Transformers are deep learning models that use attention to capture relationships between tokens in parallel. They are widely used in NLP, vision, and multimodal AI.

Detailed Explanation:
The core idea is self-attention, where each token attends to other relevant tokens to build context-aware representations. Unlike RNNs, transformers process tokens in parallel, improving training efficiency on large datasets.

Main building blocks include token embeddings, positional encoding, multi-head attention, feed-forward layers, residual connections, and layer normalization. Encoder-decoder variants are used for tasks like translation, while decoder-only variants are common in modern language models.

Transformer performance scales with model size, data quality, and compute. Challenges include memory cost for long sequences and bias/hallucination control in generation tasks.

Real-Life Example:
In machine translation, a transformer maps a source sentence to a target sentence by learning token-level dependencies and context importance through attention.

Key Points:
- Self-attention is the central mechanism.
- Parallel token processing improves efficiency.
- Multi-head attention captures diverse relationships.
- Positional encoding preserves order information.
- Widely used in modern foundation models.

Conclusion:
Transformers changed modern AI by combining scalability and strong context modeling. Understanding attention is key for exams and applied ML work.
""".strip()


def _overfitting_template() -> str:
    return """
Short Summary:
Overfitting happens when a model learns training noise instead of general patterns. It performs well on training data but poorly on unseen data.

Detailed Explanation:
Overfitting occurs when model complexity is too high relative to data quality or quantity. The model memorizes specific examples and loses generalization ability. Underfitting is the opposite case where model is too simple to capture patterns.

Common controls include regularization, dropout, early stopping, cross-validation, data augmentation, and feature selection. Proper train-validation-test split is essential to estimate real-world performance correctly.

Bias-variance tradeoff explains this behavior: lowering bias often increases variance. Good model design finds balance for stable generalization.

Real-Life Example:
A classifier shows 99% training accuracy but 70% test accuracy. Adding regularization and reducing feature noise improves test performance.

Key Points:
- Overfitting harms unseen-data accuracy.
- High complexity and noisy data increase risk.
- Validation metrics are necessary for model selection.
- Regularization and early stopping are common fixes.
- Balance bias and variance for generalization.

Conclusion:
Preventing overfitting is central to reliable ML systems. Generalization-focused evaluation is more important than training accuracy alone.
""".strip()


def _power_factor_template() -> str:
    return """
Short Summary:
Power factor shows how effectively electrical power is converted into useful work. A low power factor means more current is needed for the same real power.

Detailed Explanation:
Power factor is the ratio of real power (kW) to apparent power (kVA). In AC systems with inductive loads, current lags voltage, reducing power factor.

Low power factor increases line current, causes higher I2R losses, larger voltage drops, and reduced system efficiency. Utilities may impose penalties for poor power factor in industrial setups.

Correction methods include capacitor banks, synchronous condensers, and active power factor correction circuits. Improving power factor reduces losses and improves voltage regulation.

Real-Life Example:
An industrial motor load draws lagging current. Installing capacitor banks near the load improves power factor and lowers electricity penalty charges.

Key Points:
- PF = real power / apparent power.
- Inductive loads often cause lagging PF.
- Low PF increases losses and current demand.
- Correction improves efficiency and voltage profile.
- PF management reduces operating cost.

Conclusion:
Power factor is a practical efficiency metric in AC systems. Improving it benefits both technical performance and energy economics.
""".strip()


def _control_stability_template() -> str:
    return """
Short Summary:
Control system stability means the output remains bounded and settles appropriately after disturbances. Stable systems are predictable, safe, and usable in real operation.

Detailed Explanation:
A control system is stable when its response does not diverge over time. In linear systems, stability is linked to pole locations of the closed-loop transfer function. Poles in the left-half s-plane indicate asymptotic stability.

Tools such as Routh-Hurwitz, root locus, Bode plot, and Nyquist criterion are used to analyze stability margins. Gain margin and phase margin quantify robustness against model uncertainty and disturbances.

In practice, controller tuning must balance stability, speed, overshoot, and steady-state error. Excessive gain can improve speed but may destabilize the system.

Real-Life Example:
In cruise control, if controller gains are too high, vehicle speed oscillates around setpoint. Proper tuning yields smooth settling without persistent oscillation.

Key Points:
- Stability ensures bounded, convergent response.
- Pole location is central in linear stability analysis.
- Margins indicate robustness.
- Tuning requires tradeoff among performance metrics.
- Disturbance rejection must not break stability.

Conclusion:
Stability is the first requirement in control design. Performance improvements are meaningful only after stable operation is guaranteed.
""".strip()


def _operating_system_template() -> str:
    return """
Short Summary:
An operating system is system software that manages hardware resources and provides services to applications. It acts as the core layer between user programs and physical hardware.

Detailed Explanation:
The operating system controls process execution, memory usage, file management, and device communication. It decides which process gets CPU time, how memory is allocated, and how data is read and written safely.

Major functions include process scheduling, synchronization, deadlock handling, virtual memory management, and I/O management. These functions ensure multiple programs can run efficiently without corrupting system state.

In practice, OS design focuses on performance, fairness, security, and reliability. A well-designed OS improves throughput and response time while protecting process isolation and system integrity.

Real-Life Example:
When you run a browser, IDE, and media player together, the OS schedules CPU slices, isolates memory, and coordinates disk/network I/O so all applications remain responsive.

Key Points:
- OS is the resource manager of the computer.
- It handles CPU, memory, file, and device operations.
- Scheduling and memory policies impact performance.
- Security and isolation prevent interference between programs.
- Concurrency control is critical in multitasking systems.

Conclusion:
Operating systems are foundational for reliable and efficient computing. Understanding their core functions is essential for both exams and real software engineering.
""".strip()


def _dbms_template() -> str:
    return """
Short Summary:
A DBMS is software used to store, manage, and retrieve structured data efficiently. It helps maintain consistency, security, and scalability of data operations.

Detailed Explanation:
DBMS organizes data into tables and provides query capabilities through SQL. It supports transactions, concurrency control, and recovery mechanisms so data remains correct even during failures.

Important DBMS concepts include normalization, indexing, ACID properties, and schema design. Normalization reduces redundancy, indexing improves query speed, and ACID ensures reliable transactions.

In real systems, DBMS performance depends on query optimization, indexing strategy, and transaction workload. Proper design choices directly affect latency and storage efficiency.

Real-Life Example:
In an online shopping app, DBMS stores users, products, carts, and orders. During checkout, transaction control ensures payment and order creation are committed together or rolled back safely.

Key Points:
- DBMS manages structured data with SQL.
- ACID ensures transaction reliability.
- Normalization improves data integrity.
- Indexes improve query performance.
- Concurrency control prevents conflicting updates.

Conclusion:
DBMS is central to modern applications because it ensures reliable and efficient data management under real-world load.
""".strip()


def _computer_networks_template() -> str:
    return """
Short Summary:
Computer networks connect devices to exchange data using standard communication protocols. Networking enables internet access, distributed systems, and cloud services.

Detailed Explanation:
Networking follows layered architecture where each layer handles a specific role, such as addressing, routing, transport reliability, and application messaging. Core concepts include IP addressing, subnetting, routing, TCP/UDP behavior, and error control.

Protocols define how data is formatted, transmitted, acknowledged, and retransmitted on failure. Reliable communication depends on congestion control, flow control, and fault recovery techniques.

In engineering practice, network design balances throughput, latency, scalability, and security. Understanding protocol behavior is essential for troubleshooting and performance tuning.

Real-Life Example:
When opening a website, DNS resolves the domain, TCP establishes a connection, TLS secures the session, and HTTP transfers page content to your browser.

Key Points:
- Networks use layered protocol architecture.
- IP and routing decide packet path.
- TCP provides reliable ordered delivery.
- Congestion and flow control affect speed.
- Security layers protect data in transit.

Conclusion:
Computer networks are the backbone of modern digital systems. Strong protocol-level understanding improves both exam answers and practical debugging skills.
""".strip()


def _machine_learning_template() -> str:
    return """
Short Summary:
Machine learning enables systems to learn patterns from data and make predictions without explicitly coded rules. It is widely used for classification, regression, and intelligent decision support.

Detailed Explanation:
ML workflow typically includes data collection, preprocessing, feature selection, model training, validation, and testing. The model learns relationships from training data and generalizes to unseen inputs.

Core concepts include overfitting, underfitting, bias-variance tradeoff, and evaluation metrics such as accuracy, precision, recall, and F1-score. Proper model selection and tuning are needed to balance performance and generalization.

In real engineering systems, ML quality depends on data quality, feature relevance, and continuous monitoring after deployment. Model drift and data imbalance are common practical challenges.

Real-Life Example:
An email spam filter learns from labeled emails and predicts whether a new email is spam based on learned textual patterns.

Key Points:
- ML learns patterns from data.
- Training and validation are separate stages.
- Overfitting harms real-world generalization.
- Metrics must match problem objective.
- Data quality strongly controls model quality.

Conclusion:
Machine learning is powerful when used with sound data practices and proper evaluation. Conceptual clarity is critical for both exams and practical model building.
""".strip()


def _cyber_security_template() -> str:
    return """
Short Summary:
Cyber security protects systems, networks, and data from unauthorized access and attacks. It combines preventive, detective, and corrective controls.

Detailed Explanation:
Core security goals are confidentiality, integrity, and availability. Threats include malware, phishing, injection attacks, credential theft, and denial-of-service incidents.

Effective protection uses layered security: secure coding, strong authentication, encryption, access control, patch management, and monitoring. Incident response planning is essential to detect and contain attacks quickly.

In practice, cyber security is continuous, not one-time. Risk assessment, logging, vulnerability scanning, and user awareness training are required for sustained defense.

Real-Life Example:
A company secures its portal with MFA, HTTPS, role-based access, and SIEM monitoring. When suspicious login behavior appears, automated alerts trigger account lock and investigation.

Key Points:
- Security is based on confidentiality, integrity, availability.
- Defense-in-depth reduces attack success.
- Monitoring and incident response are mandatory.
- Human factors are a major risk source.
- Regular patching closes known vulnerabilities.

Conclusion:
Cyber security is an engineering discipline that requires layered controls, continuous monitoring, and rapid response to evolving threats.
""".strip()


def _ece_eee_template() -> str:
    return """
Short Summary:
ECE and EEE core topics focus on signal behavior, circuit operation, control, and power systems. These concepts explain how electrical and electronic systems are designed and optimized.

Detailed Explanation:
In ECE/EEE, analysis starts with fundamental laws and system models, then moves to response behavior under real operating conditions. Key areas include circuit theory, digital electronics, signals and systems, control systems, and power conversion/distribution.

Understanding these topics requires linking equations with physical meaning. Students should explain what each parameter represents, how system response changes with input conditions, and how stability or efficiency is evaluated.

In practical engineering, component selection, operating limits, and protection mechanisms are critical. Real systems are optimized for reliability, power quality, and performance under varying load conditions.

Real-Life Example:
A motor control setup uses sensing, controller logic, and power electronics to maintain speed under changing load. Proper tuning improves stability and reduces energy loss.

Key Points:
- Start from physical meaning, not formula memorization.
- Explain system response under changing inputs.
- Stability and efficiency are key design goals.
- Protection and limits are essential in real systems.
- Connect theory with measured practical behavior.

Conclusion:
ECE/EEE concepts become easier when studied as model-to-behavior flow. This approach improves both exam clarity and practical engineering reasoning.
""".strip()


def _generic_template(topic: str) -> str:
    return f"""
Short Summary:
{topic.capitalize()} is an important technical concept used to understand system behavior and solve engineering problems.

Detailed Explanation:
{topic.capitalize()} explains how a system is modeled, how inputs are processed, and how final outputs are interpreted in real conditions. A strong explanation includes definition, core mechanism, and expected behavior under normal operating assumptions.

In engineering practice, {topic} is evaluated using measurable parameters, performance constraints, and reliability goals. Understanding these links makes the topic easier to apply in analysis, design, and troubleshooting.

Real-Life Example:
In a practical system, {topic} is used to improve correctness, reliability, or efficiency under known constraints.

Key Points:
- Core definition and scope of {topic}.
- Mechanism from input to output.
- Practical engineering application.
- One realistic limitation.

Conclusion:
{topic.capitalize()} is best understood when concept, mechanism, and application are connected clearly. This improves both learning quality and practical use.
""".strip()


def generate_template(query: str, flags: dict | None = None) -> str:
    topic = _clean_topic(query)
    key = detect_topic_key(query, topic)
    if key == "backpropagation":
        base = _backpropagation_template()
    elif key == "data_analytics":
        base = _data_analytics_template()
    elif key == "sql_injection":
        base = _sql_injection_template()
    elif key == "mysql":
        base = _mysql_template()
    elif key == "deadlock":
        base = _deadlock_template()
    elif key == "cpu_scheduling":
        base = _cpu_scheduling_template()
    elif key == "normalization":
        base = _normalization_template()
    elif key == "stress_strain":
        base = _stress_strain_template()
    elif key == "routing":
        base = _routing_template()
    elif key == "osi_tcpip":
        base = _osi_tcpip_template()
    elif key == "transformers":
        base = _transformers_template()
    elif key == "overfitting":
        base = _overfitting_template()
    elif key == "power_factor":
        base = _power_factor_template()
    elif key == "control_stability":
        base = _control_stability_template()
    elif key == "operating_system":
        base = _operating_system_template()
    elif key == "dbms":
        base = _dbms_template()
    elif key == "computer_networks":
        base = _computer_networks_template()
    elif key == "machine_learning":
        base = _machine_learning_template()
    elif key == "cyber_security":
        base = _cyber_security_template()
    elif key == "ece_eee":
        base = _ece_eee_template()
    else:
        base = _generic_template(topic)

    return _upgrade_template(base, topic, key, flags)
