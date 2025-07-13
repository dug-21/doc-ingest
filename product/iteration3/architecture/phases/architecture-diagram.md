# NeuralDocFlow Architecture Diagram

## System Overview

```mermaid
graph TB
    subgraph "Phase 1: Core Foundation"
        PDF[PDF Input] --> Parser[PDF Parser Engine]
        Parser --> TextExt[Text Extractor]
        Parser --> LayoutAnal[Layout Analyzer]
        Parser --> MetaExt[Metadata Extractor]
        TextExt --> CoreOut[ProcessedDocument]
        LayoutAnal --> CoreOut
        MetaExt --> CoreOut
    end
    
    subgraph "Phase 2: Swarm Coordination"
        CoreOut --> Coord[Swarm Coordinator]
        Coord --> Agent1[Parser Agent]
        Coord --> Agent2[Extractor Agent]
        Coord --> Agent3[Analyzer Agent]
        Coord --> MsgBus[Message Bus]
        Agent1 --> MsgBus
        Agent2 --> MsgBus
        Agent3 --> MsgBus
        MsgBus --> SwarmOut[Distributed Results]
    end
    
    subgraph "Phase 3: Neural Engine"
        SwarmOut --> Neural[RUV-FANN Engine]
        Neural --> Train[Training Pipeline]
        Neural --> Infer[Inference Engine]
        Neural --> Pattern[Pattern Recognition]
        Train --> NeuralOut[Neural Features]
        Infer --> NeuralOut
        Pattern --> NeuralOut
    end
    
    subgraph "Phase 4: Document Intelligence"
        NeuralOut --> Trans[Transformer Models]
        Trans --> Embed[Embeddings]
        Trans --> NER[Entity Recognition]
        Trans --> Semantic[Semantic Analysis]
        Embed --> IntelOut[Document Intelligence]
        NER --> IntelOut
        Semantic --> IntelOut
    end
    
    subgraph "Phase 5: SEC Specialization"
        IntelOut --> SECClass[Filing Classifier]
        IntelOut --> FinTable[Financial Table Extractor]
        IntelOut --> XBRL[XBRL Mapper]
        SECClass --> SECOut[SEC Filing Data]
        FinTable --> SECOut
        XBRL --> SECOut
    end
    
    subgraph "Phase 6: API & Integration"
        SECOut --> PyBind[Python Bindings]
        SECOut --> REST[REST API]
        SECOut --> GraphQL[GraphQL API]
        SECOut --> Stream[Streaming API]
        PyBind --> APIOut[Client SDKs]
        REST --> APIOut
        GraphQL --> APIOut
        Stream --> APIOut
    end
    
    subgraph "Phase 7: Production Excellence"
        APIOut --> Monitor[Monitoring]
        APIOut --> Trace[Distributed Tracing]
        APIOut --> Scale[Auto-scaling]
        Monitor --> ProdOut[Production System]
        Trace --> ProdOut
        Scale --> ProdOut
    end
    
    style PDF fill:#f9f,stroke:#333,stroke-width:4px
    style ProdOut fill:#9f9,stroke:#333,stroke-width:4px
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Input
        PDF[PDF Document]
        Config[Configuration]
    end
    
    subgraph "Processing Pipeline"
        P1[Phase 1<br/>Core PDF] --> P2[Phase 2<br/>Swarm]
        P2 --> P3[Phase 3<br/>Neural]
        P3 --> P4[Phase 4<br/>Intelligence]
        P4 --> P5[Phase 5<br/>SEC]
        P5 --> P6[Phase 6<br/>API]
        P6 --> P7[Phase 7<br/>Production]
    end
    
    subgraph Output
        JSON[JSON API]
        XBRL[XBRL Format]
        SDK[Python SDK]
        Stream[WebSocket]
    end
    
    PDF --> P1
    Config --> P1
    P7 --> JSON
    P7 --> XBRL
    P7 --> SDK
    P7 --> Stream
```

## Component Architecture

```mermaid
graph TD
    subgraph "Rust Core"
        Core[neuralflow-core<br/>Phase 1]
        Swarm[neuralflow-swarm<br/>Phase 2]
        Neural[neuralflow-neural<br/>Phase 3]
        Intel[neuralflow-intelligence<br/>Phase 4]
        SEC[neuralflow-sec<br/>Phase 5]
    end
    
    subgraph "API Layer"
        API[neuralflow-api<br/>Phase 6]
        Python[Python Bindings]
        REST[REST Server]
        GraphQL[GraphQL Server]
    end
    
    subgraph "Production"
        Monitor[neuralflow-monitor<br/>Phase 7]
        Deploy[Deployment]
        Scale[Scaling]
    end
    
    Core --> Swarm
    Core --> Neural
    Swarm --> Intel
    Neural --> Intel
    Intel --> SEC
    SEC --> API
    API --> Python
    API --> REST
    API --> GraphQL
    API --> Monitor
    Monitor --> Deploy
    Monitor --> Scale
```

## Swarm Topology (Phase 2)

```mermaid
graph TD
    Coord[Coordinator] --> PA1[Parser Agent 1]
    Coord --> PA2[Parser Agent 2]
    Coord --> PA3[Parser Agent 3]
    Coord --> EA1[Extractor Agent 1]
    Coord --> EA2[Extractor Agent 2]
    Coord --> AA1[Analyzer Agent 1]
    Coord --> AA2[Analyzer Agent 2]
    Coord --> Agg[Aggregator Agent]
    
    PA1 -.-> MB[Message Bus]
    PA2 -.-> MB
    PA3 -.-> MB
    EA1 -.-> MB
    EA2 -.-> MB
    AA1 -.-> MB
    AA2 -.-> MB
    Agg -.-> MB
    
    MB --> State[Distributed State]
    
    style Coord fill:#f96,stroke:#333,stroke-width:2px
    style Agg fill:#69f,stroke:#333,stroke-width:2px
```

## Neural Architecture (Phase 3)

```mermaid
graph LR
    Input[Document Features] --> FE[Feature Extraction]
    FE --> NN1[Neural Network 1<br/>Classification]
    FE --> NN2[Neural Network 2<br/>Pattern Recognition]
    FE --> NN3[Neural Network 3<br/>Anomaly Detection]
    
    NN1 --> Ensemble[Ensemble Layer]
    NN2 --> Ensemble
    NN3 --> Ensemble
    
    Ensemble --> Output[Predictions]
    
    Train[Training Data] --> TM[Training Manager]
    TM --> NN1
    TM --> NN2
    TM --> NN3
```

## SEC Processing Pipeline (Phase 5)

```mermaid
flowchart TD
    Doc[Processed Document] --> Classify[Filing Classifier]
    
    Classify -->|10-K| K10[10-K Processor]
    Classify -->|10-Q| Q10[10-Q Processor]
    Classify -->|8-K| K8[8-K Processor]
    Classify -->|Other| Other[Generic Processor]
    
    K10 --> Extract[Financial Extractor]
    Q10 --> Extract
    K8 --> Extract
    Other --> Extract
    
    Extract --> Tables[Table Parser]
    Extract --> Text[Text Analyzer]
    
    Tables --> XBRL[XBRL Converter]
    Text --> XBRL
    
    XBRL --> Validate[Validator]
    Validate --> Output[SEC Filing Object]
```

## Deployment Architecture (Phase 7)

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[HAProxy/Nginx]
    end
    
    subgraph "API Servers"
        API1[API Server 1]
        API2[API Server 2]
        API3[API Server 3]
    end
    
    subgraph "Processing Nodes"
        Node1[Processing Node 1<br/>8 Agents]
        Node2[Processing Node 2<br/>8 Agents]
        Node3[Processing Node 3<br/>8 Agents]
    end
    
    subgraph "Storage"
        Cache[Redis Cache]
        DB[PostgreSQL]
        S3[S3 Storage]
    end
    
    subgraph "Monitoring"
        Prom[Prometheus]
        Graf[Grafana]
        Trace[Jaeger]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> Node1
    API2 --> Node2
    API3 --> Node3
    
    Node1 --> Cache
    Node2 --> Cache
    Node3 --> Cache
    
    Node1 --> DB
    Node2 --> DB
    Node3 --> DB
    
    Node1 --> S3
    Node2 --> S3
    Node3 --> S3
    
    API1 --> Prom
    API2 --> Prom
    API3 --> Prom
    
    Prom --> Graf
    API1 --> Trace
    API2 --> Trace
    API3 --> Trace
```

## Performance Metrics Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Swarm
    participant Neural
    participant Monitor
    
    Client->>API: Process PDF Request
    API->>Monitor: Start Trace
    API->>Swarm: Distribute Task
    Swarm->>Neural: Process with AI
    Neural-->>Swarm: Return Results
    Swarm-->>API: Aggregate Results
    API->>Monitor: Record Metrics
    API-->>Client: Return Response
    Monitor-->>Monitor: Update Dashboard
```

## Technology Stack

```mermaid
mindmap
  root((NeuralDocFlow))
    Rust Core
      pdf-extract
      tokio
      rayon
      serde
    Swarm
      tokio channels
      dashmap
      uuid
    Neural
      RUV-FANN
      ndarray
      WASM
    Intelligence
      ONNX Runtime
      tokenizers
      transformers
    API
      PyO3
      actix-web
      async-graphql
      tungstenite
    Production
      OpenTelemetry
      Prometheus
      Docker
      Kubernetes
```