# ============================================================================
# COUCHE INFRASTRUCTURE - SYSTÈME LAM
# 7 Modules Fondamentaux avec Garanties Formelles
# ============================================================================

from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json
import uuid
import threading
from queue import Queue, PriorityQueue


# ============================================================================
# MODULE 1: DAG MANAGER
# Gestion des graphes acycliques et structures de workflow
# Garantie: INV1 (Acyclicité)
# ============================================================================

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowNode:
    """Nœud atomique d'un workflow"""
    id: str
    action_id: str
    parameters: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowDAG:
    """Représentation complète d'un workflow"""
    id: str
    nodes: Dict[str, WorkflowNode]
    edges: Dict[str, Set[str]]  # node_id -> set of dependent node_ids
    start_nodes: Set[str]
    goal_nodes: Set[str]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DAGManager:
    """
    Gestionnaire de DAG avec vérification d'acyclicité (INV1)
    Garantie: Toujours maintient un graphe acyclique dirigé
    """
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDAG] = {}
        self._lock = threading.Lock()
    
    def create_workflow(self, metadata: Optional[Dict] = None) -> str:
        """Crée un nouveau workflow vide"""
        workflow_id = str(uuid.uuid4())
        workflow = WorkflowDAG(
            id=workflow_id,
            nodes={},
            edges={},
            start_nodes=set(),
            goal_nodes=set(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.workflows[workflow_id] = workflow
        
        return workflow_id
    
    def add_node(self, workflow_id: str, node: WorkflowNode) -> bool:
        """Ajoute un nœud au workflow"""
        with self._lock:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            workflow.nodes[node.id] = node
            workflow.edges[node.id] = set()
            
            # Si pas de dépendances, c'est un nœud de départ
            if not node.dependencies:
                workflow.start_nodes.add(node.id)
            
            return True
    
    def add_edge(self, workflow_id: str, from_node: str, to_node: str) -> bool:
        """
        Ajoute une arête entre deux nœuds
        Vérifie l'acyclicité avant insertion (Garantie INV1)
        """
        with self._lock:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            
            # Vérification d'existence des nœuds
            if from_node not in workflow.nodes or to_node not in workflow.nodes:
                return False
            
            # Vérification d'acyclicité AVANT insertion
            if self._would_create_cycle(workflow, from_node, to_node):
                raise ValueError(f"Adding edge {from_node}->{to_node} would create a cycle (INV1 violation)")
            
            # Ajout de l'arête
            workflow.edges[from_node].add(to_node)
            workflow.nodes[to_node].dependencies.add(from_node)
            
            # Mise à jour des nœuds de départ
            if to_node in workflow.start_nodes:
                workflow.start_nodes.remove(to_node)
            
            return True
    
    def _would_create_cycle(self, workflow: WorkflowDAG, from_node: str, to_node: str) -> bool:
        """
        Détection de cycle via DFS
        Retourne True si l'ajout de l'arête créerait un cycle
        """
        visited = set()
        
        def dfs(node: str) -> bool:
            if node == from_node:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            for neighbor in workflow.edges.get(node, []):
                if dfs(neighbor):
                    return True
            
            return False
        
        return dfs(to_node)
    
    def get_topological_order(self, workflow_id: str) -> List[str]:
        """
        Retourne un ordre topologique des nœuds
        Garantie: Ordre valide si et seulement si le graphe est acyclique
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return []
        
        in_degree = defaultdict(int)
        for node_id in workflow.nodes:
            in_degree[node_id] = len(workflow.nodes[node_id].dependencies)
        
        queue = Queue()
        for node_id in workflow.start_nodes:
            queue.put(node_id)
        
        result = []
        while not queue.empty():
            node_id = queue.get()
            result.append(node_id)
            
            for dependent in workflow.edges[node_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.put(dependent)
        
        # Vérification: si tous les nœuds ne sont pas dans le résultat, il y a un cycle
        if len(result) != len(workflow.nodes):
            raise ValueError("Cycle detected in workflow (INV1 violation)")
        
        return result
    
    def verify_acyclicity(self, workflow_id: str) -> bool:
        """Vérification explicite de l'invariant INV1"""
        try:
            self.get_topological_order(workflow_id)
            return True
        except ValueError:
            return False


# ============================================================================
# MODULE 2: KNOWLEDGE BASE
# Stockage des patterns, workflows historiques et métriques
# Garantie: Amélioration bornes PAC via enrichissement
# ============================================================================

@dataclass
class WorkflowPattern:
    """Pattern de workflow réutilisable"""
    id: str
    name: str
    domain: str  # Finance, Healthcare, etc.
    dag_structure: Dict[str, Any]
    success_rate: float
    avg_performance: float
    usage_count: int = 0
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowExecution:
    """Historique d'exécution d'un workflow"""
    workflow_id: str
    pattern_id: Optional[str]
    context: Dict[str, Any]
    performance_score: float
    execution_time: float
    node_scores: Dict[str, float]
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)

class KnowledgeBase:
    """
    Base de connaissances évolutive
    Stockage: Patterns + Historique d'exécutions
    Garantie: Support pour calcul de bornes PAC
    """
    
    def __init__(self):
        self.patterns: Dict[str, WorkflowPattern] = {}
        self.executions: List[WorkflowExecution] = []
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
    
    def add_pattern(self, pattern: WorkflowPattern):
        """Ajoute un pattern à la base"""
        with self._lock:
            self.patterns[pattern.id] = pattern
            self.domain_index[pattern.domain].add(pattern.id)
    
    def search_patterns(self, domain: str, context: Dict[str, Any], 
                       top_k: int = 5) -> List[Tuple[WorkflowPattern, float]]:
        """
        Recherche de patterns similaires avec scoring
        Retourne: Liste de (pattern, similarity_score)
        """
        candidates = []
        
        with self._lock:
            pattern_ids = self.domain_index.get(domain, set())
            
            for pattern_id in pattern_ids:
                pattern = self.patterns[pattern_id]
                similarity = self._compute_similarity(pattern, context)
                candidates.append((pattern, similarity))
        
        # Tri par similarité décroissante
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _compute_similarity(self, pattern: WorkflowPattern, context: Dict[str, Any]) -> float:
        """
        Calcul de similarité contextuelle
        Méthode simple basée sur l'intersection des clés
        """
        if not pattern.contexts:
            return 0.5
        
        # Moyenne des similarités avec les contextes historiques
        similarities = []
        for hist_context in pattern.contexts:
            common_keys = set(context.keys()) & set(hist_context.keys())
            if not common_keys:
                similarities.append(0.0)
                continue
            
            matching = sum(1 for k in common_keys if context.get(k) == hist_context.get(k))
            similarities.append(matching / len(common_keys))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def record_execution(self, execution: WorkflowExecution):
        """Enregistre une exécution de workflow"""
        with self._lock:
            self.executions.append(execution)
            
            # Mise à jour du pattern associé
            if execution.pattern_id and execution.pattern_id in self.patterns:
                pattern = self.patterns[execution.pattern_id]
                pattern.usage_count += 1
                pattern.contexts.append(execution.context)
                
                # Mise à jour moyenne mobile du taux de succès
                alpha = 0.1  # Facteur de lissage
                if execution.success:
                    pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * 1.0
                else:
                    pattern.success_rate = (1 - alpha) * pattern.success_rate
                
                # Mise à jour de la performance moyenne
                pattern.avg_performance = (
                    (1 - alpha) * pattern.avg_performance + 
                    alpha * execution.performance_score
                )
    
    def compute_pac_bounds(self, domain: str, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calcul des bornes PAC pour la généralisation
        Formule: Err_gen ≤ Err_emp + sqrt(VC_dim / n) + O(ln(1/δ) / n)
        """
        import math
        
        with self._lock:
            domain_executions = [
                ex for ex in self.executions 
                if ex.pattern_id in self.domain_index.get(domain, set())
            ]
            
            n = len(domain_executions)
            if n == 0:
                return {"error": float('inf'), "confidence": 0.0}
            
            # Erreur empirique
            failures = sum(1 for ex in domain_executions if not ex.success)
            err_empirical = failures / n
            
            # VC dimension (estimation heuristique basée sur la complexité)
            vc_dim = len(self.domain_index.get(domain, set())) * 2  # Approximation
            
            # Terme de confiance
            delta = 1 - confidence
            confidence_term = math.sqrt(math.log(1 / delta) / (2 * n))
            
            # Borne PAC
            complexity_term = math.sqrt(vc_dim / n)
            err_generalization = err_empirical + complexity_term + confidence_term
            
            return {
                "empirical_error": err_empirical,
                "generalization_bound": err_generalization,
                "sample_size": n,
                "vc_dimension": vc_dim,
                "confidence": confidence
            }


# ============================================================================
# MODULE 3: ACTION CATALOG
# Catalogue des actions atomiques avec contrats
# Garantie: Préconditions/Postconditions formelles
# ============================================================================

@dataclass
class ActionContract:
    """Contrat formel d'une action"""
    preconditions: List[str]
    postconditions: List[str]
    invariants: List[str]
    side_effects: List[str]

@dataclass
class ActionSpec:
    """Spécification complète d'une action atomique"""
    id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    contract: ActionContract
    implementation: Callable
    timeout: float = 30.0
    retries: int = 3
    robustness_epsilon: float = 0.1  # Borne de robustesse locale

class ActionCatalog:
    """
    Catalogue des actions atomiques
    Garantie: Toutes les actions ont des contrats vérifiables
    """
    
    def __init__(self):
        self.actions: Dict[str, ActionSpec] = {}
        self._lock = threading.Lock()
    
    def register_action(self, action: ActionSpec):
        """Enregistre une action dans le catalogue"""
        with self._lock:
            self.actions[action.id] = action
    
    def get_action(self, action_id: str) -> Optional[ActionSpec]:
        """Récupère une action par son ID"""
        return self.actions.get(action_id)
    
    def verify_contract_compatibility(self, from_action: str, to_action: str) -> bool:
        """
        Vérifie la compatibilité des contrats entre deux actions
        Postconditions(A) ⊆ Preconditions(B) pour A -> B
        """
        action_a = self.actions.get(from_action)
        action_b = self.actions.get(to_action)
        
        if not action_a or not action_b:
            return False
        
        # Vérification simple: les postconditions de A doivent satisfaire les préconditions de B
        post_a = set(action_a.contract.postconditions)
        pre_b = set(action_b.contract.preconditions)
        
        # Au moins une postcondition de A doit satisfaire chaque précondition de B
        return len(pre_b - post_a) == 0
    
    def search_actions(self, domain: str = None, 
                      capabilities: List[str] = None) -> List[ActionSpec]:
        """Recherche d'actions par domaine ou capacités"""
        results = []
        
        for action in self.actions.values():
            if domain and domain not in action.description.lower():
                continue
            if capabilities:
                # Recherche dans les postconditions
                action_capabilities = set(action.contract.postconditions)
                if not any(cap in action_capabilities for cap in capabilities):
                    continue
            
            results.append(action)
        
        return results


# ============================================================================
# MODULE 4: RESOURCE MANAGER
# Gestion budgets, SLA, allocation ressources
# Garantie: INV2 (Conservation de ressources)
# ============================================================================

@dataclass
class ResourceBudget:
    """Budget de ressources pour un workflow"""
    cpu_seconds: float
    memory_mb: float
    api_calls: int
    cost_usd: float
    max_duration_seconds: float

@dataclass
class ResourceUsage:
    """Utilisation effective des ressources"""
    cpu_seconds: float = 0.0
    memory_mb: float = 0.0
    api_calls: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0

class ResourceManager:
    """
    Gestionnaire de ressources avec garantie INV2
    Garantie: budget_utilisé ≤ budget_alloué
    """
    
    def __init__(self):
        self.budgets: Dict[str, ResourceBudget] = {}
        self.usage: Dict[str, ResourceUsage] = {}
        self._lock = threading.Lock()
    
    def allocate_budget(self, workflow_id: str, budget: ResourceBudget):
        """Alloue un budget à un workflow"""
        with self._lock:
            self.budgets[workflow_id] = budget
            self.usage[workflow_id] = ResourceUsage()
    
    def check_budget_available(self, workflow_id: str, 
                              requested: ResourceUsage) -> bool:
        """
        Vérifie si le budget permet l'allocation demandée
        Garantie INV2: Empêche les dépassements
        """
        with self._lock:
            if workflow_id not in self.budgets:
                return False
            
            budget = self.budgets[workflow_id]
            current = self.usage[workflow_id]
            
            # Vérification de chaque dimension
            checks = [
                current.cpu_seconds + requested.cpu_seconds <= budget.cpu_seconds,
                current.memory_mb + requested.memory_mb <= budget.memory_mb,
                current.api_calls + requested.api_calls <= budget.api_calls,
                current.cost_usd + requested.cost_usd <= budget.cost_usd,
                current.duration_seconds + requested.duration_seconds <= budget.max_duration_seconds
            ]
            
            return all(checks)
    
    def consume_resources(self, workflow_id: str, consumed: ResourceUsage) -> bool:
        """
        Consomme des ressources si le budget le permet
        Garantie: Transaction atomique (tout ou rien)
        """
        with self._lock:
            if not self.check_budget_available(workflow_id, consumed):
                return False
            
            current = self.usage[workflow_id]
            current.cpu_seconds += consumed.cpu_seconds
            current.memory_mb += consumed.memory_mb
            current.api_calls += consumed.api_calls
            current.cost_usd += consumed.cost_usd
            current.duration_seconds += consumed.duration_seconds
            
            return True
    
    def get_remaining_budget(self, workflow_id: str) -> Optional[ResourceBudget]:
        """Retourne le budget restant"""
        with self._lock:
            if workflow_id not in self.budgets:
                return None
            
            budget = self.budgets[workflow_id]
            current = self.usage[workflow_id]
            
            return ResourceBudget(
                cpu_seconds=budget.cpu_seconds - current.cpu_seconds,
                memory_mb=budget.memory_mb - current.memory_mb,
                api_calls=budget.api_calls - current.api_calls,
                cost_usd=budget.cost_usd - current.cost_usd,
                max_duration_seconds=budget.max_duration_seconds - current.duration_seconds
            )


# ============================================================================
# MODULE 5: STATE MANAGER
# Gestion états workflow, transitions, historique
# Garantie: INV4 (État sûr - chemin vers goal ou safe_exit)
# ============================================================================

@dataclass
class WorkflowState:
    """État complet d'un workflow à un instant t"""
    workflow_id: str
    current_nodes: Set[str]  # Nœuds en cours d'exécution
    completed_nodes: Set[str]
    failed_nodes: Set[str]
    global_context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class StateManager:
    """
    Gestionnaire d'états avec historique
    Garantie: Traçabilité complète des transitions
    """
    
    def __init__(self, dag_manager: DAGManager):
        self.dag_manager = dag_manager
        self.current_states: Dict[str, WorkflowState] = {}
        self.state_history: Dict[str, List[WorkflowState]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def initialize_state(self, workflow_id: str, initial_context: Dict[str, Any]):
        """Initialise l'état d'un workflow"""
        workflow = self.dag_manager.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        state = WorkflowState(
            workflow_id=workflow_id,
            current_nodes=workflow.start_nodes.copy(),
            completed_nodes=set(),
            failed_nodes=set(),
            global_context=initial_context
        )
        
        with self._lock:
            self.current_states[workflow_id] = state
            self.state_history[workflow_id].append(state)
    
    def transition_node(self, workflow_id: str, node_id: str, 
                       status: NodeStatus, result: Any = None):
        """
        Effectue une transition d'état pour un nœud
        Met à jour l'état global du workflow
        """
        with self._lock:
            state = self.current_states.get(workflow_id)
            if not state:
                raise ValueError(f"No state for workflow {workflow_id}")
            
            # Mise à jour selon le statut
            if status == NodeStatus.SUCCESS:
                state.current_nodes.discard(node_id)
                state.completed_nodes.add(node_id)
                
                # Activation des nœuds dépendants
                workflow = self.dag_manager.workflows[workflow_id]
                for dependent in workflow.edges[node_id]:
                    # Vérifier que toutes les dépendances sont satisfaites
                    deps = workflow.nodes[dependent].dependencies
                    if deps.issubset(state.completed_nodes):
                        state.current_nodes.add(dependent)
            
            elif status == NodeStatus.FAILED:
                state.current_nodes.discard(node_id)
                state.failed_nodes.add(node_id)
            
            # Sauvegarde dans l'historique
            self.state_history[workflow_id].append(state)
    
    def verify_safe_state(self, workflow_id: str) -> bool:
        """
        Vérifie INV4: ∃ chemin vers goal OU safe_exit
        Garantie: Détection de deadlock
        """
        state = self.current_states.get(workflow_id)
        workflow = self.dag_manager.workflows.get(workflow_id)
        
        if not state or not workflow:
            return False
        
        # Si aucun nœud en cours et des nœuds non complétés, c'est un deadlock
        if not state.current_nodes and len(state.completed_nodes) < len(workflow.nodes):
            return False
        
        # Vérification qu'un chemin vers un goal est possible
        reachable_goals = self._find_reachable_goals(workflow_id)
        return len(reachable_goals) > 0
    
    def _find_reachable_goals(self, workflow_id: str) -> Set[str]:
        """Trouve les nœuds goal atteignables depuis l'état courant"""
        state = self.current_states.get(workflow_id)
        workflow = self.dag_manager.workflows.get(workflow_id)
        
        if not state or not workflow:
            return set()
        
        # BFS depuis les nœuds courants
        visited = set()
        queue = Queue()
        
        for node in state.current_nodes:
            queue.put(node)
        
        reachable_goals = set()
        
        while not queue.empty():
            node = queue.get()
            if node in visited:
                continue
            
            visited.add(node)
            
            # Si c'est un goal, l'ajouter
            if node in workflow.goal_nodes:
                reachable_goals.add(node)
            
            # Explorer les dépendants
            for dependent in workflow.edges.get(node, []):
                if dependent not in visited:
                    queue.put(dependent)
        
        return reachable_goals


# ============================================================================
# MODULE 6: EVENT BUS
# Communication asynchrone entre modules
# Garantie: Découplage et extensibilité
# ============================================================================

@dataclass
class Event:
    """Événement système"""
    type: str
    workflow_id: str
    node_id: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class EventBus:
    """
    Bus d'événements pour communication inter-modules
    Pattern Publisher-Subscriber
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: Queue = Queue()
        self._lock = threading.Lock()
        self._running = False
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Abonne un handler à un type d'événement"""
        with self._lock:
            self.subscribers[event_type].append(handler)
    
    def publish(self, event: Event):
        """Publie un événement"""
        self.event_queue.put(event)
    
    def start(self):
        """Démarre le traitement des événements"""
        self._running = True
        
        def process_events():
            while self._running:
                try:
                    event = self.event_queue.get(timeout=1.0)
                    self._dispatch_event(event)
                except:
                    pass
        
        thread = threading.Thread(target=process_events, daemon=True)
        thread.start()
    
    def stop(self):
        """Arrête le traitement des événements"""
        self._running = False
    
    def _dispatch_event(self, event: Event):
        """Dispatche un événement vers ses subscribers"""
        with self._lock:
            handlers = self.subscribers.get(event.type, [])
            handlers.extend(self.subscribers.get("*", []))  # Wildcard handlers
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler: {e}")


# ============================================================================
# MODULE 7: LOGGER & TRACER
# Observabilité profonde, traces structurées
# Garantie: Traçabilité complète pour debugging et audit
# ============================================================================

@dataclass
class TraceEntry:
    """Entrée de trace structurée"""
    trace_id: str
    workflow_id: str
    node_id: Optional[str]
    phase: str  # construction, execution, validation, adaptation, backprop
    event_type: str
    context: Dict[str, Any]
    metrics: Dict[str, float]
    decision: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class LoggerTracer:
    """
    Système d'observabilité profonde
    Garantie: Traces structurées pour analyse post-mortem
    """
    
    def __init__(self):
        self.traces: Dict[str, List[TraceEntry]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def log_trace(self, entry: TraceEntry):
        """Enregistre une entrée de trace"""
        with self._lock:
            self.traces[entry.trace_id].append(entry)
    
    def get_workflow_trace(self, workflow_id: str) -> List[TraceEntry]:
        """Récupère toutes les traces d'un workflow"""
        result = []
        with self._lock:
            for trace_id, entries in self.traces.items():
                result.extend([e for e in entries if e.workflow_id == workflow_id])
        
        # Tri chronologique
        result.sort(key=lambda e: e.timestamp)
        return result
    
    def export_trace_json(self, trace_id: str) -> str:
        """Exporte une trace au format JSON"""
        with self._lock:
            entries = self.traces.get(trace_id, [])
        
        trace_data = {
            "trace_id": trace_id,
            "entry_count": len(entries),
            "entries": [
                {
                    "workflow_id": e.workflow_id,
                    "node_id": e.node_id,
                    "phase": e.phase,
                    "event_type": e.event_type,
                    "context": e.context,
                    "metrics": e.metrics,
                    "decision": e.decision,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in entries
            ]
        }
        
        return json.dumps(trace_data, indent=2)
    
    def analyze_performance(self, workflow_id: str) -> Dict[str, Any]:
        """Analyse de performance d'un workflow"""
        traces = self.get_workflow_trace(workflow_id)
        
        if not traces:
            return {"error": "No traces found"}
        
        # Calcul de métriques agrégées
        phases = defaultdict(list)
        for trace in traces:
            phases[trace.phase].append(trace)
        
        analysis = {
            "workflow_id": workflow_id,
            "total_duration": (traces[-1].timestamp - traces[0].timestamp).total_seconds(),
            "phases": {}
        }
        
        for phase, entries in phases.items():
            phase_metrics = [e.metrics for e in entries if e.metrics]
            analysis["phases"][phase] = {
                "event_count": len(entries),
                "avg_metrics": self._average_metrics(phase_metrics) if phase_metrics else {}
            }
        
        return analysis
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calcule la moyenne des métriques"""
        if not metrics_list:
            return {}
        
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        result = {}
        for key in all_keys:
            values = [m.get(key, 0.0) for m in metrics_list]
            result[key] = sum(values) / len(values)
        
        return result


# ============================================================================
# EXEMPLE D'UTILISATION INTÉGRÉE
# ============================================================================

def example_integrated_usage():
    """Exemple d'utilisation des 7 modules ensemble"""
    
    # 1. Initialisation des modules
    dag_manager = DAGManager()
    knowledge_base = KnowledgeBase()
    action_catalog = ActionCatalog()
    resource_manager = ResourceManager()
    state_manager = StateManager(dag_manager)
    event_bus = EventBus()
    logger = LoggerTracer()
    
    # 2. Enregistrement d'actions atomiques
    def mock_action(input_data):
        return {"result": "success", "data": input_data}
    
    action1 = ActionSpec(
        id="extract_data",
        name="Extract Data",
        description="Extract data from source",
        input_schema={"source": "string"},
        output_schema={"data": "object"},
        contract=ActionContract(
            preconditions=["source_available"],
            postconditions=["data_extracted"],
            invariants=["data_integrity"],
            side_effects=[]
        ),
        implementation=mock_action
    )
    action_catalog.register_action(action1)
    
    # 3. Création d'un workflow
    trace_id = str(uuid.uuid4())
    workflow_id = dag_manager.create_workflow({"domain": "Finance"})
    
    # Log de création
    logger.log_trace(TraceEntry(
        trace_id=trace_id,
        workflow_id=workflow_id,
        node_id=None,
        phase="construction",
        event_type="workflow_created",
        context={"domain": "Finance"},
        metrics={},
        decision={"workflow_id": workflow_id}
    ))
    
    # 4. Ajout de nœuds
    node1 = WorkflowNode(
        id="node_1",
        action_id="extract_data",
        parameters={"source": "database"}
    )
    dag_manager.add_node(workflow_id, node1)
    
    # 5. Allocation de budget
    budget = ResourceBudget(
        cpu_seconds=300.0,
        memory_mb=1024.0,
        api_calls=100,
        cost_usd=5.0,
        max_duration_seconds=600.0
    )
    resource_manager.allocate_budget(workflow_id, budget)
    
    # 6. Initialisation de l'état
    state_manager.initialize_state(workflow_id, {"user": "analyst_1"})
    
    # 7. Événement de démarrage
    event_bus.publish(Event(
        type="workflow.started",
        workflow_id=workflow_id,
        node_id=None,
        data={"status": "initialized"}
    ))
    
    # 8. Vérification des garanties
    print(f"Workflow acyclique (INV1): {dag_manager.verify_acyclicity(workflow_id)}")
    print(f"Budget disponible (INV2): {resource_manager.get_remaining_budget(workflow_id)}")
    print(f"État sûr (INV4): {state_manager.verify_safe_state(workflow_id)}")
    
    # 9. Export de trace
    trace_json = logger.export_trace_json(trace_id)
    print(f"\nTrace JSON:\n{trace_json}")
    
    return {
        "workflow_id": workflow_id,
        "trace_id": trace_id,
        "modules": {
            "dag_manager": dag_manager,
            "knowledge_base": knowledge_base,
            "action_catalog": action_catalog,
            "resource_manager": resource_manager,
            "state_manager": state_manager,
            "event_bus": event_bus,
            "logger": logger
        }
    }


if __name__ == "__main__":
    print("=" * 80)
    print("COUCHE INFRASTRUCTURE - 7 MODULES LAM")
    print("=" * 80)
    print("\nModules implémentés:")
    print("1. DAG Manager         - Garantie INV1 (Acyclicité)")
    print("2. Knowledge Base      - Bornes PAC, patterns historiques")
    print("3. Action Catalog      - Contrats formels préconditions/postconditions")
    print("4. Resource Manager    - Garantie INV2 (Conservation ressources)")
    print("5. State Manager       - Garantie INV4 (États sûrs)")
    print("6. Event Bus           - Communication asynchrone découplée")
    print("7. Logger & Tracer     - Observabilité profonde, traces structurées")
    print("\n" + "=" * 80)
    print("\nExécution de l'exemple intégré...\n")
    
    result = example_integrated_usage()
    print(f"\n✓ Workflow créé: {result['workflow_id']}")
    print(f"✓ Trace ID: {result['trace_id']}")
    print("\n" + "=" * 80)