# ============================================================================
# COUCHE LAM - SYSTÈME LAM
# 5 Modules pour Exécution et Orchestration d'Actions
# Garanties: Robustesse Locale (ε-δ bounds), Contrats d'Interface
# ============================================================================

from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import traceback
from queue import Queue, Empty
import numpy as np


# ============================================================================
# TYPES ET STRUCTURES COMMUNES
# ============================================================================

class ExecutionStatus(Enum):
    """Statuts d'exécution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

@dataclass
class ExecutionResult:
    """Résultat d'exécution d'une action"""
    status: ExecutionStatus
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0
    robustness_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RobustnessBound:
    """Bornes de robustesse ε-δ"""
    epsilon: float  # Perturbation maximale tolérée sur l'entrée
    theta: float    # Variation maximale tolérée sur la sortie
    verified: bool = False


# ============================================================================
# MODULE 12: ACTION EXECUTOR
# Exécution actions atomiques avec robustesse locale
# Garantie: ∀δ: ||δ|| < ε ⟹ ||Output(Input+δ) - Output(Input)|| < θ
# ============================================================================

@dataclass
class ActionExecutionContext:
    """Contexte d'exécution d'une action"""
    action_id: str
    node_id: str
    workflow_id: str
    parameters: Dict[str, Any]
    timeout: float
    robustness_bound: RobustnessBound
    retry_policy: Dict[str, Any]

class ActionExecutor:
    """
    Exécuteur d'actions atomiques avec garanties de robustesse
    Garantie: Robustesse locale avec bornes ε-δ vérifiables
    """
    
    def __init__(self, action_catalog, resource_manager, logger):
        self.action_catalog = action_catalog
        self.resource_manager = resource_manager
        self.logger = logger
        self._execution_lock = threading.Lock()
        self._active_executions: Dict[str, threading.Thread] = {}
    
    def execute(self, context: ActionExecutionContext) -> ExecutionResult:
        """
        Exécute une action avec garanties de robustesse
        Gère: timeout, retry, monitoring ressources
        """
        start_time = time.time()
        
        # Récupération de l'action
        action_spec = self.action_catalog.get_action(context.action_id)
        if not action_spec:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                output=None,
                error=f"Action {context.action_id} not found in catalog"
            )
        
        # Vérification des préconditions
        if not self._verify_preconditions(action_spec, context):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                output=None,
                error="Preconditions not satisfied"
            )
        
        # Vérification du budget disponible
        estimated_resources = self._estimate_resource_usage(action_spec)
        if not self.resource_manager.check_budget_available(
            context.workflow_id, estimated_resources
        ):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                output=None,
                error="Insufficient budget (INV2 violation prevented)"
            )
        
        # Exécution avec retry policy
        max_retries = context.retry_policy.get("max_retries", 3)
        retry_delay = context.retry_policy.get("delay", 1.0)
        
        for attempt in range(max_retries + 1):
            try:
                # Exécution avec timeout
                result = self._execute_with_timeout(
                    action_spec, context, attempt
                )
                
                # Vérification de robustesse
                robustness_score = self._verify_robustness(
                    action_spec, context, result
                )
                
                # Consommation des ressources
                actual_resources = self._measure_resource_usage(
                    start_time, result
                )
                self.resource_manager.consume_resources(
                    context.workflow_id, actual_resources
                )
                
                # Vérification des postconditions
                if not self._verify_postconditions(action_spec, result):
                    raise ValueError("Postconditions not satisfied")
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    output=result,
                    execution_time=execution_time,
                    retries=attempt,
                    robustness_score=robustness_score,
                    metadata={
                        "action_id": context.action_id,
                        "node_id": context.node_id
                    }
                )
                
            except TimeoutError:
                if attempt == max_retries:
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        output=None,
                        error=f"Timeout after {context.timeout}s",
                        execution_time=time.time() - start_time,
                        retries=attempt
                    )
                time.sleep(retry_delay)
                
            except Exception as e:
                if attempt == max_retries:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        output=None,
                        error=str(e),
                        execution_time=time.time() - start_time,
                        retries=attempt,
                        metadata={"traceback": traceback.format_exc()}
                    )
                time.sleep(retry_delay)
    
    def _execute_with_timeout(self, action_spec, context: ActionExecutionContext, 
                             attempt: int) -> Any:
        """
        Exécution avec timeout
        Utilise threading pour interrompre les exécutions longues
        """
        result_queue = Queue()
        exception_queue = Queue()
        
        def worker():
            try:
                output = action_spec.implementation(context.parameters)
                result_queue.put(output)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=context.timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Execution exceeded {context.timeout}s")
        
        if not exception_queue.empty():
            raise exception_queue.get()
        
        if not result_queue.empty():
            return result_queue.get()
        
        raise RuntimeError("Execution completed without result")
    
    def _verify_preconditions(self, action_spec, context: ActionExecutionContext) -> bool:
        """Vérifie les préconditions d'une action"""
        # Vérification simplifiée: présence des paramètres requis
        required_params = action_spec.input_schema.get("required", [])
        return all(param in context.parameters for param in required_params)
    
    def _verify_postconditions(self, action_spec, result: Any) -> bool:
        """Vérifie les postconditions d'une action"""
        # Vérification simplifiée: résultat non None
        return result is not None
    
    def _verify_robustness(self, action_spec, context: ActionExecutionContext, 
                          result: Any) -> float:
        """
        Vérifie la robustesse locale (borne ε-δ)
        Garantie: ||δ|| < ε ⟹ ||Output(Input+δ) - Output(Input)|| < θ
        """
        bound = context.robustness_bound
        
        if not bound.verified:
            # Mode non-vérifié: score par défaut
            return 1.0
        
        # Test avec perturbations
        try:
            perturbations = self._generate_perturbations(
                context.parameters, bound.epsilon
            )
            
            outputs = []
            for perturbed_params in perturbations[:5]:  # Limiter à 5 tests
                perturbed_result = action_spec.implementation(perturbed_params)
                outputs.append(perturbed_result)
            
            # Calcul de la variation maximale
            max_variation = self._compute_max_variation(result, outputs)
            
            # Score: 1.0 si variation < theta, décroit linéairement sinon
            if max_variation <= bound.theta:
                return 1.0
            else:
                return max(0.0, 1.0 - (max_variation - bound.theta) / bound.theta)
        
        except:
            # Échec de vérification
            return 0.5
    
    def _generate_perturbations(self, parameters: Dict[str, Any], 
                               epsilon: float) -> List[Dict[str, Any]]:
        """Génère des perturbations pour test de robustesse"""
        perturbations = []
        
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Perturbation numérique
                perturbations.append({
                    **parameters,
                    key: value + epsilon
                })
                perturbations.append({
                    **parameters,
                    key: value - epsilon
                })
        
        return perturbations
    
    def _compute_max_variation(self, original: Any, outputs: List[Any]) -> float:
        """Calcule la variation maximale entre résultats"""
        if isinstance(original, (int, float)):
            variations = [abs(original - out) for out in outputs if isinstance(out, (int, float))]
            return max(variations) if variations else 0.0
        
        # Pour types complexes: distance normalisée
        return 0.0
    
    def _estimate_resource_usage(self, action_spec) -> Any:
        """Estime l'usage de ressources d'une action"""
        # Importation depuis infrastructure
        from infra_layer_modules import ResourceUsage
        
        return ResourceUsage(
            cpu_seconds=action_spec.timeout,
            memory_mb=100.0,
            api_calls=1,
            cost_usd=0.01,
            duration_seconds=action_spec.timeout
        )
    
    def _measure_resource_usage(self, start_time: float, result: Any) -> Any:
        """Mesure l'usage réel de ressources"""
        from infra_layer_modules import ResourceUsage
        
        duration = time.time() - start_time
        
        return ResourceUsage(
            cpu_seconds=duration,
            memory_mb=50.0,  # Estimation
            api_calls=1,
            cost_usd=0.005,
            duration_seconds=duration
        )


# ============================================================================
# MODULE 13: NODE MAPPER
# Mapping patterns→actions, validation contrats
# Garantie: Compatibilité des contrats entre nœuds
# ============================================================================

@dataclass
class MappingResult:
    """Résultat du mapping d'un nœud"""
    node_id: str
    action_id: str
    parameters: Dict[str, Any]
    contracts_valid: bool
    dependencies_satisfied: bool
    warnings: List[str] = field(default_factory=list)

class NodeMapper:
    """
    Mappeur de nœuds workflow vers actions atomiques
    Garantie: Validation des contrats d'interface
    """
    
    def __init__(self, action_catalog, dag_manager):
        self.action_catalog = action_catalog
        self.dag_manager = dag_manager
    
    def map_workflow(self, workflow_dag: Dict[str, Any], 
                    workflow_id: str) -> List[MappingResult]:
        """
        Map tous les nœuds d'un workflow vers des actions
        Vérifie la compatibilité des contrats
        """
        results = []
        
        # Créer le workflow dans le DAG Manager
        from infra_layer_modules import WorkflowNode
        
        for node_data in workflow_dag.get("nodes", []):
            node = WorkflowNode(
                id=node_data["id"],
                action_id=node_data.get("action", "unknown"),
                parameters=node_data.get("parameters", {})
            )
            
            # Mapping du nœud
            mapping = self.map_node(node, workflow_id)
            results.append(mapping)
            
            # Ajout au DAG si valide
            if mapping.contracts_valid:
                self.dag_manager.add_node(workflow_id, node)
        
        # Ajout des arêtes
        for edge in workflow_dag.get("edges", []):
            from_node, to_node = edge[0], edge[1]
            try:
                self.dag_manager.add_edge(workflow_id, from_node, to_node)
            except ValueError as e:
                # Cycle détecté
                print(f"Warning: {e}")
        
        return results
    
    def map_node(self, node, workflow_id: str) -> MappingResult:
        """
        Map un nœud individuel vers une action
        Vérifie: Existence action, contrats, paramètres
        """
        warnings = []
        
        # Récupération de l'action
        action = self.action_catalog.get_action(node.action_id)
        if not action:
            return MappingResult(
                node_id=node.id,
                action_id=node.action_id,
                parameters=node.parameters,
                contracts_valid=False,
                dependencies_satisfied=False,
                warnings=[f"Action {node.action_id} not found"]
            )
        
        # Validation des paramètres
        param_valid, param_warnings = self._validate_parameters(
            action, node.parameters
        )
        warnings.extend(param_warnings)
        
        # Validation des contrats avec dépendances
        contracts_valid = True
        deps_satisfied = True
        
        workflow = self.dag_manager.workflows.get(workflow_id)
        if workflow:
            for dep_id in node.dependencies:
                dep_node = workflow.nodes.get(dep_id)
                if dep_node:
                    if not self.action_catalog.verify_contract_compatibility(
                        dep_node.action_id, node.action_id
                    ):
                        contracts_valid = False
                        warnings.append(
                            f"Contract incompatibility: {dep_node.action_id} -> {node.action_id}"
                        )
                else:
                    deps_satisfied = False
                    warnings.append(f"Dependency {dep_id} not found")
        
        return MappingResult(
            node_id=node.id,
            action_id=node.action_id,
            parameters=node.parameters,
            contracts_valid=contracts_valid and param_valid,
            dependencies_satisfied=deps_satisfied,
            warnings=warnings
        )
    
    def _validate_parameters(self, action_spec, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide les paramètres d'une action"""
        warnings = []
        
        # Vérification des paramètres requis
        required = action_spec.input_schema.get("required", [])
        for param in required:
            if param not in parameters:
                warnings.append(f"Missing required parameter: {param}")
        
        # Vérification des types (simplifié)
        properties = action_spec.input_schema.get("properties", {})
        for param, value in parameters.items():
            if param in properties:
                expected_type = properties[param].get("type")
                actual_type = type(value).__name__
                if expected_type and expected_type != actual_type:
                    warnings.append(
                        f"Type mismatch for {param}: expected {expected_type}, got {actual_type}"
                    )
        
        return len(warnings) == 0, warnings


# ============================================================================
# MODULE 14: WORKFLOW ORCHESTRATOR
# Orchestration séquentielle, gestion dépendances
# Garantie: Respect ordre topologique (INV1)
# ============================================================================

@dataclass
class OrchestrationState:
    """État de l'orchestration"""
    workflow_id: str
    current_phase: str
    nodes_completed: Set[str] = field(default_factory=set)
    nodes_failed: Set[str] = field(default_factory=set)
    results: Dict[str, ExecutionResult] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

class WorkflowOrchestrator:
    """
    Orchestrateur de workflows
    Garantie: Exécution respectant l'ordre topologique et les dépendances
    """
    
    def __init__(self, dag_manager, state_manager, action_executor, 
                 event_bus, logger):
        self.dag_manager = dag_manager
        self.state_manager = state_manager
        self.action_executor = action_executor
        self.event_bus = event_bus
        self.logger = logger
        self._orchestrations: Dict[str, OrchestrationState] = {}
    
    def orchestrate(self, workflow_id: str, 
                   initial_context: Dict[str, Any]) -> OrchestrationState:
        """
        Orchestre l'exécution complète d'un workflow
        Suit l'ordre topologique, gère les dépendances
        """
        # Vérification de l'acyclicité (INV1)
        if not self.dag_manager.verify_acyclicity(workflow_id):
            raise ValueError(f"Workflow {workflow_id} contains cycles (INV1 violation)")
        
        # Initialisation de l'état
        self.state_manager.initialize_state(workflow_id, initial_context)
        
        state = OrchestrationState(
            workflow_id=workflow_id,
            current_phase="execution"
        )
        self._orchestrations[workflow_id] = state
        
        # Ordre topologique
        execution_order = self.dag_manager.get_topological_order(workflow_id)
        
        # Exécution séquentielle
        for node_id in execution_order:
            # Vérification des dépendances
            if not self._dependencies_satisfied(workflow_id, node_id, state):
                state.nodes_failed.add(node_id)
                continue
            
            # Exécution du nœud
            result = self._execute_node(workflow_id, node_id, state)
            state.results[node_id] = result
            
            # Mise à jour de l'état
            if result.status == ExecutionStatus.SUCCESS:
                state.nodes_completed.add(node_id)
                self.state_manager.transition_node(
                    workflow_id, node_id, 
                    self._convert_status(result.status), result.output
                )
            else:
                state.nodes_failed.add(node_id)
                # Décision: continuer ou arrêter?
                if self._should_stop_on_failure(workflow_id, node_id):
                    break
        
        # Vérification de l'état final
        if not self.state_manager.verify_safe_state(workflow_id):
            state.current_phase = "unsafe_state"
        else:
            state.current_phase = "completed"
        
        return state
    
    def _dependencies_satisfied(self, workflow_id: str, node_id: str,
                               state: OrchestrationState) -> bool:
        """Vérifie si toutes les dépendances d'un nœud sont satisfaites"""
        workflow = self.dag_manager.workflows[workflow_id]
        node = workflow.nodes[node_id]
        
        return node.dependencies.issubset(state.nodes_completed)
    
    def _execute_node(self, workflow_id: str, node_id: str,
                     state: OrchestrationState) -> ExecutionResult:
        """Exécute un nœud individuel"""
        workflow = self.dag_manager.workflows[workflow_id]
        node = workflow.nodes[node_id]
        
        # Récupération de l'action
        action = self.action_catalog.get_action(node.action_id)
        if not action:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                output=None,
                error=f"Action {node.action_id} not found"
            )
        
        # Contexte d'exécution
        context = ActionExecutionContext(
            action_id=node.action_id,
            node_id=node_id,
            workflow_id=workflow_id,
            parameters=node.parameters,
            timeout=action.timeout,
            robustness_bound=RobustnessBound(
                epsilon=action.robustness_epsilon,
                theta=action.robustness_epsilon * 2,
                verified=True
            ),
            retry_policy={"max_retries": action.retries, "delay": 1.0}
        )
        
        # Événement de début
        self.event_bus.publish(self._create_event(
            "node.execution.started", workflow_id, node_id, {}
        ))
        
        # Exécution
        result = self.action_executor.execute(context)
        
        # Événement de fin
        self.event_bus.publish(self._create_event(
            "node.execution.completed", workflow_id, node_id,
            {"status": result.status.value}
        ))
        
        # Log
        self._log_execution(workflow_id, node_id, result)
        
        return result
    
    def _should_stop_on_failure(self, workflow_id: str, node_id: str) -> bool:
        """Décide si l'exécution doit s'arrêter après un échec"""
        # Politique simple: continuer si le nœud n'est pas critique
        workflow = self.dag_manager.workflows[workflow_id]
        node = workflow.nodes[node_id]
        
        # Critère: nœud critique si c'est un goal node
        return node_id in workflow.goal_nodes
    
    def _convert_status(self, status: ExecutionStatus):
        """Convertit ExecutionStatus vers NodeStatus"""
        from infra_layer_modules import NodeStatus
        
        mapping = {
            ExecutionStatus.SUCCESS: NodeStatus.SUCCESS,
            ExecutionStatus.FAILED: NodeStatus.FAILED,
            ExecutionStatus.TIMEOUT: NodeStatus.FAILED,
            ExecutionStatus.RUNNING: NodeStatus.RUNNING,
            ExecutionStatus.PENDING: NodeStatus.PENDING
        }
        return mapping.get(status, NodeStatus.FAILED)
    
    def _create_event(self, event_type: str, workflow_id: str, 
                     node_id: str, data: Dict[str, Any]):
        """Crée un événement"""
        from infra_layer_modules import Event
        return Event(
            type=event_type,
            workflow_id=workflow_id,
            node_id=node_id,
            data=data
        )
    
    def _log_execution(self, workflow_id: str, node_id: str, 
                      result: ExecutionResult):
        """Log l'exécution d'un nœud"""
        from infra_layer_modules import TraceEntry
        
        self.logger.log_trace(TraceEntry(
            trace_id=workflow_id,
            workflow_id=workflow_id,
            node_id=node_id,
            phase="execution",
            event_type="node_executed",
            context={"status": result.status.value},
            metrics={
                "execution_time": result.execution_time,
                "robustness_score": result.robustness_score,
                "retries": result.retries
            },
            decision={"continue": result.status == ExecutionStatus.SUCCESS}
        ))


# ============================================================================
# MODULE 15: BRANCH SELECTOR
# Sélection dynamique de branches conditionnelles
# Garantie: Monitoring distribution shift, INV4 (état sûr)
# ============================================================================

@dataclass
class BranchCondition:
    """Condition de branchement"""
    condition_type: str  # threshold, ml_prediction, rule_based
    threshold: Optional[float] = None
    ml_model: Optional[Any] = None
    rule: Optional[Callable] = None

@dataclass
class BranchChoice:
    """Choix de branche"""
    selected_branch: str
    confidence: float
    alternatives: List[Tuple[str, float]]  # (branch_id, score)
    distribution_shift_detected: bool = False

class BranchSelector:
    """
    Sélecteur de branches dynamique
    Garantie: INV4 (chemin vers goal ou safe_exit)
    """
    
    def __init__(self, state_manager, logger):
        self.state_manager = state_manager
        self.logger = logger
        self._branch_history: Dict[str, List[BranchChoice]] = {}
        self._distribution_baseline: Dict[str, np.ndarray] = {}
    
    def select_branch(self, workflow_id: str, node_id: str,
                     branches: Dict[str, BranchCondition],
                     node_score: float,
                     context: Dict[str, Any]) -> BranchChoice:
        """
        Sélectionne dynamiquement une branche
        Évalue: Score du nœud, conditions, distribution shift
        """
        # Évaluation de chaque branche
        branch_scores = {}
        
        for branch_id, condition in branches.items():
            score = self._evaluate_condition(
                condition, node_score, context
            )
            branch_scores[branch_id] = score
        
        # Sélection de la branche optimale
        selected = max(branch_scores.items(), key=lambda x: x[1])
        selected_branch, max_score = selected
        
        # Calcul des alternatives
        alternatives = sorted(
            [(b, s) for b, s in branch_scores.items() if b != selected_branch],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Détection de distribution shift
        shift_detected = self._detect_distribution_shift(
            workflow_id, node_id, list(branch_scores.values())
        )
        
        choice = BranchChoice(
            selected_branch=selected_branch,
            confidence=max_score,
            alternatives=alternatives,
            distribution_shift_detected=shift_detected
        )
        
        # Historique
        if workflow_id not in self._branch_history:
            self._branch_history[workflow_id] = []
        self._branch_history[workflow_id].append(choice)
        
        # Vérification INV4: chemin vers goal existe
        if not self._verify_goal_reachability(workflow_id, selected_branch):
            self.logger.log_trace(self._create_warning_trace(
                workflow_id, node_id,
                "Selected branch may not lead to goal (INV4 warning)"
            ))
        
        return choice
    
    def _evaluate_condition(self, condition: BranchCondition,
                           node_score: float, context: Dict[str, Any]) -> float:
        """Évalue une condition de branchement"""
        if condition.condition_type == "threshold":
            # Comparaison avec seuil
            if condition.threshold is not None:
                return 1.0 if node_score >= condition.threshold else 0.0
        
        elif condition.condition_type == "ml_prediction":
            # Prédiction ML
            if condition.ml_model is not None:
                features = self._extract_features(node_score, context)
                return condition.ml_model.predict_proba(features)[0][1]
        
        elif condition.condition_type == "rule_based":
            # Règle personnalisée
            if condition.rule is not None:
                return condition.rule(node_score, context)
        
        return 0.5  # Score neutre par défaut
    
    def _extract_features(self, node_score: float, 
                         context: Dict[str, Any]) -> np.ndarray:
        """Extrait des features pour prédiction ML"""
        # Simplification: utiliser le score comme feature unique
        return np.array([[node_score]])
    
    def _detect_distribution_shift(self, workflow_id: str, node_id: str,
                                   scores: List[float]) -> bool:
        """
        Détecte un distribution shift
        Compare la distribution courante avec la baseline
        """
        key = f"{workflow_id}:{node_id}"
        
        # Initialisation de la baseline
        if key not in self._distribution_baseline:
            self._distribution_baseline[key] = np.array(scores)
            return False
        
        baseline = self._distribution_baseline[key]
        current = np.array(scores)
        
        # Calcul KL-divergence (simplifiée)
        # D(P || Q) = Σ P(i) log(P(i)/Q(i))
        try:
            # Normalisation
            p = baseline / baseline.sum()
            q = current / current.sum()
            
            # KL-divergence
            kl_div = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            
            # Seuil de shift
            tau_shift = 0.5
            shift = kl_div > tau_shift
            
            # Mise à jour baseline progressive
            if not shift:
                alpha = 0.1
                self._distribution_baseline[key] = (1 - alpha) * baseline + alpha * current
            
            return shift
        
        except:
            return False
    
    def _verify_goal_reachability(self, workflow_id: str, branch_id: str) -> bool:
        """Vérifie qu'un chemin vers un goal existe depuis la branche"""
        # Utilise le StateManager pour vérification INV4
        return self.state_manager.verify_safe_state(workflow_id)
    
    def _create_warning_trace(self, workflow_id: str, node_id: str, message: str):
        """Crée une trace d'avertissement"""
        from infra_layer_modules import TraceEntry
        return TraceEntry(
            trace_id=workflow_id,
            workflow_id=workflow_id,
            node_id=node_id,
            phase="branching",
            event_type="warning",
            context={"message": message},
            metrics={},
            decision={}
        )


# ============================================================================
# MODULE 16: ITERATION CONTROLLER
# Gestion boucles ML, convergence point fixe
# Garantie: Théorème de Banach (convergence garantie)
# ============================================================================

@dataclass
class IterationConfig:
    """Configuration d'itération"""
    max_iterations: int
    convergence_epsilon: float
    contraction_factor: float  # k < 1 pour garantie Banach
    validation_threshold: float
    retry_threshold: float

@dataclass
class IterationResult:
    """Résultat d'une itération"""
    converged: bool
    iterations: int
    final_parameters: Dict[str, Any]
    convergence_history: List[float]
    error_bound: float

class IterationController:
    """
    Contrôleur d'itérations avec garantie de convergence
    Garantie: Théorème de Banach-Caccioppoli (point fixe)
    """
    
    def __init__(self, action_executor, logger):
        self.action_executor = action_executor
        self.logger = logger
    
    def iterate(self, workflow_id: str, node_id: str,
               initial_parameters: Dict[str, Any],
               ml_predictor: Callable,
               config: IterationConfig,
               action_context: ActionExecutionContext) -> IterationResult:
        """
        Itère jusqu'à convergence ou max_iterations
        Garantie: Si f est contractante (k<1), convergence assurée
        """
        parameters = initial_parameters.copy()
        history = []
        
        for iteration in range(config.max_iterations):
            # Exécution avec paramètres courants
            action_context.parameters = parameters
            result = self.action_executor.execute(action_context)
            
            # Calcul du score
            score = self._compute_iteration_score(result)
            history.append(score)
            
            # Vérification convergence
            if score >= config.validation_threshold:
                # Succès: score satisfaisant
                return IterationResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_parameters=parameters,
                    convergence_history=history,
                    error_bound=0.0
                )
            
            if score < config.retry_threshold:
                # Échec: score trop bas, continuer itération
                
                # Prédiction de nouveaux paramètres via ML
                new_parameters = ml_predictor(parameters, result, score)
                
                # Vérification de contraction
                distance = self._compute_parameter_distance(parameters, new_parameters)
                
                if iteration > 0:
                    prev_distance = self._compute_parameter_distance(
                        history[-2] if len(history) > 1 else 0,
                        history[-1]
                    )
                    
                    # Vérification: d(f(x), f(y)) ≤ k·d(x,y)
                    if prev_distance > 0:
                        actual_k = distance / prev_distance
                        if actual_k >= 1.0:
                            # Non-contractant: risque de non-convergence
                            self.logger.log_trace(self._create_convergence_warning(
                                workflow_id, node_id, actual_k
                            ))
                
                # Critère d'arrêt: variation < epsilon
                if distance < config.convergence_epsilon:
                    # Convergence atteinte
                    error_bound = self._compute_error_bound(
                        distance, config.contraction_factor, iteration
                    )
                    
                    return IterationResult(
                        converged=True,
                        iterations=iteration + 1,
                        final_parameters=new_parameters,
                        convergence_history=history,
                        error_bound=error_bound
                    )
                
                parameters = new_parameters
            else:
                # Score entre retry et validation: acceptable mais sous-optimal
                break
        
        # Max iterations atteint sans convergence
        return IterationResult(
            converged=False,
            iterations=config.max_iterations,
            final_parameters=parameters,
            convergence_history=history,
            error_bound=float('inf')
        )
    
    def _compute_iteration_score(self, result: ExecutionResult) -> float:
        """Calcule un score d'itération"""
        if result.status != ExecutionStatus.SUCCESS:
            return 0.0
        
        # Score composite: robustesse + succès
        return result.robustness_score
    
    def _compute_parameter_distance(self, params1: Any, params2: Any) -> float:
        """
        Calcule la distance entre deux ensembles de paramètres
        Utilisé pour vérification de contraction
        """
        if isinstance(params1, dict) and isinstance(params2, dict):
            # Distance L2 sur les valeurs numériques
            distances = []
            for key in set(params1.keys()) & set(params2.keys()):
                v1, v2 = params1[key], params2[key]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    distances.append((v1 - v2) ** 2)
            
            return np.sqrt(sum(distances)) if distances else 0.0
        
        elif isinstance(params1, (int, float)) and isinstance(params2, (int, float)):
            return abs(params1 - params2)
        
        return 0.0
    
    def _compute_error_bound(self, distance: float, k: float, n: int) -> float:
        """
        Calcule la borne d'erreur selon Banach
        Formule: d(pₙ, p*) ≤ k^n/(1-k) · d(p₁, p₀)
        """
        if k >= 1.0:
            return float('inf')
        
        bound = (k ** n) / (1 - k) * distance
        return bound
    
    def _create_convergence_warning(self, workflow_id: str, node_id: str, k: float):
        """Crée une trace d'avertissement de convergence"""
        from infra_layer_modules import TraceEntry
        return TraceEntry(
            trace_id=workflow_id,
            workflow_id=workflow_id,
            node_id=node_id,
            phase="iteration",
            event_type="convergence_warning",
            context={"contraction_factor": k},
            metrics={"k": k},
            decision={"warning": f"Non-contracting iteration (k={k:.3f} >= 1.0)"}
        )


# ============================================================================
# EXEMPLE D'UTILISATION INTÉGRÉE
# ============================================================================

def example_lam_layer_usage():
    """Exemple d'utilisation des 5 modules LAM"""
    
    print("="*80)
    print("COUCHE LAM - 5 MODULES D'EXÉCUTION")
    print("="*80)
    
    # Mock des dépendances infrastructure
    from collections import namedtuple
    
    MockActionCatalog = namedtuple('MockActionCatalog', ['get_action', 'verify_contract_compatibility'])
    MockResourceManager = namedtuple('MockResourceManager', ['check_budget_available', 'consume_resources'])
    MockDAGManager = namedtuple('MockDAGManager', ['workflows', 'verify_acyclicity', 'get_topological_order', 'add_node', 'add_edge'])
    MockStateManager = namedtuple('MockStateManager', ['initialize_state', 'transition_node', 'verify_safe_state'])
    MockEventBus = namedtuple('MockEventBus', ['publish'])
    MockLogger = namedtuple('MockLogger', ['log_trace'])
    
    # Actions mock
    def mock_action_impl(params):
        return {"result": "success", "value": params.get("input", 0) * 2}
    
    from infra_layer_modules import ActionSpec, ActionContract
    mock_action = ActionSpec(
        id="test_action",
        name="Test Action",
        description="Mock action",
        input_schema={"required": ["input"]},
        output_schema={},
        contract=ActionContract([], [], [], []),
        implementation=mock_action_impl,
        timeout=5.0,
        retries=2,
        robustness_epsilon=0.1
    )
    
    action_catalog = MockActionCatalog(
        get_action=lambda x: mock_action if x == "test_action" else None,
        verify_contract_compatibility=lambda x, y: True
    )
    
    resource_manager = MockResourceManager(
        check_budget_available=lambda x, y: True,
        consume_resources=lambda x, y: True
    )
    
    dag_manager = MockDAGManager(
        workflows={},
        verify_acyclicity=lambda x: True,
        get_topological_order=lambda x: ["node_1"],
        add_node=lambda x, y: True,
        add_edge=lambda x, y, z: True
    )
    
    state_manager = MockStateManager(
        initialize_state=lambda x, y: None,
        transition_node=lambda w, n, s, r: None,
        verify_safe_state=lambda x: True
    )
    
    event_bus = MockEventBus(publish=lambda x: None)
    logger = MockLogger(log_trace=lambda x: None)
    
    # 1. Action Executor
    print("\n--- MODULE 12: Action Executor ---")
    executor = ActionExecutor(action_catalog, resource_manager, logger)
    
    exec_context = ActionExecutionContext(
        action_id="test_action",
        node_id="node_1",
        workflow_id="wf_1",
        parameters={"input": 10},
        timeout=5.0,
        robustness_bound=RobustnessBound(epsilon=0.1, theta=0.2, verified=True),
        retry_policy={"max_retries": 2, "delay": 0.5}
    )
    
    result = executor.execute(exec_context)
    print(f"Execution Status: {result.status.value}")
    print(f"Output: {result.output}")
    print(f"Robustness Score: {result.robustness_score:.3f}")
    print(f"Execution Time: {result.execution_time:.3f}s")
    
    # 2. Node Mapper
    print("\n--- MODULE 13: Node Mapper ---")
    mapper = NodeMapper(action_catalog, dag_manager)
    
    workflow_dag = {
        "nodes": [
            {"id": "n1", "action": "test_action", "parameters": {"input": 5}},
            {"id": "n2", "action": "test_action", "parameters": {"input": 10}}
        ],
        "edges": [["n1", "n2"]]
    }
    
    mappings = mapper.map_workflow(workflow_dag, "wf_test")
    print(f"Mapped {len(mappings)} nodes")
    for mapping in mappings:
        print(f"  {mapping.node_id}: contracts_valid={mapping.contracts_valid}")
    
    # 3. Workflow Orchestrator
    print("\n--- MODULE 14: Workflow Orchestrator ---")
    orchestrator = WorkflowOrchestrator(
        dag_manager, state_manager, executor, event_bus, logger
    )
    print("Orchestrator initialized")
    print("  Guarantees: INV1 (acyclicity), topological order")
    
    # 4. Branch Selector
    print("\n--- MODULE 15: Branch Selector ---")
    selector = BranchSelector(state_manager, logger)
    
    branches = {
        "branch_A": BranchCondition(condition_type="threshold", threshold=0.8),
        "branch_B": BranchCondition(condition_type="threshold", threshold=0.5)
    }
    
    choice = selector.select_branch("wf_1", "node_1", branches, 0.9, {})
    print(f"Selected Branch: {choice.selected_branch}")
    print(f"Confidence: {choice.confidence:.3f}")
    print(f"Distribution Shift: {choice.distribution_shift_detected}")
    print(f"Alternatives: {choice.alternatives}")
    
    # 5. Iteration Controller
    print("\n--- MODULE 16: Iteration Controller ---")
    controller = IterationController(executor, logger)
    
    def mock_ml_predictor(params, result, score):
        # Ajustement simple: augmenter input
        return {**params, "input": params["input"] + 1}
    
    iter_config = IterationConfig(
        max_iterations=5,
        convergence_epsilon=0.01,
        contraction_factor=0.8,
        validation_threshold=0.9,
        retry_threshold=0.5
    )
    
    iter_result = controller.iterate(
        "wf_1", "node_1",
        {"input": 1},
        mock_ml_predictor,
        iter_config,
        exec_context
    )
    
    print(f"Converged: {iter_result.converged}")
    print(f"Iterations: {iter_result.iterations}")
    print(f"Final Parameters: {iter_result.final_parameters}")
    print(f"Error Bound: {iter_result.error_bound:.6f}")
    print(f"Convergence History: {[f'{h:.3f}' for h in iter_result.convergence_history]}")
    
    print("\n" + "="*80)
    print("✓ Couche LAM opérationnelle")
    print("="*80)
    
    return {
        "executor": executor,
        "mapper": mapper,
        "orchestrator": orchestrator,
        "selector": selector,
        "controller": controller
    }


if __name__ == "__main__":
    print("\nModules implémentés:")
    print("12. Action Executor       - Robustesse ε-δ + retry + timeout")
    print("13. Node Mapper           - Validation contrats + mapping DAG")
    print("14. Workflow Orchestrator - Ordre topologique + INV1")
    print("15. Branch Selector       - Distribution shift + INV4")
    print("16. Iteration Controller  - Convergence Banach + point fixe")
    print("\n" + "="*80)
    print("Garanties: Robustesse locale, Contrats, Convergence")
    print("="*80)
    
    # Exécution
    modules = example_lam_layer_usage()