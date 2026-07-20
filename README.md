# Action-Workflow — Blueprint d'Architecture Cible
### Agencement complet pour un Large Action Model robotique

*Document de synthèse architecturale. Consolide la formalisation catégorielle du projet, les 4 papiers de référence inclus dans le dépôt, et les 33 modules déjà implémentés (foundation, LLM, Main-LAM, ML, AutoAdaptation, Backprop, Formal), en un agencement cible unique.*

---

## 0. Principe directeur

Toute décision d'architecture ci-dessous découle d'une seule règle, posée dans *Skills as Morphisms* : **le LLM ne pilote jamais le robot**. Il choisit *quoi* composer ; le RL/IL détermine *comment* le réaliser. Trois catégories, un foncteur, une section :

```
   T  (buts — langage naturel)
   │
   │   R  — foncteur monoïdal : "compilateur d'abstractions comportementales"
   │        (Couche 3 : LLM_layer + synthèse de code type Code-as-Policies
   │         quand aucune primitive existante ne couvre le besoin)
   ▼
   S  (skills — objets = conditions sur l'état partagé, morphismes = policies
   ▲   fermées et terminantes)
   │        (Couche 1+2 : foundation.py + vocabulaire ActionPiece)
   │
   │   p  — forgetful map (le "grounding" : oublie le détail du contrôleur)
   │        (Couche 4 : ActionExecutor)
   │
   C  (contrôle bas niveau — trajectoires continues)
        (grounding continu informé par Coarse-to-Control)

   Γ : T → C, section de p au-dessus de R (p∘Γ = R), construite par RL
       (Couche 5 : ML_layer + Backprop_layer, mémoire procédurale via SkillRL)
```

Tout ajout à l'architecture doit répondre à trois contraintes non négociables déjà posées par le document théorique :
1. **Composition = vérification de contrat** (Def. 3.2) — jamais de branchement A→B sans `Postconditions(A) ⊆ Preconditions(B)`.
2. **Dégradation bornée et explicite** (Prop. 3.3) — toute composition peut perdre en fiabilité, mais cette perte doit être calculée, jamais silencieuse.
3. **Croissance par colimite, jamais par mutation destructive** (§6) — une nouvelle capacité s'ajoute à la chaîne `M0 ↪ M1 ↪ M2 ↪ …` de sous-catégories de S (notée M pour Mémoire, afin de ne pas la confondre avec les couches L0-L9 ci-dessous), elle ne remplace rien qui existait.

---

## 1. Vue synoptique — la pile à 10 couches

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ L9  OBSERVABILITÉ            EventBus · LoggerTracer (mod. 6-7)                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ L8  MÉMOIRE PROCÉDURALE À VIE      colim(Mn) — Ind-objet, SkillBank unifié              │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│ L3              │ L4              │ L5              │ L6              │ L7              │
│ COMPILATEUR     │ ORCHESTRATION   │ APPRENTISSAGE   │ RÉSILIENCE &    │ VÉRIFICATION    │
│ DE CAPACITÉS    │ & EXÉCUTION     │ & GROUNDING     │ ADAPTATION      │ FORMELLE        │
│ LLM_layer 8-11  │ Main-LAM 12-16  │ ML_layer +      │ AutoAdaptation  │ Formal_layer    │
│                 │                 │ Backprop_layer  │ 22-25           │ 30-33           │
│ R : T → S       │                 │ 17-21, 26-29    │                 │ V, U : V→S      │
│                 │                 │ Γ (section)     │                 │                 │
├─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┤
│ L2  VOCABULAIRE D'ACTIONS CONTEXTUEL (ActionPiece)                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ L1  SUBSTRAT CATÉGORIEL / FONDATION           foundation.py (mod. 1-7)                  │
│     catégorie S concrète : objets = états, morphismes = skills+contrats                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ L0  PERCEPTION & WORLD MODEL                  (P0 — état symbolique partagé)            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

Table de lecture rapide :

| Couche | Rôle en une ligne | Statut actuel |
|---|---|---|
| L0 Perception & World Model | Produit l'état symbolique partagé P0 | **Absent** — aucun module dans le code actuel |
| L1 Fondation | Réifie la catégorie S (objets, morphismes, contrats) | **Solide** — 7 modules complets |
| L2 Vocabulaire d'actions | Tokenisation contextuelle des états/skills | **Disponible mais non branché** — `action_piece-main` est un sous-module isolé |
| L3 Compilateur (LLM) | Foncteur R, génération/adaptation/fusion de workflows | **Solide** — 4 modules complets, synthèse de primitives manquante |
| L4 Orchestration & Exécution | Exécute le DAG, branche, itère | **Solide** — 5 modules complets, grounding continu générique |
| L5 Apprentissage & Grounding | Construit la section Γ | **Riche mais fragmenté** — 9 modules, mécanisme fédérateur absent |
| L6 Résilience & Adaptation | Dégradation gracieuse L0→L5, patchs structurels | **Solide** — 4 modules, vérification de patch dupliquée avec L7 |
| L7 Vérification formelle | Catégorie V, foncteur oubli U | **Squelette non fonctionnel** — 4 modules, cœurs de preuve simulés |
| L8 Mémoire procédurale | Colimite filtrée, Ind-objet | **Dispersée** — pas de module fédérateur |
| L9 Observabilité | Traçage, pub/sub | **Solide** — 2 modules, déjà alignés sur les 5 phases du pipeline |

---

## 2. Détail couche par couche

### L0 — Perception & World Model *(à créer)*

**Rôle.** Produire l'état symbolique/latent partagé P0 sur lequel toute la catégorie S est définie (Def. 3.1 de la théorie l'exige explicitement : les objets de S sont des *"conditions symboliques sur l'état du world-model partagé"*). Fusionne vision, proprioception, tactile, et tout autre canal capteur en une représentation unique consommée par L1 et L3.

**Fondement.** §3.1 de la théorie (mentionné mais jamais implémenté) ; *Coarse-to-Control* fournit un patron directement réutilisable — son encodeur visuel-langage produit déjà des tokens compatibles avec un pipeline VLA.

**Ce qu'il manque.** Un module `WorldModelEncoder` exposé comme service à `StateManager` (mod. 5) et `ContextAnalyzer` (mod. 8). C'est le vide le plus structurant du projet : sans lui, `state2feat` — l'entrée dont dépend tout le vocabulaire ActionPiece de L2 — doit être renseigné à la main plutôt que mesuré.

---

### L1 — Substrat catégoriel / Fondation

**Rôle.** Réifier concrètement la catégorie S : le DAG comme structure de composition, le catalogue comme ensemble de morphismes typés.

**Fondement.** §3 (Def. 3.1, 3.2), §4 (structure monoïdale).

**État actuel — solide.** `foundation.py`, modules 1-7 :
- **Mod. 1 DAGManager** — INV1 (acyclicité), détection de cycle DFS avant toute insertion d'arête.
- **Mod. 2 KnowledgeBase** — bornes PAC (`Err_gen ≤ Err_emp + √(VC_dim/n) + O(ln(1/δ)/n)`).
- **Mod. 3 ActionCatalog** — `verify_contract_compatibility` implémente *littéralement* la Def. 3.2 : composition A→B valide ssi `Postconditions(A) ⊆ Preconditions(B)`.
- **Mod. 4 ResourceManager** — INV2 (conservation, transactions atomiques).
- **Mod. 5 StateManager** — INV4 (chemin vers un but ou sortie sûre, détection de deadlock par BFS).
- **Mod. 6-7 EventBus / LoggerTracer** — pub/sub découplé, traces structurées par phase.

**Ce qui complète l'agencement parfait.**
1. Un `IndependenceChecker` : la structure monoïdale (§4.2) suppose l'indépendance de deux skills composés en parallèle, mais rien ne la *vérifie* aujourd'hui — c'est l'un des 5 problèmes ouverts nommés dans la théorie. Une fois L0 disponible, ce module compare les empreintes spatiales/ressources de deux skills candidats à la composition parallèle avant de l'autoriser ; à défaut, retomber sur la catégorie *prémonoïdale* de Power & Robinson évoquée en remarque.
2. Formaliser `ActionContract` au-delà du dict Python informel actuel — voir L7, Assumption 7.2.

---

### L2 — Vocabulaire d'actions contextuel *(disponible, non branché)*

**Rôle.** Transformer les états/skills bruts en un vocabulaire discret et appris qui capture les co-dépendances *intra-état* et *inter-état adjacent* — au lieu d'IDs de skills opaques.

**Fondement.** *ActionPiece* (hou25f, ICML 2025). Le pipeline complet, tel qu'implémenté dans `action_piece-main/genrec/models/ActionPiece/` :
`texte/observation → embedding de phrase (Sentence-T5) → quantization résiduelle-produit (FAISS OPQ+IVF+PQ) → features discrètes par catégorie (state2feat) → ActionPieceCore.train() (fusion BPE pondérée 2/M intra-état, 1/(M₁·M₂) inter-état, via listes chaînées + index inversé + priority queue) → vocabulaire de tokens`.
La segmentation applique la **Set Permutation Regularization** (permutation aléatoire des features avant segmentation BPE) — ce qui produit une invariance à l'ordre non pertinent des features. C'est très exactement la propriété que doit avoir un encodage de skills composés *en parallèle* dans une structure **monoïdale** (§4) : `f ⊗ g` ne doit pas dépendre de l'ordre de sérialisation de f et g. ActionPiece réifie cette propriété par construction, gratuitement.

**Où ça se branche.**
- **Entrée** : `state2feat` = features extraites par L0 (au lieu de `item2feat` = features produit dans le papier d'origine).
- **Sortie vers Mod. 2 (KnowledgeBase)** : chaque `WorkflowPattern` est réexprimé comme séquence de tokens ActionPiece, ce qui transforme `PatternRetriever` (mod. 9) — dont le scoring est aujourd'hui une pondération fixe arbitraire (similarité 40 % + succès 30 % + performance 20 % + usage 10 %) — en recherche par similarité d'**embedding appris**.
- **Sortie vers L8** : le vocabulaire versionné (`ActionPieceCore.save()` / `from_pretrained()`) devient un maillon naturel de la chaîne `Mn ↪ Mn+1`.

**Modèle consommateur.** Le sous-module utilise un T5 encoder-decoder (4+4 couches, d_model=128) en génération auto-régressive avec beam search (50 faisceaux) et ensembling sur 5 segmentations à l'inférence — directement réutilisable comme *tête de prédiction du prochain skill* pour `WorkflowSynthesizer` (mod. 10).

---

### L3 — Compilateur de capacités (LLM)

**Rôle.** Le foncteur monoïdal R : T → S.

**Fondement.** §5.2 de la théorie ; *Code as Policies* pour combler les trous de couverture du catalogue.

**État actuel — solide.** `LLM_layer.py`, modules 8-11, sur Llama (mock si `transformers`/`torch` indisponibles) :
- **Mod. 8 ContextAnalyzer** — extraction JSON du domaine/intention/contraintes, fallback heuristique par mots-clés.
- **Mod. 9 PatternRetriever** — recherche de patterns, suggestions d'adaptation si similarité < 0,85.
- **Mod. 10 WorkflowSynthesizer** — trois stratégies : génération *from scratch*, adaptation d'un pattern unique, ou **fusion de patterns via pullback catégoriel** (le prompt exige explicitement préservation des contraintes + acyclicité — lien direct avec `CompositionalProver.verify_pullback_consistency`, mod. 33).
- **Mod. 11 GlobalReconstructor** — reconstruction complète (niveau d'adaptation 3 sur l'échelle de résilience à 6 niveaux, cf. L6), garantie `Performance(W_adapted) ≥ α · Performance(W_original)` avec α_min = 0,7.

**Ce qui complète l'agencement parfait.** Un **Module 11bis : SkillSynthesizer**, inspiré directement de *Code as Policies* (génération hiérarchique récursive : le LLM écrit une fonction, et toute sous-fonction non définie déclenche une génération récursive du même mécanisme). Aujourd'hui, quand `WorkflowSynthesizer` ne trouve ni pattern ni composition possible, il ne peut que composer des skills *existants* — rien ne crée de nouvelle primitive. Ce module comblerait ce vide : génération de code pour un nouvel `ActionSpec`, obligatoirement soumis à `FormalVerifier` (L7) avant toute entrée dans `ActionCatalog` (mod. 3). C'est le mécanisme qui permettrait au système d'étendre son propre vocabulaire de capacités plutôt que de rester borné au catalogue initial.

---

### L4 — Orchestration & Exécution

**Rôle.** Exécuter concrètement le DAG choisi : ordre topologique, branchement, boucles.

**Fondement.** Exécution de la section Γ ; *Coarse-to-Control* pour le grounding continu.

**État actuel — solide.** `Main-LAM.py`, modules 12-16 :
- **Mod. 12 ActionExecutor** — robustesse ε-δ, budget, timeout, retry, vérification post-conditions.
- **Mod. 13 NodeMapper** — mapping pattern→action, validation de contrats (réutilise mod. 3).
- **Mod. 14 WorkflowOrchestrator** — respect strict de l'ordre topologique (INV1).
- **Mod. 15 BranchSelector** — sélection dynamique (seuil/ML/règle), détection de distribution shift par KL-divergence.
- **Mod. 16 IterationController** — convergence garantie par le théorème du point fixe de Banach, borne d'erreur explicite `d(pₙ,p*) ≤ kⁿ/(1-k)·d(p₁,p₀)`.

**Ce qui complète l'agencement parfait.** `ActionExecutor.implementation` est aujourd'hui un `Callable` Python générique — adapté à un appel API, pas à un contrôleur robotique continu. *Coarse-to-Control* fournit le contrat exact qu'il faut lui donner : je recommande un sous-type `ContinuousControlActionSpec(ActionSpec)` exposant (a) un tokenizer d'observation partagé, (b) une **tête de planification grossière** (tokens de plan internes), (c) une **tête d'exécution** (tokens → commande moteur), les deux têtes partageant un vocabulaire résiduel-VQ commun. Le plan grossier remonterait comme métadonnée dans `ActionExecutionContext`, ce qui permettrait à `BranchSelector`/`IterationController` de monitorer la progression **au niveau du plan**, et non plus seulement à la fin du skill entier — exactement le gain de robustesse sur tâches à horizon long démontré par le papier (97,90 % LIBERO / 83,3 % SimplerEnv-WidowX).

---

### L5 — Apprentissage & Grounding (RL/IL)

**Rôle.** Construire et améliorer la section Γ : les politiques concrètes derrière chaque skill.

**Fondement.** §5.3 (Γ section) ; *SkillRL*, cité en référence [12] de la théorie comme **précédent direct** pour §6.

**État actuel — riche mais fragmenté.** `ML_layer.py` (17-21) + `Backprop_layer.py` (26-29) :
- **Mod. 17 ParameterPredictor** — garantie de point fixe de Banach (mapping contractant, k < 1).
- **Mod. 18 ScoreAggregator** — arithmétique d'intervalles, propagation d'incertitude certifiée.
- **Mod. 19 DistributionMonitor** — détection de dérive (distance de Mahalanobis simplifiée).
- **Mod. 20 CreditAssigner** — remonte l'erreur globale à travers le DAG (BFS inverse), atténuation γ = 0,9, condition de Lyapunov `dV/dt < 0`.
- **Mod. 21 OnlineLearner** — met à jour les poids, vérifie les bornes PAC avant application.
- **Mod. 26 LyapunovErrorCalculator** — V(s) = distance quadratique pondérée à la cible.
- **Mod. 27 FunctorPropagator** — la backprop *littéralement* définie comme foncteur `Cat_Workflow → Cat_Gradients`, garantie `F(W₁∘W₂) = F(W₁)∘F(W₂)`.
- **Mod. 28 SafeUpdateController** — applique les mises à jour sous contrainte budgétaire (INV2).
- **Mod. 29 KnowledgeIntegrator** — transforme l'expérience en connaissance exploitable.

**Ce qui complète l'agencement parfait — précisé par lecture du dépôt de code SkillRL (pas seulement le papier).** Le dépôt révèle un mécanisme plus fin, et surtout plus directement transposable, que la description à haut niveau du papier :

1. **Distillation en deux étages, pas un canal succès/échec unique.** Le code sépare (a) un **raffinement causal amont** — nommé *"Backward Causal Chaining"* dans le prompt source — qui élague une trajectoire brute à ses seules étapes causalement pertinentes puis **abstrait** les détails concrets en templates réutilisables (ex. `"Michael Strahan career"` → `"Search for [Person] career history"`), de (b) **trois appels de distillation séparés** vers un modèle plus fort que l'agent entraîné (Azure o3 dans le dépôt) : un prompt pour les skills généraux (toutes catégories confondues), un prompt par catégorie de tâche, et un troisième prompt dédié uniquement aux **erreurs communes** (`common_mistakes` — un troisième canal à part entière, structuré `{description, why_it_happens, how_to_avoid}`, pas une variante des skills). Pour `KnowledgeIntegrator` (mod. 29) : le raffinement causal doit s'appliquer aux traces déjà phasées de `LoggerTracer` (mod. 7) *avant* toute distillation, et la distillation elle-même doit rester en trois prompts séparés — général / spécifique-par-catégorie / erreurs — plutôt qu'un seul prompt générique qui dilue les trois angles.
2. **SkillBank : un format à trois canaux et surtout deux modes de retrieval interchangeables.** Schéma observé : `general_skills` et `task_specific_skills` (par catégorie) partagent les champs `title`/`principle`/`when_to_apply` ; `common_mistakes` a son propre schéma. Le point le plus directement actionnable : SkillRL expose **un mode "template"** (règles de mots-clés codées en dur, coût nul, aucun modèle requis) et **un mode "embedding"** (Qwen3-Embedding-0.6B, embeddings pré-calculés une fois, similarité cosinus cross-catégorie) comme un simple paramètre de configuration, pas comme un choix figé à la conception. C'est une confirmation directe de la recommandation L2 : `PatternRetriever` (mod. 9) devrait garder son scoring heuristique actuel comme *mode template* (repli rapide, zéro dépendance), et faire du vocabulaire ActionPiece le moteur d'un *mode embedding* activable par configuration — exactement la bascule à deux vitesses que SkillRL a choisi d'exposer plutôt que de trancher.
3. **Évolution récursive : un déclencheur simple et des garde-fous durs, pas une boucle méta complexe.** Mécanisme réel : taux de succès par catégorie comparé à un seuil fixe (0,4 dans les runs réels du dépôt) ; déclenché soit après une passe de validation soit tous les *N* pas d'entraînement (deux modes **mutuellement exclusifs**, précisément pour ne jamais laisser des skills ajustés sur le jeu de validation contaminer les métriques de validation futures — une séparation train/val stricte) ; au plus 10 trajectoires échouées envoyées en un seul appel au modèle enseignant ; et surtout, **les identifiants des nouveaux skills sont systématiquement réassignés côté client après génération**, sans jamais faire confiance au LLM pour l'unicité — un bug réel documenté dans le code (des ID dupliqués `dyn_001, dyn_001…` silencieusement perdus par la déduplication) montre que cette précaution n'est pas gratuite. Pour `AdaptationManager` (mod. 25) et `OnlineLearner` (mod. 21) : déclencheur à seuil fixe par catégorie (déjà traçable via `LoggerTracer`), plafond dur de nouveaux patterns par cycle (`max_new_skills=3` chez SkillRL, pour borner la croissance du catalogue), et réassignation d'identifiants garantie côté système — jamais côté LLM.

---

### L6 — Résilience & Adaptation structurelle

**Rôle.** Dégrader gracieusement, patcher localement sans reconstruction totale.

**Fondement.** Colimite filtrée (§6 — chaque patch est un nouvel objet dans la chaîne) ; opérations de pushout.

**État actuel — solide dans la structure.** `AutoAdaptation_layer.py`, modules 22-25 :
- **Mod. 22 ResilienceController** — escalade stricte à 6 niveaux d'adaptation (notés 0 à 5 pour ne pas les confondre avec les couches L0-L9 de ce document) : niveau 0 nominal → niveau 1 paramétrique (ML) → niveau 2 patch structurel local (pushout) → niveau 3 reconstruction globale (LLM, mod. 11) → niveau 4 fallback statique → niveau 5 isolation/arrêt d'urgence.
- **Mod. 23 StructuralPatcher** — opération de pushout `A +_B C` pour remplacer un sous-graphe défaillant, vérification de type Liskov local (`I(New) ⊇ I(Old)`, `O(New) ⊆ O(Old)`).
- **Mod. 24 ParametricAdapter** — niveau L1, enveloppe de sécurité dure (timeout ≤ 60 s, retries ≤ 5, confidence ≥ 0,5).
- **Mod. 25 AdaptationManager** — orchestrateur, garantie α = 0,7.

**Ce qui complète l'agencement parfait.** `StructuralPatcher._verify_interface_compatibility` et `CompositionalProver.verify_pushout_coherence` (mod. 33, L7) implémentent **la même opération catégorielle deux fois, sans être connectées**. Il faut faire de `CompositionalProver` le point de vérité unique pour toute opération de pullback/pushout, appelé à la fois par `StructuralPatcher` (ici) et `WorkflowSynthesizer` (L3, mod. 10) — plutôt que deux implémentations parallèles qui peuvent diverger silencieusement.

---

### L7 — Vérification formelle & Sécurité

**Rôle.** La catégorie V des skills certifiés, foncteur oubli U : V → S — "garanties de sécurité" = U monoïdal.

**Fondement.** §7 (Assumption 7.2, Proposition 7.3).

**État actuel — squelette architectural complet, cœurs non fonctionnels.** `Formal_layer.py`, modules 30-33 :
- **Mod. 30 FormalVerifier** — vérifie des formules LTL sur un `LAM_LTS` construit depuis le DAG ; `_verify_safety` retourne aujourd'hui `True` avec un hash factice.
- **Mod. 31 RuntimeMonitor** — surveille INV1-INV4 en continu.
- **Mod. 32 RobustnessCertifier** — estimation de constante de Lipschitz, score minimax ; facteurs actuellement codés en dur.
- **Mod. 33 CompositionalProver** — `verify_pullback_consistency` / `verify_pushout_coherence`, le lien direct avec la théorie.

**Ce qui complète l'agencement parfait — le chantier le plus important du blueprint.**
1. Remplacer le model-checker LTL simulé par un moteur réel (bounded model checking sur le DAG concret, a minima, avant d'envisager une intégration type Spin/nuXmv adaptée au `LAM_LTS`).
2. **Opérationnaliser l'Assumption 7.2** (le "langage de spécification" reste un problème ouvert explicite de la théorie, §8.3) : partir des `ActionContract` déjà présents (pré/post/invariants) et les exprimer dans un sous-ensemble décidable de LTL propositionnelle sur les prédicats de `state2feat` — ce qui connecte directement le vocabulaire ActionPiece (L2) aux formules vérifiables.
3. `RobustnessCertifier` doit consommer les bornes ε-δ **réellement mesurées** par `ActionExecutor` (L4) et les taux de succès empiriques par type de tâche que *Coarse-to-Control* documente, plutôt que des facteurs fixes.

---

### L8 — Mémoire procédurale à vie *(transversale, dispersée)*

**Rôle.** M∞ = colim(M0 ↪ M1 ↪ M2 ↪ …), l'Ind-objet de Grothendieck (§6) — chaîne de sous-catégories de S, notée M pour ne pas la confondre avec les couches L0-L9.

**Fondement.** §6 ; *SkillRL* comme précédent direct (SkillBank évolutif).

**État actuel.** Dispersée entre `KnowledgeBase` (mod. 2), le vocabulaire ActionPiece versionné (L2), chaque patch de `StructuralPatcher` (L6), et `OnlineLearner` (mod. 21) — sans fédérateur.

**Ce qui complète l'agencement parfait.** Un `ProceduralMemoryManager` qui versionne explicitement chaque inclusion `Mn ↪ Mn+1` (nouveau pattern, patch de pushout, extension de vocabulaire) avec une opération unique `colimit_snapshot()` exposée à tous les modules. C'est la seule façon de transformer la propriété universelle de la colimite — *toute capacité valide au temps t reste accessible à t' > t* — d'une métaphore en un invariant réellement vérifié.

*Précision tirée du dépôt SkillRL* : leur implémentation du versionnage est volontairement minimale et directement copiable comme point de départ — chaque mise à jour du skill bank est écrite comme un **snapshot JSON complet et autonome**, nommé par le pas d'entraînement global (`updated_skills_step{N}.json`), jamais comme une mutation en place. Ce n'est pas encore une vraie colimite (chaque snapshot duplique tout le contenu au lieu d'encoder seulement l'inclusion/le delta), mais c'est une base saine : `ProceduralMemoryManager.colimit_snapshot()` peut commencer exactement ainsi (un fichier horodaté par mise à jour), et n'ajouter le graphe explicite des inclusions `Mn ↪ Mn+1` que dans un second temps, une fois le besoin de requêter "qu'est-ce qui a changé entre deux versions" avéré.

---

### L9 — Observabilité *(transversale, déjà cohérente)*

`EventBus` + `LoggerTracer` (mod. 6-7). Point positif à noter : les 5 phases déjà tracées (`construction / execution / validation / adaptation / backprop`) correspondent terme à terme à L3/L4/L7/L6/L5 ci-dessus — la seule couche du projet actuel dont l'agencement est déjà celui qu'on retrouverait dans la version cible.

---

## 3. Table de correspondance — papier/théorie → couche

| Source | Contribution clé | Couche(s) cible |
|---|---|---|
| *Skills as Morphisms* (théorie propre) | Ossature catégorielle entière : S, T, C, R, Γ, colimite, U monoïdal | Toutes — c'est le plan directeur |
| *ActionPiece* (hou25f) | Tokenisation contextuelle par fusion pondérée intra/inter-état + Set Permutation Regularization | **L2** (nouvelle couche), irrigue L3 (retrieval) et L8 (versionnage du vocabulaire) |
| *SkillRL* (2602.08234) | Raffinement causal + distillation à 3 prompts (général/spécifique/erreurs), SkillBank à 2 modes de retrieval (template/embedding), évolution récursive à seuil fixe et ID réassignés côté client, via GRPO | **L5** (mécanisme fédérateur, précisé par lecture du dépôt), **L8** (versionnage par snapshot JSON) |
| *Coarse-to-Control* (2606.07107) | Planification grossière + exécution dans un vocabulaire d'action partagé (residual-VQ) | **L4** (contrat `ContinuousControlActionSpec`), **L0** (encodeur visuel-langage) |
| *Code as Policies* (2209.07753) | Génération hiérarchique récursive de code pour des primitives non définies | **L3** (Module 11bis, synthèse de nouvelles primitives) |

---

## 4. Flux d'exécution de bout en bout — scénario concret

*"Apporte-moi la tasse rouge qui est sur la table de la cuisine."*

1. **L0** — le World Model fusionne le flux caméra + proprioception → état P0 (position robot, objets détectés avec features discrètes : couleur, catégorie, localisation).
2. **L3, mod. 8** — `ContextAnalyzer` extrait intention = `fetch_object`, entités = `{objet: tasse, couleur: rouge, lieu: cuisine}`.
3. **L2** — ces features sont encodées par `ActionPieceCore` en tokens ; **L3, mod. 9** — `PatternRetriever` cherche un `WorkflowPattern` proche dans le SkillBank (L5/L8).
4. **L3, mod. 10** — `WorkflowSynthesizer` : un pattern proche existe mais incomplet → adaptation (pas fusion) ; DAG produit : `naviguer_cuisine → localiser_tasse_rouge → saisir → naviguer_retour → poser`.
5. **L1, mod. 3** — `ActionCatalog.verify_contract_compatibility` valide chaque transition (ex. postconditions de `saisir` ⊆ préconditions de `naviguer_retour` : "objet en main").
6. **L4, mod. 13-14** — `NodeMapper` mappe chaque nœud à un `ActionSpec` concret ; `WorkflowOrchestrator` exécute dans l'ordre topologique.
7. **L4 (contrat Coarse-to-Control)** — `saisir` est un `ContinuousControlActionSpec` : tête de plan grossier (trajectoire d'approche prévue) puis tokens d'exécution moteur ; **L4, mod. 15** — `BranchSelector` surveille la progression du plan, pas seulement le résultat final.
8. **Échec partiel** (la tasse a glissé) → **L6, mod. 22** — `ResilienceController` escalade : niveau 1 (nouveaux paramètres de préhension, mod. 17) insuffisant → niveau 2 (`StructuralPatcher` propose un pushout : réessai avec approche latérale) → succès.
9. **L5** — `CreditAssigner` (mod. 20) remonte le score de la tâche à travers le DAG. Si le taux de succès de la catégorie `saisir_objet_glissant` tombe sous le seuil configuré sur une fenêtre de validation, le **mécanisme d'évolution récursive (L5, inspiré de SkillRL)** regroupe les derniers épisodes échoués de cette catégorie et les envoie en un seul appel à un modèle enseignant distinct de l'agent, qui propose un skill ciblé (préhension latérale) — ajouté au SkillBank avec un identifiant réassigné côté système, jamais laissé au LLM.
10. **L7** — `RuntimeMonitor` confirme INV1-INV4 respectés tout du long ; **L8** — le patch de pushout et le pattern raffiné entrent dans la colimite `Mn ↪ Mn+1` — la prochaine tasse glissante sera saisie correctement dès la première tentative.

---

## 5. Priorités transversales

Deux vides structurels conditionnent tout le reste : **L0** (rien n'existe sans état perçu) et **L2** (le vocabulaire irrigue le retrieval de L3, la structure de L5, et le versionnage de L8). Les combler débloque mécaniquement la qualité de tout le reste de la pile. Le chantier L7 (rendre la vérification formelle réellement fonctionnelle, pas seulement squelettique) conditionne quant à lui la validité de *toutes* les garanties déclarées ailleurs — c'est actuellement le seul endroit où l'agencement théorique et l'implémentation divergent sur le fond, pas seulement sur la forme.

---

*Document construit à partir de la lecture intégrale de : Action-Workflow_Categorical-Formalization.pdf, hou25f.pdf (ActionPiece), 2602.08234v1.pdf (SkillRL), 2606.07107v1.pdf (Coarse-to-Control), 2209.07753v4.pdf (Code as Policies), des 7 fichiers Python + sous-module action_piece-main du dépôt Action-Workflow, et — pour les sections L5/L8 — du dépôt de code SkillRL-main (agent_system/memory/, skill_generation/, verl/trainer/ppo/ray_trainer.py), qui précise et parfois nuance le papier sur les mécanismes de distillation, de retrieval et d'évolution récursive.*
