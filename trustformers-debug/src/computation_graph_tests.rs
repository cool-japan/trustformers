//! Tests for the computation_graph module.

use super::*;

#[test]
fn test_computation_graph_creation() {
    let mut analyzer = ComputationGraphAnalyzer::default();

    let operations = vec![
        (
            "input".to_string(),
            OperationType::Custom("Input".to_string()),
            vec![],
        ),
        (
            "linear1".to_string(),
            OperationType::MatMul,
            vec!["input".to_string()],
        ),
        (
            "relu1".to_string(),
            OperationType::ReLU,
            vec!["linear1".to_string()],
        ),
        (
            "linear2".to_string(),
            OperationType::MatMul,
            vec!["relu1".to_string()],
        ),
        (
            "output".to_string(),
            OperationType::Custom("Output".to_string()),
            vec!["linear2".to_string()],
        ),
    ];

    let graph_id = analyzer
        .create_graph("test_model".to_string(), operations)
        .expect("operation failed in test");
    let analysis = analyzer.analyze_graph(graph_id).expect("operation failed in test");

    assert_eq!(analysis.statistics.nodes_by_type.len(), 4); // MatMul, ReLU, Custom("Input"), Custom("Output")
    assert!(!analysis.critical_path.is_empty());
}

#[test]
fn test_optimization_detection() {
    let mut analyzer = ComputationGraphAnalyzer::default();

    let operations = vec![
        (
            "input".to_string(),
            OperationType::Custom("Input".to_string()),
            vec![],
        ),
        (
            "matmul".to_string(),
            OperationType::MatMul,
            vec!["input".to_string()],
        ),
        (
            "add".to_string(),
            OperationType::Add,
            vec!["matmul".to_string()],
        ),
    ];

    let graph_id = analyzer
        .create_graph("fusion_test".to_string(), operations)
        .expect("operation failed in test");
    let analysis = analyzer.analyze_graph(graph_id).expect("operation failed in test");

    assert!(analysis
        .optimization_opportunities
        .iter()
        .any(|op| op.optimization_type == OptimizationType::OperationFusion));
}

#[test]
fn test_dot_export() {
    let mut analyzer = ComputationGraphAnalyzer::default();

    let operations = vec![
        ("a".to_string(), OperationType::MatMul, vec![]),
        ("b".to_string(), OperationType::ReLU, vec!["a".to_string()]),
    ];

    let graph_id = analyzer
        .create_graph("simple".to_string(), operations)
        .expect("operation failed in test");
    let dot = analyzer.export_to_dot(graph_id).expect("operation failed in test");

    assert!(dot.contains("digraph"));
    assert!(dot.contains("MatMul"));
    assert!(dot.contains("ReLU"));
}

// ── Honesty regressions: peak memory, fragmentation, complexity,
//    parallelization potential, redundancy detection, critical path,
//    and clustering coefficient must all be REAL, not the old
//    fabricated constants. ──

fn set_memory(analyzer: &mut ComputationGraphAnalyzer, graph_id: Uuid, node_id: &str, bytes: u64) {
    analyzer
        .graphs
        .get_mut(&graph_id)
        .unwrap()
        .nodes
        .get_mut(node_id)
        .unwrap()
        .memory_usage = Some(bytes);
}

fn set_flops(analyzer: &mut ComputationGraphAnalyzer, graph_id: Uuid, node_id: &str, flops: u64) {
    analyzer
        .graphs
        .get_mut(&graph_id)
        .unwrap()
        .nodes
        .get_mut(node_id)
        .unwrap()
        .flop_count = Some(flops);
}

fn set_exec_time(analyzer: &mut ComputationGraphAnalyzer, graph_id: Uuid, node_id: &str, us: u64) {
    analyzer
        .graphs
        .get_mut(&graph_id)
        .unwrap()
        .nodes
        .get_mut(node_id)
        .unwrap()
        .execution_time_us = Some(us);
}

#[test]
fn test_peak_memory_usage_below_total_for_a_linear_chain() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        (
            "input".to_string(),
            OperationType::Custom("X".to_string()),
            vec![],
        ),
        (
            "a".to_string(),
            OperationType::Custom("X".to_string()),
            vec!["input".to_string()],
        ),
        (
            "b".to_string(),
            OperationType::Custom("X".to_string()),
            vec!["a".to_string()],
        ),
        (
            "c".to_string(),
            OperationType::Custom("X".to_string()),
            vec!["b".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("chain".to_string(), operations).expect("create");
    set_memory(&mut analyzer, graph_id, "input", 100);
    set_memory(&mut analyzer, graph_id, "a", 200);
    set_memory(&mut analyzer, graph_id, "b", 300);
    set_memory(&mut analyzer, graph_id, "c", 400);

    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let memory = analysis.memory_analysis.expect("memory analysis enabled");

    assert_eq!(memory.total_memory_usage, 1000);
    // Hand-computed liveness walk: peak occurs at c's step, with only
    // b (c's one live dependency) and c itself resident -- 300 + 400.
    assert_eq!(
        memory.peak_memory_usage, 700,
        "peak must reflect real liveness (each node's predecessor freed after its only \
         consumer runs), not the old `peak == total` placeholder"
    );
    assert!(memory.peak_memory_usage < memory.total_memory_usage);
}

#[test]
fn test_peak_memory_usage_diamond_keeps_both_branches_live() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        (
            "a".to_string(),
            OperationType::Custom("X".to_string()),
            vec![],
        ),
        (
            "b".to_string(),
            OperationType::Custom("X".to_string()),
            vec!["a".to_string()],
        ),
        (
            "c".to_string(),
            OperationType::Custom("X".to_string()),
            vec!["a".to_string()],
        ),
        (
            "d".to_string(),
            OperationType::Custom("X".to_string()),
            vec!["b".to_string(), "c".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("diamond".to_string(), operations).expect("create");
    set_memory(&mut analyzer, graph_id, "a", 10);
    set_memory(&mut analyzer, graph_id, "b", 20);
    set_memory(&mut analyzer, graph_id, "c", 30);
    set_memory(&mut analyzer, graph_id, "d", 5);

    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let memory = analysis.memory_analysis.expect("memory analysis enabled");

    // Peak occurs once both b and c (a's two independent consumers)
    // have been produced but before d consumes them: a (only freed
    // once BOTH b and c have run) + b + c = 10 + 20 + 30 = 60.
    assert_eq!(
        memory.peak_memory_usage, 60,
        "a diamond must keep both branches (and their shared ancestor, until its LAST \
         consumer runs) live simultaneously"
    );
    assert_eq!(memory.total_memory_usage, 65);
}

#[test]
fn test_peak_memory_usage_preserves_leaf_nodes_even_as_a_dependency() {
    // Manually constructed (not via `create_graph`): node "a" is
    // marked as a leaf/output-of-interest even though "b" also
    // consumes it -- a legitimate shape for a multi-output graph
    // where an intermediate value is *also* a published output that
    // must stay resident. `compute_peak_memory_usage` must never free
    // a `leaf_nodes` member, even when this loop identifies it as
    // some other node's "last use".
    let mut analyzer = ComputationGraphAnalyzer::default();
    let mut nodes = HashMap::new();
    for (id, mem, depth, topo) in [
        ("a", 50u64, 0usize, 0usize),
        ("b", 10, 1, 1),
        ("c", 1000, 2, 2),
    ] {
        nodes.insert(
            id.to_string(),
            GraphNode {
                id: id.to_string(),
                name: id.to_string(),
                operation_type: OperationType::Custom("X".to_string()),
                input_shapes: vec![],
                output_shapes: vec![],
                flop_count: Some(0),
                memory_usage: Some(mem),
                execution_time_us: None,
                parameter_count: None,
                topo_order: Some(topo),
                depth,
                metadata: HashMap::new(),
            },
        );
    }
    let mut edges = HashMap::new();
    edges.insert("a".to_string(), vec![]);
    edges.insert("b".to_string(), vec!["a".to_string()]);
    edges.insert("c".to_string(), vec!["b".to_string()]);
    let graph = ComputationGraph {
        id: Uuid::new_v4(),
        nodes,
        edges,
        root_nodes: ["a".to_string()].into_iter().collect(),
        leaf_nodes: ["a".to_string(), "c".to_string()].into_iter().collect(),
        metadata: GraphMetadata {
            name: "leaf_preserved".to_string(),
            node_count: 3,
            edge_count: 2,
            max_depth: 2,
            estimated_memory_usage: 1060,
            estimated_flops: 0,
            created_at: chrono::Utc::now(),
        },
    };
    let graph_id = graph.id;
    analyzer.add_graph(graph).expect("add_graph");

    let peak = analyzer.compute_peak_memory_usage(analyzer.graphs.get(&graph_id).unwrap());
    assert_eq!(
        peak, 1060,
        "a leaf_nodes member (a=50) must stay live through c's step (50+10+1000) even \
         though b was its last real consumer"
    );
}

#[test]
fn test_fragmentation_ratio_is_honestly_none() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let graph_id = analyzer
        .create_graph(
            "g".to_string(),
            vec![("a".to_string(), OperationType::Add, vec![])],
        )
        .expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    assert_eq!(
        analysis.memory_analysis.expect("enabled").fragmentation_ratio,
        None,
        "no allocator/placement model exists -- must be an honest None, never a fabricated \
         ratio"
    );
}

#[test]
fn test_complexity_time_and_space_are_honestly_none() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let graph_id = analyzer
        .create_graph(
            "g".to_string(),
            vec![("a".to_string(), OperationType::Add, vec![])],
        )
        .expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let complexity = analysis.flop_analysis.expect("enabled").complexity_analysis;
    assert_eq!(complexity.time_complexity, None);
    assert_eq!(complexity.space_complexity, None);
}

#[test]
fn test_parallelization_potential_is_zero_for_a_pure_chain() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("a".to_string(), OperationType::Add, vec![]),
        ("b".to_string(), OperationType::Add, vec!["a".to_string()]),
        ("c".to_string(), OperationType::Add, vec!["b".to_string()]),
        ("d".to_string(), OperationType::Add, vec!["c".to_string()]),
    ];
    let graph_id = analyzer.create_graph("chain".to_string(), operations).expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let potential = analysis
        .flop_analysis
        .expect("enabled")
        .complexity_analysis
        .parallelization_potential;
    assert_eq!(
        potential, 0.0,
        "every node is on the critical path in a pure chain -- there is nothing left to \
         run in parallel"
    );
}

#[test]
fn test_parallelization_potential_is_high_for_independent_branches_into_one_sink() {
    // The degenerate case that would break a FLOP/time-weighted
    // formula if the sink happens to dominate cost: 5 independent
    // roots feeding a single sink. Structurally this is highly
    // parallel (5 branches can run at once) regardless of where the
    // FLOPs concentrate, so the (deliberately unweighted) metric must
    // report that, not collapse to ~0 because of one expensive node.
    let mut analyzer = ComputationGraphAnalyzer::default();
    let mut operations: Vec<(String, OperationType, Vec<String>)> =
        (0..5).map(|i| (format!("root{i}"), OperationType::Add, vec![])).collect();
    operations.push((
        "sink".to_string(),
        OperationType::Add,
        (0..5).map(|i| format!("root{i}")).collect(),
    ));
    let graph_id = analyzer.create_graph("wide".to_string(), operations).expect("create");
    // Make the sink dominate FLOPs, to prove this metric is NOT
    // FLOP-weighted (a work/span version would misreport this case).
    for i in 0..5 {
        set_flops(&mut analyzer, graph_id, &format!("root{i}"), 1);
    }
    set_flops(&mut analyzer, graph_id, "sink", 1_000_000);

    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let potential = analysis
        .flop_analysis
        .expect("enabled")
        .complexity_analysis
        .parallelization_potential;
    // node_count=6, span=max_depth+1=2 -> 1 - 2/6 = 0.6667
    assert!(
        potential > 0.5,
        "5 independent branches into 1 sink is structurally highly parallel, got {}",
        potential
    );
    assert!((potential - (1.0 - 2.0 / 6.0)).abs() < 1e-9);
}

#[test]
fn test_redundancy_detection_flags_identical_recomputation() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        (
            "input".to_string(),
            OperationType::Custom("Input".to_string()),
            vec![],
        ),
        (
            "a".to_string(),
            OperationType::MatMul,
            vec!["input".to_string()],
        ),
        (
            "b".to_string(),
            OperationType::MatMul,
            vec!["input".to_string()],
        ), // duplicate of a
        (
            "c".to_string(),
            OperationType::Add,
            vec!["input".to_string()],
        ), // different op: not a duplicate
    ];
    let graph_id = analyzer.create_graph("cse".to_string(), operations).expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");

    let redundant: Vec<_> = analysis
        .optimization_opportunities
        .iter()
        .filter(|op| op.optimization_type == OptimizationType::RedundancyElimination)
        .collect();
    assert_eq!(redundant.len(), 1, "exactly one redundant group: {{a, b}}");
    assert_eq!(
        redundant[0].affected_nodes,
        vec!["a".to_string(), "b".to_string()]
    );
}

#[test]
fn test_redundancy_detection_does_not_flag_distinct_roots_or_distinct_inputs() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        // Two roots, same op type, both empty dependency lists -- must
        // NOT be flagged: an empty dep list proves nothing about two
        // distinct external inputs being the same value.
        (
            "input1".to_string(),
            OperationType::Custom("Input".to_string()),
            vec![],
        ),
        (
            "input2".to_string(),
            OperationType::Custom("Input".to_string()),
            vec![],
        ),
        // Same op type, but different (single) dependency -- also not
        // a duplicate.
        (
            "a".to_string(),
            OperationType::MatMul,
            vec!["input1".to_string()],
        ),
        (
            "b".to_string(),
            OperationType::MatMul,
            vec!["input2".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("no_cse".to_string(), operations).expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");

    assert!(analysis
        .optimization_opportunities
        .iter()
        .all(|op| op.optimization_type != OptimizationType::RedundancyElimination));
}

#[test]
fn test_critical_path_follows_real_flops_when_unprofiled() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("root".to_string(), OperationType::Add, vec![]),
        (
            "cheap".to_string(),
            OperationType::Add,
            vec!["root".to_string()],
        ),
        (
            "expensive".to_string(),
            OperationType::Add,
            vec!["root".to_string()],
        ),
        (
            "sink".to_string(),
            OperationType::Add,
            vec!["cheap".to_string(), "expensive".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("branch".to_string(), operations).expect("create");
    set_flops(&mut analyzer, graph_id, "root", 1);
    set_flops(&mut analyzer, graph_id, "cheap", 10);
    set_flops(&mut analyzer, graph_id, "expensive", 1000);
    set_flops(&mut analyzer, graph_id, "sink", 5);

    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    assert_eq!(
        analysis.critical_path,
        vec![
            "root".to_string(),
            "expensive".to_string(),
            "sink".to_string()
        ],
        "the real higher-FLOP branch must be the critical path, not depth (both branches \
         have equal depth) and not the cheap branch"
    );
}

#[test]
fn test_critical_path_prefers_real_execution_time_over_flops_when_profiled() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("root".to_string(), OperationType::Add, vec![]),
        (
            "cheap".to_string(),
            OperationType::Add,
            vec!["root".to_string()],
        ),
        (
            "expensive".to_string(),
            OperationType::Add,
            vec!["root".to_string()],
        ),
        (
            "sink".to_string(),
            OperationType::Add,
            vec!["cheap".to_string(), "expensive".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("branch2".to_string(), operations).expect("create");
    // Same FLOP shape as the previous test (expensive >> cheap)...
    set_flops(&mut analyzer, graph_id, "root", 1);
    set_flops(&mut analyzer, graph_id, "cheap", 10);
    set_flops(&mut analyzer, graph_id, "expensive", 1_000_000);
    set_flops(&mut analyzer, graph_id, "sink", 5);
    // ...but real profiling says the OPPOSITE: "cheap" is actually
    // slow in wall-clock time, and "expensive" was never profiled.
    // Once ANY node is profiled, real time must win for every node on
    // the SAME scale (unprofiled nodes contribute 0, never their
    // flop_count).
    set_exec_time(&mut analyzer, graph_id, "root", 1);
    set_exec_time(&mut analyzer, graph_id, "cheap", 1000);
    set_exec_time(&mut analyzer, graph_id, "sink", 5);

    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    assert_eq!(
        analysis.critical_path,
        vec!["root".to_string(), "cheap".to_string(), "sink".to_string()],
        "real execution_time_us must win over flop_count once the graph is profiled, even \
         though 'expensive' has vastly more FLOPs"
    );
}

#[test]
fn test_clustering_coefficient_zero_for_a_chain() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("a".to_string(), OperationType::Add, vec![]),
        ("b".to_string(), OperationType::Add, vec!["a".to_string()]),
        ("c".to_string(), OperationType::Add, vec!["b".to_string()]),
    ];
    let graph_id = analyzer.create_graph("chain3".to_string(), operations).expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    assert_eq!(analysis.statistics.clustering_coefficient, 0.0);
}

#[test]
fn test_clustering_coefficient_nonzero_for_a_residual_connection() {
    // x -> f(x), then Add(x, f(x)): a real triangle once treated as
    // undirected (x-f, f-add, and x-add via the direct skip
    // connection) -- exactly the residual/skip-connection pattern
    // common in transformer graphs. Disproves the old "DAGs always
    // have clustering coefficient 0" comment.
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("x".to_string(), OperationType::Add, vec![]),
        ("f".to_string(), OperationType::ReLU, vec!["x".to_string()]),
        (
            "add".to_string(),
            OperationType::Add,
            vec!["x".to_string(), "f".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("residual".to_string(), operations).expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    assert!(
        (analysis.statistics.clustering_coefficient - 1.0).abs() < 1e-9,
        "x, f and add form a complete triangle when undirected -- clustering coefficient \
         must be 1.0, not the old hardcoded 0.0, got {}",
        analysis.statistics.clustering_coefficient
    );
}

#[test]
fn test_memory_reuse_opportunity_flags_real_non_overlapping_variables() {
    // a -> b -> d, a -> c -> d: b and c are DISJOINT branches off a
    // common ancestor with no dependency on each other, so their
    // lifetimes as computed (b: born after a, dies at d; c: born
    // after a, dies at d) actually DO overlap in this topology. Use
    // a genuinely sequential shape instead: a -> b -> c -> d, where
    // a's real lifetime is [birth=a, death=b] and c's is
    // [birth=c, death=d] -- strictly non-overlapping (a dies before
    // c is even born), a real, safe reuse candidate.
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("a".to_string(), OperationType::Add, vec![]),
        ("b".to_string(), OperationType::Add, vec!["a".to_string()]),
        ("c".to_string(), OperationType::Add, vec!["b".to_string()]),
        ("d".to_string(), OperationType::Add, vec!["c".to_string()]),
    ];
    let graph_id = analyzer.create_graph("seq".to_string(), operations).expect("create");
    set_memory(&mut analyzer, graph_id, "a", 100);
    set_memory(&mut analyzer, graph_id, "b", 999); // never a candidate: it's b's own consumer of a, but b itself is consumed only by c
    set_memory(&mut analyzer, graph_id, "c", 40);

    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let dataflow = analysis.dataflow_analysis.expect("dataflow analysis enabled");

    // a: born at topo(a)=0, dies at topo(b)=1 (its only consumer).
    // c: born at topo(c)=2, dies at topo(d)=3 (its only consumer).
    // a's death (1) < c's birth (2) -- strictly non-overlapping.
    let found = dataflow
        .memory_reuse_opportunities
        .iter()
        .find(|o| o.reusable_variables == vec!["a".to_string(), "c".to_string()]);
    let opportunity =
        found.expect("a (dies before c is born) and c must be flagged as a real reuse pair");
    assert_eq!(
        opportunity.memory_savings, 40,
        "savings must be the real min(footprint_a=100, footprint_c=40), not a fabricated 1MB"
    );
    assert_eq!(opportunity.complexity, 2);
}

#[test]
fn test_memory_reuse_opportunity_excludes_overlapping_lifetimes() {
    // a's two consumers (b, c) both need a alive simultaneously --
    // NOT a safe reuse candidate with each other via a, and b/c
    // themselves are both alive from their own birth until d
    // consumes them, so b and c overlap with EACH OTHER too.
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("a".to_string(), OperationType::Add, vec![]),
        ("b".to_string(), OperationType::Add, vec!["a".to_string()]),
        ("c".to_string(), OperationType::Add, vec!["a".to_string()]),
        (
            "d".to_string(),
            OperationType::Add,
            vec!["b".to_string(), "c".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("diamond2".to_string(), operations).expect("create");
    set_memory(&mut analyzer, graph_id, "b", 10);
    set_memory(&mut analyzer, graph_id, "c", 20);

    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let dataflow = analysis.dataflow_analysis.expect("dataflow analysis enabled");

    assert!(
        dataflow
            .memory_reuse_opportunities
            .iter()
            .all(|o| o.reusable_variables != vec!["b".to_string(), "c".to_string()]),
        "b and c are both alive simultaneously (from their own birth until d consumes \
         both) -- must NOT be flagged as a reuse pair"
    );
}

#[test]
fn test_memory_reuse_opportunity_never_includes_leaf_nodes() {
    // A real falsification, not a vacuous pass: manually construct
    // (bypassing `create_graph`'s automatic leaf detection, the same
    // technique `test_peak_memory_usage_preserves_leaf_nodes_...`
    // uses) a "multi-output" graph where node "a" is BOTH a real
    // leaf/published-output AND has a real internal consumer "b" --
    // so a's natural (non-leaf) death time would be topo(b)=1, which
    // ends strictly before "c" (topo=2, consumed by "d" at topo=3) is
    // even born. If the leaf-exclusion filter in
    // `find_memory_reuse_opportunities` were missing or broken, "a"
    // and "c" WOULD be reported as a valid non-overlapping reuse
    // pair by the very same lifetime math this test suite already
    // verified as correct elsewhere -- proving this test can fail.
    let mut analyzer = ComputationGraphAnalyzer::default();
    let mut nodes = HashMap::new();
    for (id, mem, topo) in [("a", 50u64, 0usize), ("b", 5, 1), ("c", 50, 2), ("d", 5, 3)] {
        nodes.insert(
            id.to_string(),
            GraphNode {
                id: id.to_string(),
                name: id.to_string(),
                operation_type: OperationType::Custom("X".to_string()),
                input_shapes: vec![],
                output_shapes: vec![],
                flop_count: Some(0),
                memory_usage: Some(mem),
                execution_time_us: None,
                parameter_count: None,
                topo_order: Some(topo),
                depth: topo,
                metadata: HashMap::new(),
            },
        );
    }
    let mut edges = HashMap::new();
    edges.insert("a".to_string(), vec![]);
    edges.insert("b".to_string(), vec!["a".to_string()]);
    edges.insert("c".to_string(), vec![]);
    edges.insert("d".to_string(), vec!["c".to_string()]);
    let graph = ComputationGraph {
        id: Uuid::new_v4(),
        nodes,
        edges,
        root_nodes: ["a".to_string(), "c".to_string()].into_iter().collect(),
        // "a" is marked a leaf/published-output DESPITE "b" being a
        // real internal consumer -- the multi-output shape.
        leaf_nodes: ["a".to_string(), "d".to_string()].into_iter().collect(),
        metadata: GraphMetadata {
            name: "leaf_reuse_safety".to_string(),
            node_count: 4,
            edge_count: 2,
            max_depth: 3,
            estimated_memory_usage: 110,
            estimated_flops: 0,
            created_at: chrono::Utc::now(),
        },
    };
    let graph_id = graph.id;
    analyzer.add_graph(graph).expect("add_graph");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let dataflow = analysis.dataflow_analysis.expect("dataflow analysis enabled");

    assert!(
        dataflow
            .memory_reuse_opportunities
            .iter()
            .all(|o| !o.reusable_variables.contains(&"a".to_string())),
        "\"a\" is a leaf (published output) despite having an internal consumer -- it must \
         never be proposed for reuse even though its lifetime math alone would otherwise \
         make it a valid non-overlapping candidate with \"c\", got {:?}",
        dataflow.memory_reuse_opportunities
    );
}

#[test]
fn test_variable_lifetime_death_node_is_real_last_consumer_not_hashmap_order() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let operations = vec![
        ("a".to_string(), OperationType::Add, vec![]),
        ("b".to_string(), OperationType::Add, vec!["a".to_string()]),
        ("c".to_string(), OperationType::Add, vec!["a".to_string()]),
        (
            "d".to_string(),
            OperationType::Add,
            vec!["b".to_string(), "c".to_string()],
        ),
    ];
    let graph_id = analyzer.create_graph("lifetime".to_string(), operations).expect("create");
    let analysis = analyzer.analyze_graph(graph_id).expect("analyze");
    let dataflow = analysis.dataflow_analysis.expect("dataflow analysis enabled");

    let a_lifetime = dataflow.variable_lifetimes.get("a").expect("a has a lifetime entry");
    assert_eq!(a_lifetime.birth_node, "a");
    // a has two consumers (b, c) at the SAME topo depth; death_node
    // must be whichever one is really last in topo order, and
    // usage_nodes must list both real consumers -- never a
    // HashMap-iteration-order artifact.
    assert!(
        a_lifetime.death_node == "b" || a_lifetime.death_node == "c",
        "death_node must be a real consumer of a, got {}",
        a_lifetime.death_node
    );
    let mut usage = a_lifetime.usage_nodes.clone();
    usage.sort();
    assert_eq!(usage, vec!["b".to_string(), "c".to_string()]);
}

// ── Wave 6d: shape-driven estimates ──────────────────────────────────────────

/// `create_graph` has no shapes to work with, so every shape-derived estimate
/// must be absent. It used to publish `1_000_000` FLOPs for every MatMul,
/// `1_024` bytes for every node and `Some(1_000_000)` parameters -- constants
/// that no caller could distinguish from a measurement.
#[test]
fn test_create_graph_without_shapes_reports_absent_estimates() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let graph_id = analyzer
        .create_graph(
            "shapeless".to_string(),
            vec![
                ("x".to_string(), OperationType::MatMul, vec![]),
                ("y".to_string(), OperationType::ReLU, vec!["x".to_string()]),
            ],
        )
        .expect("graph creation should succeed");

    let graph = analyzer.graphs.get(&graph_id).expect("graph exists");
    for node in graph.nodes.values() {
        assert_eq!(node.flop_count, None, "node {} FLOPs", node.id);
        assert_eq!(node.memory_usage, None, "node {} memory", node.id);
        assert_eq!(node.parameter_count, None, "node {} parameters", node.id);
    }
    assert_eq!(graph.metadata.estimated_flops, 0);
    assert_eq!(graph.metadata.estimated_memory_usage, 0);
}

/// With real shapes the estimates are real arithmetic on those shapes.
#[test]
fn test_create_graph_with_shapes_computes_real_estimates() {
    let mut analyzer = ComputationGraphAnalyzer::default();
    let graph_id = analyzer
        .create_graph_with_shapes(
            "shaped".to_string(),
            vec![
                OperationSpec {
                    node_id: "proj".to_string(),
                    operation_type: OperationType::MatMul,
                    dependencies: vec![],
                    // [batch=8, in=16] x [in=16, out=32]
                    input_shapes: vec![vec![8, 16], vec![16, 32]],
                    output_shapes: vec![vec![8, 32]],
                },
                OperationSpec {
                    node_id: "act".to_string(),
                    operation_type: OperationType::ReLU,
                    dependencies: vec!["proj".to_string()],
                    input_shapes: vec![vec![8, 32]],
                    output_shapes: vec![vec![8, 32]],
                },
                OperationSpec {
                    node_id: "norm".to_string(),
                    operation_type: OperationType::LayerNorm,
                    dependencies: vec!["act".to_string()],
                    input_shapes: vec![vec![8, 32]],
                    output_shapes: vec![vec![8, 32]],
                },
            ],
        )
        .expect("graph creation should succeed");

    let graph = analyzer.graphs.get(&graph_id).expect("graph exists");
    let proj = graph.nodes.get("proj").expect("node exists");
    assert_eq!(proj.flop_count, Some(2 * 8 * 16 * 32), "2*m*k*n");
    assert_eq!(
        proj.memory_usage,
        Some((8 * 16 + 16 * 32) as u64 * 4),
        "both operands, 4 bytes per f32 element"
    );
    assert_eq!(
        proj.parameter_count,
        Some(16 * 32),
        "the weight matrix's elements"
    );

    let act = graph.nodes.get("act").expect("node exists");
    assert_eq!(act.flop_count, Some(8 * 32));
    assert_eq!(act.memory_usage, Some(8 * 32 * 4));
    assert_eq!(act.parameter_count, None, "ReLU learns nothing");

    let norm = graph.nodes.get("norm").expect("node exists");
    assert_eq!(norm.flop_count, Some(8 * 32 * 5));
    assert_eq!(
        norm.parameter_count,
        Some(2 * 32),
        "one scale and one shift per feature"
    );

    // Two different MatMul sizes must not produce the same estimate, which the
    // old constant fallback guaranteed they would.
    let mut other = ComputationGraphAnalyzer::default();
    let other_id = other
        .create_graph_with_shapes(
            "bigger".to_string(),
            vec![OperationSpec {
                node_id: "proj".to_string(),
                operation_type: OperationType::MatMul,
                dependencies: vec![],
                input_shapes: vec![vec![8, 16], vec![16, 64]],
                output_shapes: vec![vec![8, 64]],
            }],
        )
        .expect("graph creation should succeed");
    let bigger = other
        .graphs
        .get(&other_id)
        .and_then(|g| g.nodes.get("proj"))
        .expect("node exists");
    assert_ne!(bigger.flop_count, proj.flop_count);
    assert_ne!(bigger.parameter_count, proj.parameter_count);
}
