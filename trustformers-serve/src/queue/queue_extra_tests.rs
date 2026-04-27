#![cfg(test)]
/// Extended tests for the priority request queue module.
use super::*;

fn req(id: u64, p: RequestPriority) -> QueuedRequest {
    QueuedRequest::new(id, "model", "prompt", 128, p)
}

// ── 39. RequestPriority ordering — Critical > High > Normal > Low ─────────
#[test]
fn test_priority_ordering() {
    assert!(RequestPriority::Critical > RequestPriority::High);
    assert!(RequestPriority::High > RequestPriority::Normal);
    assert!(RequestPriority::Normal > RequestPriority::Low);
}

// ── 40. RequestPriority::name — returns non-empty strings ─────────────────
#[test]
fn test_priority_names_non_empty() {
    let names = [
        RequestPriority::Low.name(),
        RequestPriority::Normal.name(),
        RequestPriority::High.name(),
        RequestPriority::Critical.name(),
    ];
    for name in &names {
        assert!(!name.is_empty());
    }
}

// ── 41. RequestPriority::weight — Critical has largest weight ─────────────
#[test]
fn test_priority_weight_ordering() {
    assert!(RequestPriority::Critical.weight() > RequestPriority::High.weight());
    assert!(RequestPriority::High.weight() > RequestPriority::Normal.weight());
    assert!(RequestPriority::Normal.weight() > RequestPriority::Low.weight());
}

// ── 42. RequestPriority::weight — Critical is 8× Low ─────────────────────
#[test]
fn test_priority_weight_ratio() {
    let ratio = RequestPriority::Critical.weight() / RequestPriority::Low.weight();
    assert_eq!(ratio, 8);
}

// ── 43. QueuedRequest::new — fields match arguments ───────────────────────
#[test]
fn test_queued_request_new_fields() {
    let r = QueuedRequest::new(42, "my-model", "hello", 256, RequestPriority::High);
    assert_eq!(r.id, 42);
    assert_eq!(r.model, "my-model");
    assert_eq!(r.prompt, "hello");
    assert_eq!(r.max_tokens, 256);
    assert_eq!(r.priority, RequestPriority::High);
    assert!(!r.cancelled);
}

// ── 44. QueuedRequest::with_deadline_ms — attaches deadline ──────────────
#[test]
fn test_with_deadline_ms_attaches() {
    let r = req(1, RequestPriority::Normal).with_deadline_ms(500);
    assert_eq!(r.deadline_ms, Some(500));
}

// ── 45. QueuedRequest::with_estimated_tps — attaches tps ─────────────────
#[test]
fn test_with_estimated_tps_attaches() {
    let r = req(1, RequestPriority::Normal).with_estimated_tps(150.0);
    assert!((r.estimated_tps.unwrap_or(0.0) - 150.0).abs() < 1e-5);
}

// ── 46. PriorityQueue::push — capacity exceeded returns error ─────────────
#[test]
fn test_push_over_capacity_returns_error() {
    let mut q = PriorityQueue::new(2);
    q.push(req(1, RequestPriority::Normal)).expect("first ok");
    q.push(req(2, RequestPriority::Normal)).expect("second ok");
    let err = q.push(req(3, RequestPriority::Normal));
    assert!(err.is_err(), "should return error when at capacity");
}

// ── 47. PriorityQueue::pop — returns highest priority first ──────────────
#[test]
fn test_pop_highest_priority_first() {
    let mut q = PriorityQueue::new(16);
    q.push(req(1, RequestPriority::Low)).expect("push");
    q.push(req(2, RequestPriority::Critical)).expect("push");
    q.push(req(3, RequestPriority::Normal)).expect("push");
    let first = q.pop().expect("pop");
    assert_eq!(first.priority, RequestPriority::Critical);
}

// ── 48. PriorityQueue::pop — None when empty ──────────────────────────────
#[test]
fn test_pop_none_when_empty() {
    let mut q = PriorityQueue::new(8);
    assert!(q.pop().is_none());
}

// ── 49. PriorityQueue::pop — skips cancelled entries ─────────────────────
#[test]
fn test_pop_skips_cancelled() {
    let mut q = PriorityQueue::new(8);
    q.push(req(1, RequestPriority::Critical)).expect("push");
    q.push(req(2, RequestPriority::Normal)).expect("push");
    q.cancel(1);
    let popped = q.pop().expect("pop");
    assert_eq!(popped.id, 2, "cancelled entry must be skipped");
}

// ── 50. PriorityQueue::cancel — returns true for known id ─────────────────
#[test]
fn test_cancel_returns_true_for_known_id() {
    let mut q = PriorityQueue::new(8);
    q.push(req(10, RequestPriority::High)).expect("push");
    assert!(q.cancel(10));
}

// ── 51. PriorityQueue::cancel — returns false for unknown id ──────────────
#[test]
fn test_cancel_returns_false_for_unknown_id() {
    let mut q = PriorityQueue::new(8);
    assert!(!q.cancel(999));
}

// ── 52. PriorityQueue::peek — highest priority without removal ────────────
#[test]
fn test_peek_returns_highest_priority_without_removing() {
    let mut q = PriorityQueue::new(8);
    q.push(req(1, RequestPriority::Low)).expect("push");
    q.push(req(2, RequestPriority::High)).expect("push");
    let peeked = q.peek().expect("peek");
    assert_eq!(peeked.priority, RequestPriority::High);
    // Peek must not remove the element.
    assert_eq!(q.len(), 2);
}

// ── 53. PriorityQueue::gc — removes cancelled entries ─────────────────────
#[test]
fn test_gc_removes_cancelled_entries() {
    let mut q = PriorityQueue::new(16);
    for id in 1u64..=5 {
        q.push(req(id, RequestPriority::Normal)).expect("push");
    }
    q.cancel(2);
    q.cancel(4);
    let removed = q.gc();
    assert!(removed >= 2, "gc must remove at least 2 cancelled entries");
}

// ── 54. stats — total_enqueued grows with each push ──────────────────────
#[test]
fn test_stats_total_enqueued_grows() {
    let mut q = PriorityQueue::new(32);
    for id in 0u64..5 {
        q.push(req(id, RequestPriority::Normal)).expect("push");
    }
    let s = q.stats();
    assert_eq!(s.total_enqueued, 5);
}

// ── 55. stats — total_dequeued grows with each pop ───────────────────────
#[test]
fn test_stats_total_dequeued_grows() {
    let mut q = PriorityQueue::new(32);
    q.push(req(1, RequestPriority::Normal)).expect("push");
    q.push(req(2, RequestPriority::Normal)).expect("push");
    q.pop();
    let s = q.stats();
    assert_eq!(s.total_dequeued, 1);
}

// ── 56. display for RequestPriority ───────────────────────────────────────
#[test]
fn test_priority_display() {
    assert_eq!(RequestPriority::Low.to_string(), "low");
    assert_eq!(RequestPriority::Critical.to_string(), "critical");
}

// ── 57. PriorityQueue::is_empty — true initially ─────────────────────────
#[test]
fn test_priority_queue_is_empty_initially() {
    let q = PriorityQueue::new(8);
    assert!(q.is_empty());
}

// ── 58. PriorityQueue::is_empty — false after push ───────────────────────
#[test]
fn test_priority_queue_not_empty_after_push() {
    let mut q = PriorityQueue::new(8);
    q.push(req(1, RequestPriority::Low)).expect("push");
    assert!(!q.is_empty());
}
